"""
MDMM x SIMPLE router: JOINT slice+threshold discovery under an anti-collapse
correlation constraint -- the constrained sibling of run_simplerouter_discovery.py.

Motivation (Harshul's ADC_effect_training campaigns, audited 2026-07-27): on
contained data the plain objective finds the healthy (angles-tracking) basin
~1 seed in 5; wrapping the model in MDMM with a Pearson-correlation floor
(corr(pred, true) >= 0.5 per regressed parameter) makes it the default
outcome (5/6 runs, angle corr ~0.99). Question here: with angles FORCED to
work, does the router's slice choice migrate away from the late amplitude
window (73-89, anchor 78) toward the analog-timing region (~11-31)?

Model:   MDMM( ViT_Max_SimpleRouter ) = Input(16,16,101) -> SimpleRouterLayer
         -> SoftQuantizeLayer -> pre-LN ViT backbone (14 outputs, full-cov NLL)
         + 4x MinCorrConstraint (x,y,cotA,cotB; corr >= 0.5; scale 1e4,
         damping 1, full batch) evaluated on the PRIMARY training pass
         (constraint_pass='primary'). NOT Harshul's second deterministic pass:
         that pass dead-ends at SoftQuantizeLayer's eval-mode stop_gradient
         (router theta exactly frozen, observed run-1 2026-07-28), measures the
         constraint on the fixed top-2 view while weights train on sampled
         pairs (corr stalls), and its dropout-free train-batch corr is
         satisfiable by memorization (train corr >= 0.5, val corr 0.17, all
         lambdas frozen from epoch ~85 -> total fixed point). On the primary
         pass the corr floor is un-gameable (dropout decorrelates) and the
         penalty gradient reaches theta through the STE + sampled pair.
         Run-1 archived: runs/simplerouter_mdmm_discovery/archive_run1_frozen.
Anneal:  ONE AnnealingScheduler (quantizer only, cosine k:1->67), unchanged.
Data:    TFR_files_all101_noise_contained_discovery -- SAME data as the
         unconstrained SIMPLE study, so MDMM is the only changed variable.
Seeds:   same candidate pool as the unconstrained study -> per-seed pairing.
val_loss: plain NLL (MDMM test_step untouched) -> directly comparable to the
         unconstrained runs' -28K plateau.
Logs:    router_epochs.csv + theta_mu.npz (as before) + mdmm_epochs.csv
         (per-epoch lambdas, pred corr, pred std on one cached val batch).

Usage:
  python run_simplerouter_mdmm_discovery.py --sanity          # 3-epoch GPU check
  python run_simplerouter_mdmm_discovery.py --epochs 1000 --target 6
"""
import os, sys, json, time, glob, random, argparse, csv, re, traceback, math

# portable paths: derive from this file's location so the same script runs on
# Purdue AF and on Gautschi (where /work/... does not exist). Override the data
# location with SMARTPIX_TFR (or SMARTPIX_DATA_BASE) when it is staged elsewhere,
# e.g. /depot/cms/users/das214/SmartPixels_gautschi.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
HELPERS = os.environ.get("SMARTPIX_HELPERS",
                         os.path.join(_REPO, "two_bit_optimization_helpers"))
sys.path.insert(0, HELPERS)

import numpy as np
import tensorflow as tf
# SMARTPIX_GPU_MEM_MB caps TF to a fixed pool instead of letting it grow.
# Needed on a DEDICATED GPU: with plain memory-growth on a whole 46 GB L40S the
# run OOM'd in MultiHeadAttention's gradient einsum at epoch 311, and again at
# 563 even with TF_GPU_ALLOCATOR=cuda_malloc_async -- growth keeps handing the
# allocator more memory to fragment. The same code is stable for 5000 epochs on
# AF, where 4 co-tenant workers implicitly bound each other. A hard pool forces
# reuse. Unset (the AF default) keeps the original memory-growth behaviour.
_MEM_MB = os.environ.get("SMARTPIX_GPU_MEM_MB")
for g in tf.config.list_physical_devices("GPU"):
    try:
        if _MEM_MB:
            tf.config.set_logical_device_configuration(
                g, [tf.config.LogicalDeviceConfiguration(memory_limit=int(_MEM_MB))])
        else:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception: pass

# SMARTPIX_CHECK_NUMERICS=1: insert CheckNumerics after every op so the FIRST
# inf/nan tensor raises with the op that produced it (runs are seed-bit-exact,
# so a NaN reproduces at the same epoch). Debug only -- large slowdown.
if os.environ.get("SMARTPIX_CHECK_NUMERICS"):
    tf.debugging.enable_check_numerics()

from prepare_tfrecords import load_tfrecords
from train import create_model
from AnnealingScheduler import AnnealingScheduler
from loss import custom_loss
from conditional_nll import (custom_loss_split, custom_loss_v2, nll_terms,
                             binbalanced_loss_v2_byterm)
from mdmm import MDMM, MinCorrConstraint, MaxBlockNLLConstraint, ZeroBiasConstraint

BASE = os.environ.get(
    "SMARTPIX_DATA_BASE",
    "/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d")
TFR = os.environ.get("SMARTPIX_TFR",
                     os.path.join(BASE, "TFR_files_all101_noise_contained_discovery"))
TFR_TRAIN, TFR_TEST = os.path.join(TFR, "TFR_train"), os.path.join(TFR, "TFR_test")

N_SLICES = 101
THR_LOW, THR_HIGH = 25.0, 160.0          # random threshold window (threshold-study recipe)
LEVELS = np.array([0., 1., 2., 3.], dtype=np.float32)
ESCAPE_BELOW = 5e4                        # best val_loss below this = genuinely learned
SNAP = 25                                 # theta/mu/visits snapshot cadence (epochs)
ANNEAL_EPOCHS = 1000                      # k: 1->67 over this many epochs, then HOLD.
                                          # Pinned (not tied to --epochs) so 1000-ep and
                                          # 5000-ep runs share one schedule and a resumed
                                          # run never re-softens an annealed quantizer.

# --- MDMM config (Harshul's proven corr1e4 recipe, verbatim) ---
# Output layout is the 14-vector with means interleaved at 0,2,4,6 (same
# custom_loss lineage in both repos); labels are the normalized 4-vector.
MDMM_SCALE = 1e4          # at scale 1 the Nadam-capped lambda ascent never bites
MDMM_DAMPING = 1.0
MDMM_MIN_CORR = 0.5       # healthy runs sit at 0.98-0.996; 0.5 only evicts collapse
MDMM_OUTPUT_COLUMNS = {"x": 0, "y": 2, "cotA": 4, "cotB": 6}
MDMM_LABEL_COLUMNS = {"x": 0, "y": 1, "cotA": 2, "cotB": 3}
MDMM_CONSTRAINT_SAMPLES = None            # None = full batch (exact estimate)

# --- O12: constrain the conditional angle NLL instead of the correlation ------
# Off by default, so every existing/queued run behaves EXACTLY as before.
# ANGLE_NLL target is -log p(cotA,cotB | x,y) averaged over the batch, in the
# same units as the task loss (see conditional_nll.py). Set from measurement,
# not invented: runs/angle_block_measured.json holds the achievable frontier.
CONSTRAINT_MODE = os.environ.get('SMARTPIX_CONSTRAINT', 'corr')   # 'corr' | 'anglenll'
# Comma lists give ONE CONSTRAINT (and therefore one lambda) PER ENTRY, e.g.
#   SMARTPIX_NLL_BLOCK='cotA,cotB'  SMARTPIX_ANGLE_NLL_TARGET='-0.15,-2.20'
# O13 uses exactly that: a HARD target on cot alpha plus a LOOSE guard on cot
# beta. With a single constraint on the pair block (O12) the model satisfied it
# by improving cot beta while cot alpha got WORSE than O11's best seed
# (+0.207 vs -0.044) -- separate multipliers remove that trade in both
# directions, since neither angle can be paid for with the other.
NLL_BLOCK = os.environ.get('SMARTPIX_NLL_BLOCK', 'cotA')
ANGLE_NLL_TARGET = os.environ.get('SMARTPIX_ANGLE_NLL_TARGET', '0.0')
NLL_BLOCKS = [b.strip() for b in NLL_BLOCK.split(',') if b.strip()]
NLL_TARGETS = [float(t) for t in str(ANGLE_NLL_TARGET).split(',')]
if len(NLL_TARGETS) == 1 and len(NLL_BLOCKS) > 1:
    NLL_TARGETS = NLL_TARGETS * len(NLL_BLOCKS)
assert len(NLL_TARGETS) == len(NLL_BLOCKS), \
    f'need one target per block: {NLL_BLOCKS} vs {NLL_TARGETS}'
# Events per batch, from the TFRecord MANIFEST (batch = 5000). Only used to put
# the per-event constraint on the same footing as the summed task loss.
BATCH = int(os.environ.get('SMARTPIX_BATCH', '5000'))
# O20 warm-start probe: initialise backbone + head + thresholds from a trained
# checkpoint while the router keeps its fresh init. WARM_SRC_MODEL is the
# factory that WROTE the checkpoint (weight counts differ between router
# variants, e.g. smooth_sigma), not the factory being trained.
WARM_START = os.environ.get('SMARTPIX_WARM_START', '')
WARM_SRC_MODEL = os.environ.get('SMARTPIX_WARM_SRC_MODEL', 'ViT_MaxDeep_SimpleRouter')
# Loss v2 (O21.v2+): softplus diagonal, no clip -- see conditional_nll.py.
# DIAG_MODE follows the loss so every diagnostic that interprets the raw diag
# outputs (MDMMStateLogger's nll columns) reads the dialect the model is
# actually being trained in; cross-dialect scoring is ~7 nats/event of pure
# artifact. val_loss/history numbers under v2 are NOT comparable with the
# clip-dialect O11..O21 ledger.
LOSS_V2 = bool(os.environ.get('SMARTPIX_LOSS_V2', ''))

# --- O22: residual-bias removal (quasi-binning + per-bin zero-bias) --------
# SMARTPIX_ZEROBIAS=1   add one ZeroBiasConstraint per target, NBINS soft bins
#                       each, one multiplier per bin.
# SMARTPIX_BINBAL=byterm  swap the task loss for the by-term bin-balanced one.
#                       Each conditional term is balanced in ITS OWN target's
#                       bins, which is why no single per-event weight has to
#                       serve four different binnings.
# Both default OFF, so every existing recipe is byte-identical.
ZEROBIAS     = bool(os.environ.get('SMARTPIX_ZEROBIAS', ''))
BINBAL       = os.environ.get('SMARTPIX_BINBAL', '')
NBINS        = int(os.environ.get('SMARTPIX_NBINS', '15'))
# scale is BATCH-sized, like MaxBlockNLLConstraint, NOT MinCorr's 1e4. The task
# loss is a batch SUM, and the infeasibility is now dimensionless, so a
# batch-sized scale makes one unit of normalised bias cost about what one event
# of NLL costs. MinCorr needs 1e4 because correlation infeasibility is tiny and
# not commensurate with a loss; normalised bias is.
BIAS_SCALE   = float(os.environ.get('SMARTPIX_BIAS_SCALE', '0') or 0)
BIAS_DAMPING = float(os.environ.get('SMARTPIX_BIAS_DAMPING', '1.0'))
# tol in units of the target's residual spread; 0.06 ~ 3x the statistical error
# on a bin mean, i.e. "consistent with zero". max_lambda is the backstop.
BIAS_TOL     = float(os.environ.get('SMARTPIX_BIAS_TOL', '0.17'))
# SMARTPIX_BIAS_START=N  hold the zero-bias term OFF until epoch N. From a
# cold start the constraint is unsatisfiable -- see BiasGateCallback.
BIAS_START   = int(os.environ.get('SMARTPIX_BIAS_START', '0') or 0)
BIAS_MAXLAM  = float(os.environ.get('SMARTPIX_BIAS_MAXLAM', '0') or 0)
BIAS_INFCAP  = float(os.environ.get('SMARTPIX_BIAS_INFCAP', '0') or 0)
FREEZE_THR   = bool(os.environ.get('SMARTPIX_FREEZE_THR', ''))
# How to hold a pair-lattice router fixed.
#   onehot  -- spike psi at the chosen pair. Readout is that pair, but training
#              then samples that SAME pair every step.
#   inherit -- keep the source checkpoint's learned psi and just make it
#              non-trainable. Readout is still argmax = the chosen pair, and the
#              training-time sampling is preserved exactly as the parent had it.
# 'onehot' looks like the stricter freeze and is a trap: the layer samples ONE
# pair per training step from softmax(phi) and only uses argmax at eval, so
# seed 22042 drew its winning pair on just 3.6% of steps and something else on
# the other 96.4%. That sampling is a very large augmentation; removing it made
# both O22 arms overfit within 200 epochs (control: train -47,218 -> -49,415
# while val went -39,581 -> -34,220).
PIN_MODE     = os.environ.get('SMARTPIX_PIN_MODE', 'onehot')
DIAG_MODE = 'softplus' if LOSS_V2 else 'relu'
# Ceiling on the effective Lagrange multiplier. '' = uncapped (O12/O13 original).
# O13 without a cap: lambda reached ~30, penalty ~80,000 vs a task loss of
# ~-13,000, and 2 of 3 seeds reversed cot alpha (-0.02 -> +0.38) and collapsed
# onto the adjacent pair [7,8]. 25 holds the pressure without swamping the loss.
LAMBDA_CAP = os.environ.get('SMARTPIX_LAMBDA_CAP', '')
LAMBDA_CAP = float(LAMBDA_CAP) if LAMBDA_CAP else None
# Ceiling on the infeasibility inside the penalty. The damping term scale*inf^2/2
# has no lambda in it, so LAMBDA_CAP cannot bound it: a seed starting on the
# clipped plateau has inf ~ 21 and a ~1.1e6 damping term that NaN'd seed 13442 on
# epoch 2. '' = uncapped.
INF_CAP = os.environ.get('SMARTPIX_INF_CAP', '')
INF_CAP = float(INF_CAP) if INF_CAP else None

# --- O15: annealed Gaussian kernel on the router logits ----------------------
# SMARTPIX_SMOOTH_SIGMA0=4 SMARTPIX_SMOOTH_EPOCHS=1500 turns it on. sigma goes
# sigma0 -> 0 on a cosine over SMOOTH_EPOCHS and is then held at EXACTLY 0, so
# the run finishes as a plain SimpleRouter. Measured on real checkpoints: at
# sigma=1 the sharp solutions become inexpressible, so the endpoint must be a
# true zero and the anneal must finish with epochs to spare.
SMOOTH_SIGMA0 = float(os.environ.get('SMARTPIX_SMOOTH_SIGMA0', '0') or 0)
# Explicit model override (e.g. ViT_MaxDeep_SimpleRouter for the O17 deep-head
# ablation). Takes precedence over the smooth/beta selection below. '' = off.
MODEL_NAME_OVERRIDE = os.environ.get('SMARTPIX_MODEL_NAME', '')
SMOOTH_EPOCHS = int(os.environ.get('SMARTPIX_SMOOTH_EPOCHS', '1500'))

# --- fixed-slice ablation ----------------------------------------------------
# SMARTPIX_FIX_SLICES='19,20' pins the router to one pair and FREEZES theta, so
# the run measures what that pair can support rather than what the router can
# find. Every run so far conflates the two: only 1 of 8 O11 seeds picked an early
# pair, and it beat every late-slice seed on cot alpha by 0.36 nats -- suggestive
# at n=1. theta = +30 on the two chosen slices makes the sampled pair
# deterministic (p ~ 1 - 99*exp(-30)) on the training path and the top-2 on the
# eval path, so both paths see exactly those slices.
FIX_SLICES = os.environ.get('SMARTPIX_FIX_SLICES', '')
FIX_PAIR = ([int(v) for v in FIX_SLICES.split(',')] if FIX_SLICES else None)
# clip=True reproduces loss.custom_loss bit-for-bit (density clipped to
# [1e-9,1e9]). clip=False is the untruncated NLL: the 0.16% worst-fit events stop
# being capped at 20.72, and the dead-gradient plateau at init disappears.
LOSS_CLIP = os.environ.get('SMARTPIX_LOSS_CLIP', '1') == '1'


class SmoothSigmaScheduler(tf.keras.callbacks.Callback):
    """sigma0 -> 0 on a cosine over `epochs`, then EXACTLY 0 for the rest.

    Cosine (not linear) so the width falls fast through the useless-wide region
    and lingers in the 0-1 range where the solution is actually resolved.

    Set from the ABSOLUTE epoch, so a resumed run continues the schedule instead
    of restarting it -- the same reason the quantizer anneal is pinned to a fixed
    horizon rather than to --epochs.
    """
    def __init__(s, router, sigma0, epochs):
        super().__init__(); s.r = router; s.s0 = float(sigma0); s.T = int(epochs)

    def _sigma(s, ep):
        if s.T <= 0 or ep >= s.T:
            return 0.0
        return s.s0 * 0.5 * (1.0 + math.cos(math.pi * ep / s.T))

    def on_epoch_begin(s, epoch, logs=None):
        s.r.smooth_sigma.assign([np.float32(s._sigma(epoch))])


class AbortOnStuck(tf.keras.callbacks.Callback):
    """Stuck = val_loss above `thr` AND not improving for `patience` epochs.

    thr=9.9e4: the clipped init plateau is val_loss = 99,113.2, so a threshold of
    1e5 sits just ABOVE it and a permanently-stuck seed is NEVER caught -- O11's
    seed_42 and O12's seed_12042 both sat there burning a GPU slot. 9.9e4 sits
    just below, so the plateau now counts as stuck while healthy seeds (which
    leave it within a few epochs, well inside `patience`) are untouched.
    NOT the unconstrained study's 1e4: at init the NLL is fully
    clipped (val ~99,113 -- likelihoods underflow the loss's 1e-9 clip, zero NLL
    gradient) and under MDMM the escape off that plateau is constraint-DRIVEN
    and gradual (corr rises epoch by epoch), not the unconstrained study's
    few-epoch dropout lottery. With thr=1e4 the clipped plateau itself counts as
    'bad' and every seed would be killed mid-escape at `patience` epochs. 1e5
    sits ABOVE the plateau (Harshul's exact setting, proven 5/6 on his MDMM
    campaign), so only true divergence (>1e5 or non-finite) aborts."""
    def __init__(s, thr=9.9e4, patience=20, min_delta=1.0):
        super().__init__(); s.thr=thr; s.pat=patience; s.min_delta=min_delta
        s.best=np.inf; s.bad=0; s.aborted=False
    def on_epoch_end(s, e, logs=None):
        v=(logs or {}).get("val_loss", np.inf)
        if not np.isfinite(v): s.bad+=1
        elif v < s.best - s.min_delta: s.best=v; s.bad=0
        elif v > s.thr: s.bad+=1
        else: s.bad=0
        if s.bad>=s.pat:
            print(f"[AbortOnStuck] val_loss {v:.1f} stuck -- aborting seed.")
            s.aborted=True; s.model.stop_training=True


class HoldAnnealingScheduler(AnnealingScheduler):
    """Anneal k over a FIXED horizon, then HOLD at final_k -- decoupled from
    fit(epochs=...). Needed once runs go past 1000 epochs: the stock scheduler
    stretches the cosine over the full fit length, so a 5000-epoch run would
    still be at k~7 by epoch 1000 (vs k=67 for the 1000-epoch runs) and a
    RESUMED run would reset an already-hardened quantizer back to soft.
    Fixing the horizon at ANNEAL_EPOCHS keeps every seed on one schedule, so
    'more epochs' is the only variable vs the 1000-epoch results."""
    def __init__(s, *a, anneal_epochs=1000, **kw):
        super().__init__(*a, **kw); s.anneal_epochs=int(anneal_epochs)
    def on_train_begin(s, logs=None):
        super().on_train_begin(logs)
        s.schedule_params['total_epochs'] = s.anneal_epochs
    def on_epoch_begin(s, epoch, logs=None):
        super().on_epoch_begin(min(epoch, s.anneal_epochs), logs)


class BiasGateCallback(tf.keras.callbacks.Callback):
    """Hold the zero-bias constraints off until `start`, then switch them on.

    O24's first attempt ran the constraint from epoch zero on a cold model and
    never learned the task: |bias|/sigma saturates inf_cap in every bin when the
    weights are random, so the penalty pinned at its 900,000 ceiling against an
    NLL of 40,411 and 97% of the gradient went into a constraint that could not
    be satisfied. Note that O22's own warm start WAS a warm-up -- the parent had
    already trained 10,000 unconstrained epochs before the constraint was added.

    Keyed off the ABSOLUTE epoch, so a resumed chunk re-establishes the right
    state instead of restarting the warm-up.
    """
    def __init__(s, constraints, start):
        super().__init__(); s.cons = list(constraints); s.start = int(start)
    def on_epoch_begin(s, epoch, logs=None):
        want = 1.0 if epoch >= s.start else 0.0
        for c in s.cons:
            if abs(float(c.gate.numpy()) - want) > 1e-6:
                c.gate.assign(want)
                print(f"[bias gate] epoch {epoch}: zero-bias constraints "
                      f"{'ENABLED' if want else 'held off'}", flush=True)


class DelayedHoldAnnealingScheduler(HoldAnnealingScheduler):
    """HoldAnnealingScheduler that also HOLDS AT THE INITIAL value until
    `start_epoch`. Used for the router's beta (option O4): sharpening from
    epoch 0 would freeze whatever slice ordering exists while the backbone is
    still incompetent at angles -- and since the NLL is position-dominated
    early, that ordering is the high-amplitude late slices. Delaying the ramp
    gives the exploration phase the NAS supernet literature prescribes before
    committing."""
    def __init__(s, *a, start_epoch=0, **kw):
        super().__init__(*a, **kw); s.start_epoch = int(start_epoch)
    def on_epoch_begin(s, epoch, logs=None):
        super().on_epoch_begin(max(0, epoch - s.start_epoch), logs)


class OptimizerStateCheckpoint(tf.keras.callbacks.Callback):
    """Persist the OPTIMIZER state (Nadam 1st/2nd moments + step counter) so a
    resumed run continues with a warm optimizer instead of a cold one.

    Weights alone are not enough: Nadam's moment estimates are what make its
    effective step size sane, and a cold restart re-derives them over tens of
    steps with oversized updates in between. Tolerable once; NOT tolerable on a
    12h-capped queue where a 5000-epoch run is chopped into ~8 chunks.

    Uses tf.train.Checkpoint, whose restore is DEFERRED by design: optimizer slot
    variables do not exist until the first apply_gradients, and the saved values
    are applied automatically at the moment they are created. That is why the
    restore is issued before fit() and no manual slot-building is needed.
    ~3 MB per write for this model, so writing every epoch is free."""
    def __init__(s, ckpt, path):
        super().__init__(); s.ckpt=ckpt; s.path=path
    def on_epoch_end(s, epoch, logs=None):
        try:
            s.ckpt.write(s.path)
        except Exception as e:                      # never kill a run over this
            print(f"[OptimizerStateCheckpoint] write failed: {e}", flush=True)


class SimpleRouterLogger(tf.keras.callbacks.Callback):
    """Per-epoch CSV of the SIMPLE router state + periodic theta/mu/visits snapshots."""
    def __init__(s, csv_path, npz_path, snap=SNAP, resume=False):
        super().__init__(); s.csv_path=csv_path; s.npz_path=npz_path; s.snap=snap
        s.snap_epochs=[]; s.snap_theta=[]; s.snap_mu=[]; s.snap_visits=[]
        s.phi_path=os.path.join(os.path.dirname(npz_path), 'phi_history.npz')
        s.snap_phi=[]; s.snap_phi_ep=[]
        # phi_history carries its OWN epoch list: snap_epochs restarts empty on
        # every chunk resume (theta_mu.npz is never written by these chains), so
        # deriving phi's epochs from it desynchronised the two arrays.
        if resume and os.path.exists(s.phi_path):      # keep prior phi snapshots
            try:
                z=np.load(s.phi_path)
                s.snap_phi=list(z['phi']); s.snap_phi_ep=list(z['epochs'])
            except Exception:
                s.snap_phi=[]; s.snap_phi_ep=[]
        if resume and os.path.exists(s.npz_path):     # keep prior snapshots
            try:
                z=np.load(s.npz_path)
                s.snap_epochs=list(z['epochs']); s.snap_theta=list(z['theta'])
                s.snap_mu=list(z['mu']);         s.snap_visits=list(z['visits'])
            except Exception: pass
        if resume and os.path.exists(s.csv_path) and os.path.getsize(s.csv_path)>0:
            return                                    # append to the existing log
        with open(s.csv_path,'w',newline='') as f:
            csv.writer(f).writerow(
                ['epoch','i1','i2','theta_top5_idx','theta_top5_val',
                 'mu_top5_idx','mu_top5_val','mu_entropy','visits_top5_idx','sigma'])
    def on_epoch_end(s, epoch, logs=None):
        r=s.model.get_layer('simple_router_output')
        th=r.theta.numpy().astype(np.float64)
        mu=r.mu_numpy().astype(np.float64)                 # exact marginals, sum=2
        vis=r.visits.numpy().astype(np.float64)
        i1,i2=r.selected_indices()
        tidx=np.argsort(th)[::-1][:5]
        midx=np.argsort(mu)[::-1][:5]
        vidx=np.argsort(vis)[::-1][:5]
        q=mu/mu.sum()                                       # normalized importance
        ent=float(-(q*np.log(q+1e-12)).sum())               # ln(101)=4.615 at uniform
        j=lambda v: ';'.join(str(x) for x in v)
        with open(s.csv_path,'a',newline='') as f:
            csv.writer(f).writerow(
                [epoch, i1, i2,
                 j(int(k) for k in tidx), j(f'{th[k]:.6g}' for k in tidx),
                 j(int(k) for k in midx), j(f'{mu[k]:.6g}' for k in midx),
                 f'{ent:.6g}', j(int(k) for k in vidx),
                 f'{float(r.sigma()):.4g}'])
        if epoch % s.snap == 0:
            s.snap_epochs.append(epoch)
            s.snap_theta.append(th.astype(np.float32))
            s.snap_mu.append(mu.astype(np.float32))
            s.snap_visits.append(vis.astype(np.float32))
            # PairLattice only: the FULL smoothed pair logits, so the learned
            # 2-D distribution can be replayed per epoch instead of only at the
            # end. Written incrementally (not in on_train_end, which never fires
            # on the Gautschi chunk chains) and appended across resumes; ~20 kB
            # per snapshot in float16, i.e. a few MB per 5000-epoch run.
            if hasattr(r, 'psi'):
                try:
                    phi = r.smooth(tf.convert_to_tensor(r.psi)).numpy().astype(np.float16)
                    s.snap_phi.append(phi); s.snap_phi_ep.append(epoch)
                    np.savez_compressed(s.phi_path,
                                        epochs=np.array(s.snap_phi_ep),
                                        phi=np.stack(s.snap_phi))
                except Exception:
                    pass
    def on_train_end(s, logs=None):
        if s.snap_epochs:
            np.savez_compressed(s.npz_path,
                epochs=np.array(s.snap_epochs),             # (S,)
                theta=np.stack(s.snap_theta),               # (S, 101)
                mu=np.stack(s.snap_mu),                     # (S, 101)
                visits=np.stack(s.snap_visits))             # (S, 101)


def pred_stats(inner_model, x, y):
    """(stds, corrs) of the deterministic predictions on one cached batch.

    This per-epoch diagnostic pass LEAKS ~122 MB/epoch on the Gautschi L40S --
    isolated by bisecting the callbacks (profile_leak.py): the leak is present in
    exactly the modes that run this pass, flat otherwise, and it survives the
    data-generator fix, so it is a SECOND independent leak.

    Passing `x` as a tf.Tensor rather than numpy is good practice but is NOT the
    cause: an A/B (modes no_router vs mdmm_tensor) leaked at an identical
    +122.1 MB/epoch either way. Root cause still unknown -- do not assume it is
    the input handoff.
    """
    preds = inner_model(x, training=False).numpy()
    stds, corrs = {}, {}
    for name, col in MDMM_OUTPUT_COLUMNS.items():
        p = preds[:, col]
        t = y[:, MDMM_LABEL_COLUMNS[name]]
        stds[name] = float(p.std())
        denom = p.std() * t.std() + 1e-6
        corrs[name] = float(np.mean((p - p.mean()) * (t - t.mean())) / denom)
    return stds, corrs


class MDMMStateLogger(tf.keras.callbacks.Callback):
    """Per-epoch lambdas + deterministic pred corr/std on one cached val batch --
    shows the constraints engaging (lambda rising while collapsed) and releasing
    (infeasibility -> 0). Mirrors Harshul's MDMMStateLoggerCallback."""
    def __init__(s, csv_path, constraints, inner_model, x, y, resume=False):
        super().__init__()
        s.csv_path=csv_path; s.constraints=constraints
        s.inner=inner_model; s.x=x; s.y=y
        names=list(MDMM_OUTPUT_COLUMNS)
        # A run started before the per-target NLL columns existed has a NARROWER
        # header. Appending the wider rows to it would silently corrupt the CSV,
        # so detect that and keep emitting the old width for those runs.
        # How many NLL columns the EXISTING header has: 0 (pre-dates them), 6
        # (raw only), or 12 (raw + clipped). Emitting more than the header holds
        # silently corrupts the CSV, and a run resumed under newer code must keep
        # writing its original width.
        s.n_nll = 12
        if resume and os.path.exists(s.csv_path) and os.path.getsize(s.csv_path)>0:
            with open(s.csv_path) as f:
                hdr = f.readline() or ''
            s.n_nll = 12 if 'nll_angle_c' in hdr else (6 if 'nll_angle' in hdr else 0)
            return                                    # append to the existing log
        with open(s.csv_path,'w',newline='') as f:
            csv.writer(f).writerow(
                ['epoch'] + [f'lmbda_{c.name}' for c in s.constraints] +
                [f'pred_corr_{n}' for n in names] + [f'pred_std_{n}' for n in names] +
                # the four conditional NLL terms: where the joint loss is actually
                # spent. Invisible in the fused scalar, and the quantity the O12
                # constraint bounds (nll_angle = nll_cotA + nll_cotB).
                ['nll_x','nll_y','nll_cotA','nll_cotB','nll_pos','nll_angle'] +
                # same per-event bound the loss and the constraint use. The RAW
                # columns show the degenerate regime (values ~1e36 while L's
                # diagonal is on its 1e-9 floor); the _c columns show what the
                # optimizer and lambda actually act on.
                ['nll_x_c','nll_y_c','nll_cotA_c','nll_cotB_c','nll_pos_c','nll_angle_c'])
    def on_epoch_end(s, epoch, logs=None):
        stds, corrs = pred_stats(s.inner, s.x, s.y)
        # A ZeroBiasConstraint carries one lambda PER BIN, so lmbda is a vector.
        # Log its MEAN here (the column stays one number per constraint, so every
        # existing reader keeps working) -- the per-bin detail that matters is the
        # bias itself, which the residual panels show directly.
        lmb=[]
        for c in s.constraints:
            _v = np.atleast_1d(c.lmbda.numpy())
            lmb.append(float(_v[0]) if _v.size == 1 else float(_v.mean()))
        row = ([epoch] + [f'{v:.6g}' for v in lmb] +
               [f'{corrs[n]:.6g}' for n in MDMM_OUTPUT_COLUMNS] +
               [f'{stds[n]:.6g}' for n in MDMM_OUTPUT_COLUMNS])
        if getattr(s, 'n_nll', 12):
            _raw = nll_terms(tf.cast(s.y, tf.float64),
                             tf.cast(s.inner(s.x, training=False), tf.float64),
                             diag_mode=DIAG_MODE).numpy()
            t = _raw.mean(0)
            row += [f'{t[0]:.6g}',f'{t[1]:.6g}',f'{t[2]:.6g}',f'{t[3]:.6g}',
                    f'{t[0]+t[1]:.6g}',f'{t[2]+t[3]:.6g}']
            if getattr(s, 'n_nll', 12) >= 12:
                LO, HI = -np.log(1e9), -np.log(1e-9)
                tc = np.clip(_raw, LO, HI).mean(0)
                pos_c = float(np.clip(_raw[:, 0] + _raw[:, 1], LO, HI).mean())
                ang_c = float(np.clip(_raw[:, 2] + _raw[:, 3], LO, HI).mean())
                row += [f'{tc[0]:.6g}',f'{tc[1]:.6g}',f'{tc[2]:.6g}',f'{tc[3]:.6g}',
                        f'{pos_c:.6g}',f'{ang_c:.6g}']
        with open(s.csv_path,'a',newline='') as f:
            csv.writer(f).writerow(row)


def last_logged_epoch(seed_dir):
    """Last epoch present in history.csv, or -1 if this seed has never run."""
    p=os.path.join(seed_dir,'history.csv')
    if not os.path.exists(p): return -1
    try:
        rows=[r for r in csv.DictReader(open(p)) if r.get('epoch','').isdigit()]
        return int(rows[-1]['epoch']) if rows else -1
    except Exception:
        return -1


def _pair_lattice_layer(model):
    """The pair-lattice router, or None.

    Found by ATTRIBUTE, not by name. It shares the name 'simple_router_output'
    with the 1-D router, so get_layer() finds it and the old FIX_SLICES branch
    then tries to assign a 101-vector to a 5050-entry psi. Testing for psi/ia/ib
    is what actually distinguishes the two."""
    for l in model.layers:
        if hasattr(l, "psi") and hasattr(l, "ia") and hasattr(l, "ib"):
            return l
    return None


def _labels_scale():
    """labels_scale from the dataset metadata -- needed to turn the stored
    cotangents into the degrees the residual plots are drawn in."""
    import json as _json
    try:
        m = _json.load(open(os.path.join(TFR_TEST, "metadata.json")))
        return np.asarray(m["labels_scale"], dtype=np.float64)
    except Exception:
        return np.ones(4)


def _bias_bin_specs(y_sample, nbins):
    """Soft-bin centres per target, in the space the bias is READ in.

    x and y stay in their own units; the angles are converted to degrees,
    because cot-space bias and degree-space bias are not the same thing once a
    bin is wide, and degrees is what the plot that started this shows.

    Returns (constraint_specs, loss_specs, describe) where
      constraint_specs: (out_col, label_col, centres, sigma, transform, scale)
      loss_specs:       (term_index, label_col, centres, sigma)   [native space]
    """
    sc = _labels_scale()
    names = ["x", "y", "cotA", "cotB"]
    out_cols = [0, 2, 4, 6]
    cons, loss, desc = [], [], []
    for k, nm in enumerate(names):
        v = np.asarray(y_sample[:, k], dtype=np.float64)
        if nm in ("cotA", "cotB"):
            ang = np.arctan2(1.0, v * sc[k]) * 180.0 / np.pi
            lo, hi = np.percentile(ang, [1, 99])
            cen = np.linspace(lo, hi, nbins).astype(np.float32)
            cons.append((out_cols[k], k, cen, float(cen[1] - cen[0]), "cot2deg", float(sc[k])))
            desc.append(f"{nm}: {lo:.1f}..{hi:.1f} deg")
        else:
            lo, hi = np.percentile(v, [1, 99])
            cen = np.linspace(lo, hi, nbins).astype(np.float32)
            cons.append((out_cols[k], k, cen, float(cen[1] - cen[0]), "identity", 1.0))
            desc.append(f"{nm}: {lo:.3f}..{hi:.3f}")
        # the loss weights bin in the NATIVE label space -- it only needs the
        # occupancy, and staying native keeps atan out of the hot loop
        lo2, hi2 = np.percentile(v, [1, 99])
        cen2 = np.linspace(lo2, hi2, nbins).astype(np.float32)
        loss.append((k, k, cen2, float(cen2[1] - cen2[0])))
    return cons, loss, "; ".join(desc)


def run_one_seed(seed, epochs, out_root, tg, vg, stamp, cbatch, beta=None):
    OUT=os.path.join(out_root, f'seed_{seed}'); os.makedirs(OUT, exist_ok=True)
    tf.keras.backend.clear_session()
    thr0=sorted(np.random.default_rng(seed).uniform(THR_LOW, THR_HIGH, 3).tolist())
    tf.random.set_seed(seed); np.random.seed(seed); random.seed(seed)

    # --- resume: weights + lambdas + append-mode logs -------------------------
    # 'last' (written every epoch) is the exact resume point; 'best' is the
    # fallback for runs that predate it (costs a few epochs of progress).
    prev_ep=last_logged_epoch(OUT)
    ck_last=os.path.join(OUT,'last.weights.hdf5'); ck_best=os.path.join(OUT,'best.weights.hdf5')

    def _usable_ckpt(p):
        """True only if p is a checkpoint we can actually load.

        A chunk killed mid-write (OOM, walltime, node eviction -- routine on the
        leaky Gautschi GPUs) leaves a 0-byte or truncated .hdf5. Resume then
        dies with `OSError: Unable to synchronously open file (file signature
        not found)` BEFORE the first epoch, the sbatch tail sees no progress and
        resubmits, and the seed crash-loops through the queue forever. Measured:
        both seed-22042 arms of the cold campaign burned queue slots this way.
        Validating here turns that into a clean fresh start.
        """
        if not p or not os.path.exists(p) or os.path.getsize(p) == 0:
            return False
        try:
            import h5py
            with h5py.File(p, 'r'):
                return True
        except Exception as e:
            print(f'  [resume] ignoring unreadable checkpoint {os.path.basename(p)}: {e}', flush=True)
            return False

    resume_ck = ck_last if _usable_ckpt(ck_last) else (ck_best if _usable_ckpt(ck_best) else None)
    resume = prev_ep >= 0 and resume_ck is not None
    if prev_ep >= 0 and resume_ck is None:
        print(f'  [resume] epoch {prev_ep} logged but no usable checkpoint -- restarting this seed from scratch', flush=True)
        prev_ep = -1
    initial_epoch = prev_ep+1 if resume else 0
    if resume and initial_epoch >= epochs:
        stamp(f"[seed {seed}] already at epoch {prev_ep} >= target {epochs} -- nothing to do")
        return None, None
    if resume:
        stamp(f"[seed {seed}] RESUMING from epoch {initial_epoch} "
              f"({os.path.basename(resume_ck)}) -> {epochs} ep")
    else:
        stamp(f"[seed {seed}] thr0={[round(t,1) for t in thr0]}  -> fitting (max {epochs} ep, MDMM corr>={MDMM_MIN_CORR})")

    if MODEL_NAME_OVERRIDE:
        model_name = MODEL_NAME_OVERRIDE
    elif SMOOTH_SIGMA0 > 0:
        model_name = 'ViT_Max_SimpleRouterSmooth'
    else:
        model_name = 'ViT_Max_SimpleRouterBeta' if beta else 'ViT_Max_SimpleRouter'
    vit=create_model(model_name, timeslices=N_SLICES, soft_quantize_layer=True,
                     initial_thresholds=thr0, threshold_offset=0.0, initial_levels=LEVELS)
    if FIX_PAIR is not None and PIN_MODE == 'onehot' and _pair_lattice_layer(vit) is not None:
        # The 1-D branch below drives 'simple_router_output'. A pair-lattice
        # router has no such layer -- its parameter is one logit per PAIR -- so
        # pinning it means spiking psi at the (i, j) entry, killing the
        # smoothing kernel, and making psi non-trainable.
        _pl = _pair_lattice_layer(vit)
        _i, _j = sorted(FIX_PAIR)
        _ia, _ib = _pl.ia.numpy(), _pl.ib.numpy()
        _hit = np.where((_ia == _i) & (_ib == _j))[0]
        if len(_hit) != 1:
            raise SystemExit(f"pair ({_i},{_j}) not found in the lattice")
        _psi = np.full(_ia.shape[0], -30.0, dtype=np.float32)
        _psi[_hit[0]] = 30.0
        _pl.psi.assign(_psi)
        _pl.psi._trainable = False
        if hasattr(_pl, 'smooth_sigma'):
            _pl.smooth_sigma.assign(np.zeros_like(_pl.smooth_sigma.numpy()))
        _chk = _pl.selected_indices()
        if _chk != [_i, _j]:
            raise SystemExit(f"pin failed: router still reads {_chk}")
        stamp(f"[seed {seed}] router PINNED to ({_i}, {_j}) and frozen "
              f"(psi trainable={_pl.psi.trainable}, sigma=0)")
    elif FIX_PAIR is not None and _pair_lattice_layer(vit) is None:
        # 1-D SimpleRouter only. A pair lattice under PIN_MODE=inherit is held
        # in the warm-start block instead; falling through to here would drive
        # .theta, which a lattice layer does not have.
        _r = vit.get_layer('simple_router_output')
        _v = np.full(N_SLICES, 0.0, dtype=np.float32)
        for _i in FIX_PAIR:
            _v[_i] = 30.0
        _r.theta.assign(_v)
        _r.trainable = False          # drops theta from trainable_variables
        print(f'  [fix-slices] router pinned to {sorted(FIX_PAIR)}, theta frozen', flush=True)

    if WARM_START and not resume:
        # The probe asks whether a DEVELOPED early-slice representation flips
        # the free router's preference, so the router must not inherit the
        # checkpoint's theta (O17 checkpoints carry theta pinned +30 on the
        # answer). Fresh starts only: chunk resumes reload their own
        # last.weights via the resume path above.
        # The main model is built first so its seed-determined init consumes
        # the RNG in the same order as every other campaign; the src model is
        # matched BY POSITION because Keras uniquifies auto-generated layer
        # names across consecutive builds in one session.
        src = create_model(WARM_SRC_MODEL, timeslices=N_SLICES, soft_quantize_layer=True,
                           initial_thresholds=thr0, threshold_offset=0.0, initial_levels=LEVELS)
        src.load_weights(WARM_START)
        assert len(src.layers) == len(vit.layers), \
            f'layer count mismatch: src {len(src.layers)} vs main {len(vit.layers)}'
        # 'simple_router_output' is the name BOTH router families answer to, but
        # the 1-D one parameterises theta and the pair lattice parameterises psi.
        _rt = vit.get_layer('simple_router_output')
        _is_lattice = _pair_lattice_layer(vit) is not None
        _rattr = 'theta' if hasattr(_rt, 'theta') else 'psi'
        theta_before = getattr(_rt, _rattr).numpy().copy()
        n_moved = 0
        for sl, ml in zip(src.layers, vit.layers):
            if ml.name.startswith('simple_router'):
                if PIN_MODE in ('inherit', 'inherit_free') and FIX_PAIR is not None and _is_lattice:
                    _w = sl.get_weights()
                    if _w:
                        ml.set_weights(_w)      # psi, visits, smooth_sigma
                continue
            w = sl.get_weights()
            if w:
                ml.set_weights(w)
                n_moved += 1
        theta_after = getattr(_rt, _rattr).numpy()
        if PIN_MODE == 'inherit_free' and FIX_PAIR is not None and _is_lattice:
            # O24b: inherit the source's psi but leave it TRAINABLE. 'inherit'
            # freezes the readout, which is what O22 needed to attribute its
            # result to the objective alone. Here the question is the opposite --
            # given a bias-corrected starting point, does the router MOVE off
            # (10,21) when the bias term is part of what it optimises? A fresh
            # psi (the default warm-start path) would not answer that: it would
            # feed a backbone trained on slices 10 and 21 a uniform draw over all
            # 5,050 pairs, re-creating the enormous initial bias that saturated
            # the constraint on the cold attempt.
            _pl = _pair_lattice_layer(vit)
            _sel = _pl.selected_indices()
            _phi = _pl.smooth(tf.convert_to_tensor(_pl.psi)).numpy()
            _pk = float(np.exp(_phi - _phi.max()).sum())
            _pk = float(np.max(np.exp(_phi - _phi.max()) / _pk))
            stamp(f"[seed {seed}] router INHERITED and left FREE: starts on "
                  f"{_sel}, psi trainable={_pl.psi.trainable}, sampler draws the "
                  f"incumbent {100*_pk:.1f}% of steps")
        elif PIN_MODE == 'inherit' and FIX_PAIR is not None and _is_lattice:
            _pl = _pair_lattice_layer(vit)
            _pl.psi._trainable = False
            _sel = _pl.selected_indices()
            if _sel != sorted(FIX_PAIR):
                raise SystemExit(f'inherit pin: source router reads {_sel}, '
                                 f'expected {sorted(FIX_PAIR)}')
            _phi = _pl.smooth(tf.convert_to_tensor(_pl.psi)).numpy()
            _pk = float(np.exp(_phi - _phi.max()).sum())
            _pk = float(np.max(np.exp(_phi - _phi.max()) / _pk))
            stamp(f"[seed {seed}] router INHERITED from the source and frozen: "
                  f"reads {_sel}, psi trainable={_pl.psi.trainable}, "
                  f"sampler still draws the winner {100*_pk:.1f}% of steps "
                  f"(the augmentation the one-hot pin would have destroyed)")
        elif PIN_MODE not in ('inherit', 'inherit_free'):
            assert np.array_equal(theta_before, theta_after), \
                f'router {_rattr} was touched by warm start'
        del src
        stamp(f"[seed {seed}] WARM START from {os.path.basename(WARM_START)} "
              f"({WARM_SRC_MODEL}): {n_moved} weighted layers transferred, router theta fresh")
    if FREEZE_THR:
        # AFTER the warm start on purpose. The quantizer is not a 'simple_router'
        # layer, so its thresholds ARE transferred from the source checkpoint --
        # freezing before the transfer would still end up frozen at the right
        # numbers, but would report thr0, which is a seed-derived random triple
        # and not what is actually being held.
        # Freeze ONLY threshold_deltas_raw: k is annealed by a scheduler that
        # assigns to it directly, and the levels are a separate calibration, so
        # freezing the layer wholesale would be a bigger change than was asked.
        _nfrozen, _thr_now = 0, 'unreadable'
        for _l in vit.layers:
            if hasattr(_l, 'threshold_deltas_raw'):
                _l.threshold_deltas_raw._trainable = False
                _nfrozen += 1
                try:
                    _thr_now = [round(float(v), 2)
                                for v in np.ravel(_l.thresholds.numpy())]
                except Exception as _e:
                    _thr_now = f'unreadable ({type(_e).__name__})'
        stamp(f"[seed {seed}] ADC thresholds FROZEN at {_thr_now} "
              f"({_nfrozen} quantizer layer(s))")

    if WARM_START and not resume:
        with open(os.path.join(OUT, 'warm_start.json'), 'w') as f:
            json.dump({'src': WARM_START, 'src_model': WARM_SRC_MODEL,
                       'layers_transferred': n_moved}, f)

    if CONSTRAINT_MODE == 'anglenll':
        # ONE constraint, on -log p(cotA,cotB | x,y). The four corr constraints are
        # dropped on purpose: this subsumes them (a collapsed angle has a bad
        # conditional NLL) and running both would confound the comparison. No
        # lambda cap -- if the target is beyond what two 2-bit slices support the
        # constraint stays infeasible and lambda climbs, which is the signal we
        # want to see rather than hide.
        # SCALE = batch size, NOT 1.0. The task loss is a SUM over the batch
        # (loss.py uses K.sum, which is why NLL reads ~-30,000 not ~-5.9) while
        # the constraint's fn is a per-event MEAN. Without this they differ by a
        # factor of BATCH and lambda would be pushing against a loss 5,000x its
        # size. Target stays in per-event nats so it is readable and matches
        # runs/angle_block_measured.json.
        constraints=[MaxBlockNLLConstraint(max_value=tv, block=bk,
                                           scale=float(BATCH), damping=MDMM_DAMPING,
                                           max_lambda=LAMBDA_CAP, inf_cap=INF_CAP,
                                           name=f'nll_{bk}')
                     for bk, tv in zip(NLL_BLOCKS, NLL_TARGETS)]
    else:
        constraints=[
            MinCorrConstraint(column=MDMM_OUTPUT_COLUMNS[p], label_column=MDMM_LABEL_COLUMNS[p],
                              min_value=MDMM_MIN_CORR, scale=MDMM_SCALE, damping=MDMM_DAMPING,
                              name=f"corr_{p}")
            for p in MDMM_OUTPUT_COLUMNS
        ]
    if ZEROBIAS:
        _cons_specs, _loss_specs, _desc = _bias_bin_specs(cbatch[1], NBINS)
        _names = ["x", "y", "cotA", "cotB"]
        _bscale = BIAS_SCALE if BIAS_SCALE > 0 else float(BATCH)
        for (_oc, _lc, _cen, _sg, _tf, _sc) in _cons_specs:
            constraints.append(
                ZeroBiasConstraint(column=_oc, label_column=_lc, centers=_cen,
                                   sigma=_sg, transform=_tf, label_scale=_sc,
                                   scale=_bscale, damping=BIAS_DAMPING,
                                   tol=BIAS_TOL,
                                   max_lambda=(BIAS_MAXLAM or None),
                                   inf_cap=(BIAS_INFCAP or None),
                                   name=f"zbias_{_names[_lc]}"))
        stamp(f"[seed {seed}] O22 ZERO-BIAS: {len(_cons_specs)} targets x {NBINS} "
              f"soft bins = {len(_cons_specs)*NBINS} multipliers, scale={_bscale:g} "
              f"(bias/sigma, tol={BIAS_TOL:g}, max_lambda={BIAS_MAXLAM or None}, "
              f"inf_cap={BIAS_INFCAP or None})")
        if BIAS_START > 0:
            for _c in constraints:
                if hasattr(_c, 'gate'):
                    _c.gate.assign(0.0)
            stamp(f"[seed {seed}] O22 ZERO-BIAS held OFF until epoch {BIAS_START} "
                  f"(cold start: the constraint is unsatisfiable on random weights)")
        stamp(f"[seed {seed}] O22 bin ranges -- {_desc}")
    model=MDMM(vit, constraints, constraint_samples=MDMM_CONSTRAINT_SAMPLES,
               constraint_pass='primary', name='mdmm_vit_router')
    # scale=1.0 for the NLL constraint: it is already in the task loss's units, so
    # the 1e4 fudge MinCorr needed (correlation and NLL are not commensurate) is
    # not just unnecessary here, it would swamp the objective.
    # clip=True -> use the ORIGINAL custom_loss verbatim, so an O12 run is an
    # apples-to-apples comparison with O11: identical objective, the constraint is
    # the only difference. (custom_loss_split(clip=True) is proven bit-identical
    # to it, but there is no reason to take even that risk.) clip=False switches
    # to the untruncated NLL and IS a change of objective.
    if LOSS_V2:
        # O21.v2+: softplus diag, no clip. Overrides LOSS_CLIP entirely.
        task_loss = custom_loss_v2
        stamp(f"[seed {seed}] LOSS V2 (softplus diag, unclipped) -- NLLs not "
              f"comparable with the clip-dialect O11..O21 ledger")
    else:
        task_loss = custom_loss if LOSS_CLIP else (lambda y, p: custom_loss_split(y, p, clip=False))
    if BINBAL == 'byterm':
        _bspecs = _bias_bin_specs(cbatch[1], NBINS)[1] if not ZEROBIAS else _loss_specs
        task_loss = (lambda y, p: binbalanced_loss_v2_byterm(y, p, _bspecs))
        stamp(f"[seed {seed}] O22 BIN-BALANCED loss (by-term, {NBINS} bins/target) "
              f"-- the TRAINING loss value is a weighted composite; use plain_nll "
              f"for anything comparable")

    # plain_nll: the UNWEIGHTED v2 loss, logged every epoch as a metric so the
    # run stays comparable to the ledger no matter what the training objective
    # is. v2 and not v1 on purpose -- these checkpoints are softplus-dialect,
    # and scoring them under v1's relu mapping is the ~7 nat/event artefact.
    def plain_nll(y, p):
        return custom_loss_v2(y, p)
    _metrics = [plain_nll] if LOSS_V2 else []
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3),
                  loss=task_loss, metrics=_metrics)

    if resume:
        model.load_weights(resume_ck)          # delegates to the inner ViT
        # Lambdas live on the wrapper, so they are NOT in the checkpoint; without
        # this they would restart at 0 and the angle constraint would have to
        # re-ratchet from scratch (~100 wasted epochs). Replay the last logged
        # values instead.
        mp=os.path.join(OUT,'mdmm_epochs.csv')
        if os.path.exists(mp):
            try:
                rows=[r for r in csv.DictReader(open(mp)) if r.get('epoch','').isdigit()]
                if rows:
                    for c in constraints:
                        v=rows[-1].get(f'lmbda_{c.name}')
                        if v not in (None,'','nan'): c.lmbda.assign(float(v))
                    stamp(f"[seed {seed}] lambdas restored: "
                          f"{ {c.name: round(float(c.lmbda.numpy()),4) for c in constraints} }")
            except Exception as e:
                stamp(f"[seed {seed}] lambda restore failed ({e}) -- starting from 0")

    # --- optimizer state (Nadam moments) + Lagrange multipliers, written every
    # epoch and restored on resume. The constraints are in the SAME checkpoint
    # because MDMM.save_weights delegates to the inner model, so the lambdas are
    # absent from the .hdf5; this restores them exactly, where the CSV replay
    # above is only 6 significant figures (and is kept as the fallback for runs
    # that predate this checkpoint).
    opt_ckpt = tf.train.Checkpoint(optimizer=model.optimizer,
                                   constraints=list(constraints))
    opt_path = os.path.join(OUT, 'opt_state')
    if resume and glob.glob(opt_path + '*'):
        try:
            opt_ckpt.read(opt_path).expect_partial()   # deferred: applies at slot creation
            stamp(f"[seed {seed}] optimizer state restored (warm Nadam moments)")
        except Exception as e:
            stamp(f"[seed {seed}] optimizer state restore failed ({e}) -- cold optimizer")
    elif resume:
        stamp(f"[seed {seed}] no optimizer state found -- cold optimizer "
              f"(expected for runs started before opt-state checkpointing)")

    cx, cy = cbatch
    # O15: drive the router's smoothing width. Only exists when the model was
    # built with smooth_logits=True, i.e. when SMARTPIX_SMOOTH_SIGMA0 > 0.
    smooth_cb=([SmoothSigmaScheduler(vit.get_layer('simple_router_output'),
                                     SMOOTH_SIGMA0, SMOOTH_EPOCHS)]
               if SMOOTH_SIGMA0 > 0 else [])
    # SMARTPIX_ABORT_THR: the default 9.9e4 is calibrated to the CLIP-dialect
    # plateau (val 99,113). Under loss v2 there is no cap and val legitimately
    # spikes past it (e.g. the UCB sweep scoring mismatched pairs), so v2 runs
    # set this huge -- the non-finite (NaN) abort path still protects them.
    stuck=AbortOnStuck(thr=float(os.environ.get('SMARTPIX_ABORT_THR', '9.9e4')))
    cbs=[SimpleRouterLogger(os.path.join(OUT,'router_epochs.csv'),
                            os.path.join(OUT,'theta_mu.npz'), resume=resume),
         MDMMStateLogger(os.path.join(OUT,'mdmm_epochs.csv'), constraints, vit, cx, cy,
                         resume=resume),
         tf.keras.callbacks.CSVLogger(os.path.join(OUT,'history.csv'), append=resume),
         # ONE annealer: the quantizer only (get_layer delegates through the wrapper).
         # Fixed 1000-epoch horizon then hold at k=67 -- see HoldAnnealingScheduler.
         HoldAnnealingScheduler('cosine', target_layer_name='soft_quantizer_output',
                                initial_k=1.0, final_k=67.0, verbose=0,
                                anneal_epochs=ANNEAL_EPOCHS),
         *([DelayedHoldAnnealingScheduler(
                'cosine', target_layer_name='simple_router_output',
                initial_k=1.0, final_k=beta['final'], verbose=0,
                anneal_epochs=beta['epochs'], start_epoch=beta['start'])] if beta else []),
         *([BiasGateCallback([c for c in constraints if hasattr(c, 'gate')],
                            BIAS_START)] if (ZEROBIAS and BIAS_START > 0) else []),
         *smooth_cb,
         # save_weights delegates to the inner ViT -> checkpoint loads into a
         # plain create_model for eval, exactly like the unconstrained study.
         # SMARTPIX_CKPT_MONITOR: which metric picks best.weights. Default val_loss
         # is the bin-balanced COMPOSITE; under O24b that chose a later epoch and
         # the plain-NLL optimum (-40.4K, epochs 2-6) was never written to disk.
         tf.keras.callbacks.ModelCheckpoint(os.path.join(OUT,'best.weights.hdf5'),
                                            save_weights_only=True,
                                            monitor=os.environ.get('SMARTPIX_CKPT_MONITOR','val_loss'),
                                            save_best_only=True),
         tf.keras.callbacks.ModelCheckpoint(os.path.join(OUT,'last.weights.hdf5'),
                                            save_weights_only=True, save_freq='epoch'),
         OptimizerStateCheckpoint(opt_ckpt, opt_path),
         stuck]
    t0=time.time()
    h=model.fit(tg, validation_data=vg, epochs=epochs, initial_epoch=initial_epoch,
                callbacks=cbs, shuffle=False, verbose=2)
    # best over the FULL history (this segment + any resumed prefix)
    vl=[]
    try:
        for r in csv.DictReader(open(os.path.join(OUT,'history.csv'))):
            v=r.get('val_loss','')
            if v not in ('','nan','val_loss'):
                try: vl.append(float(v))
                except ValueError: pass
    except Exception: pass
    if not vl: vl=h.history.get('val_loss',[np.inf])
    best=float(min(vl)); best_ep=int(np.argmin(vl))
    escaped=(not stuck.aborted) and (best < ESCAPE_BELOW)

    router=model.get_layer('simple_router_output'); quant=model.get_layer('soft_quantizer_output')
    mu=router.mu_numpy().astype(np.float64)
    idx=router.selected_indices()                        # ascending top-2 of theta
    mu_top5=[[int(k), float(mu[k])] for k in np.argsort(mu)[::-1][:5]]
    final_stds, final_corrs = pred_stats(vit, cx, cy)
    info={'seed':seed,'epochs':len(vl),'max_epochs':epochs,'best_epoch':best_ep,
          'escaped':escaped,'aborted_stuck':stuck.aborted,
          'init_thresholds':thr0,
          'final_indices':idx,
          'final_mu_top5':mu_top5,
          'final_slot_commitment':[float(mu[idx[0]]), float(mu[idx[1]])],
          'final_thresholds':[float(t) for t in np.array(quant.thresholds).ravel()],
          'best_val_loss':best,'final_val_loss':float(vl[-1]),
          'beta_anneal':beta,
          'mdmm':{'scale':MDMM_SCALE,'damping':MDMM_DAMPING,'min_corr':MDMM_MIN_CORR,
                  'constraint_samples':MDMM_CONSTRAINT_SAMPLES,
                  'constraint_pass':'primary',
                  # per-bin constraints carry a vector; keep the full list for those
            'final_lambdas':{c.name: (float(np.atleast_1d(c.lmbda.numpy())[0])
                                      if np.atleast_1d(c.lmbda.numpy()).size == 1
                                      else [round(float(v), 6) for v in
                                            np.atleast_1d(c.lmbda.numpy())])
                             for c in constraints},
                  'final_pred_corr':final_corrs,'final_pred_std':final_stds},
          'wall_sec':round(time.time()-t0),'data':TFR}
    json.dump(info, open(os.path.join(OUT,'result.json'),'w'), indent=1)
    del model, vit
    if not escaped:
        try: os.rename(OUT, OUT+'_STUCK')
        except OSError: pass
    return escaped, info


def scan_existing(out_root):
    conv=set(); attempted=set()
    for d in glob.glob(os.path.join(out_root,'seed_*')):
        base=os.path.basename(d)
        m=re.match(r'seed_(\d+)$', base)
        if m:
            s=int(m.group(1)); attempted.add(s)
            rj=os.path.join(d,'result.json')
            if os.path.exists(rj):
                try:
                    if json.load(open(rj)).get('escaped'): conv.add(s)
                except Exception: pass
            continue
        m2=re.match(r'seed_(\d+)_', base)
        if m2: attempted.add(int(m2.group(1)))
    return conv, attempted


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--target', type=int, default=6, help='converged seeds to collect')
    ap.add_argument('--out', default=os.path.join(_REPO, 'runs', 'simplerouter_mdmm_discovery'))
    ap.add_argument('--seeds', default=None, help='comma-sep candidate seeds')
    ap.add_argument('--sanity', action='store_true',
                    help='3-epoch single-seed GPU/memory/timing check, then exit')
    # --- O4: sharpen the pair distribution over training ---------------------
    ap.add_argument('--beta-final', type=float, default=None,
                    help='O4: anneal the router inverse temperature beta 1 -> this value '
                         '(cosine). Omit for the plain un-sharpened router. Uses the '
                         'ViT_Max_SimpleRouterBeta model, whose checkpoints are NOT '
                         'compatible with plain SimpleRouter runs -- use a separate --out.')
    ap.add_argument('--beta-start', type=int, default=1000,
                    help='O4: hold beta=1 until this epoch, then ramp (default 1000, i.e. '
                         'after the quantizer anneal finishes). Sharpening earlier freezes '
                         'the position-driven late-slice ordering.')
    ap.add_argument('--beta-epochs', type=int, default=2000,
                    help='O4: epochs over which beta ramps to --beta-final, then holds.')
    ap.add_argument('--extend', action='store_true',
                    help='resume already-converged seeds up to --epochs (ignores --target). '
                         'Use with --seeds to extend specific runs; run as a SEPARATE process '
                         'from the collecting campaign and keep the seed lists disjoint.')
    a=ap.parse_args()
    BETA = ({'final':a.beta_final, 'start':a.beta_start, 'epochs':a.beta_epochs}
            if a.beta_final else None)
    if a.sanity:
        a.epochs=3; a.target=1; a.out=os.path.join(a.out,'sanity')
        if BETA: BETA={**BETA, 'start':0, 'epochs':3}   # exercise the ramp in 3 epochs
    os.makedirs(a.out, exist_ok=True)
    log=open(os.path.join(a.out,'discovery.log'),'a',buffering=1)
    stamp=lambda m: log.write(f"[{time.strftime('%H:%M:%S')}] {m}\n") or log.flush() or print(m, flush=True)
    # same candidate pool as the unconstrained study -> per-seed pairing
    seeds=[int(s) for s in a.seeds.split(',')] if a.seeds else [42+1000*i for i in range(40)]

    try:
        gpus=tf.config.list_physical_devices("GPU")
        conv, attempted = scan_existing(a.out)
        stamp(f"MDMM x SIMPLE DISCOVERY collect target={a.target}: epochs={a.epochs} "
              f"thr window[{THR_LOW},{THR_HIGH}] GPUs={len(gpus)} "
              f"MDMM corr>={MDMM_MIN_CORR} scale={MDMM_SCALE:g} damping={MDMM_DAMPING:g} "
              f"full-batch constraint on PRIMARY pass ESCAPE_BELOW={ESCAPE_BELOW:g} "
              + (f"| O4 beta 1->{BETA['final']:g} over {BETA['epochs']} ep from ep {BETA['start']}"
                 if BETA else "| beta OFF (plain router)"))
        stamp(f"resume: {len(conv)} converged ({sorted(conv)}); {len(attempted-conv)} stuck/attempted")
        stamp(f"data: {TFR_TRAIN} (noise baked -> noise=-1)")
        if not a.sanity and not a.extend and len(conv) >= a.target:
            stamp("target already met -- nothing to do"); return
        tg,vg=load_tfrecords(TFR_TRAIN, TFR_TEST, noise=-1, seed=42)
        # One cached val batch for the per-epoch MDMM diagnostics. X is held as a
        # tf.Tensor (tidier than re-converting numpy every epoch), y stays numpy
        # for the correlation maths. NOTE: this does NOT fix the 122 MB/epoch
        # leak in that diagnostic pass -- an A/B showed numpy and tensor input
        # leak identically. See pred_stats().
        cx, cy = vg[0]
        cbatch=(tf.convert_to_tensor(np.asarray(cx)), np.asarray(cy))
        stamp(f"cached diagnostic batch: X{cbatch[0].shape} y{cbatch[1].shape}")
        for seed in seeds:
            if not a.extend and len(conv) >= a.target: break
            d=os.path.join(a.out, f'seed_{seed}')
            # A seed is revisitable if its own dir exists and is short of --epochs;
            # anything already renamed *_STUCK stays abandoned.
            unfinished = os.path.isdir(d) and last_logged_epoch(d) < a.epochs-1
            if seed in attempted and not unfinished: continue
            attempted.add(seed)
            escaped, info = run_one_seed(seed, a.epochs, a.out, tg, vg, stamp, cbatch, beta=BETA)
            if info is None: continue                      # already at target length
            if a.sanity:
                try:
                    mem=tf.config.experimental.get_memory_info('GPU:0')
                    stamp(f"SANITY GPU peak memory: {mem['peak']/2**30:.2f} GiB")
                except Exception as e:
                    stamp(f"SANITY GPU memory query unavailable: {e}")
                lam=info['mdmm']['final_lambdas']; cor=info['mdmm']['final_pred_corr']
                stamp(f"SANITY done: {info['wall_sec']}s for {info['epochs']} epochs "
                      f"(~{info['wall_sec']/max(info['epochs'],1):.1f}s/ep) "
                      f"indices={info['final_indices']} thr={[round(t,1) for t in info['final_thresholds']]} "
                      f"lambdas={ {k: round(v,4) for k,v in lam.items()} } "
                      f"corrs={ {k: round(v,3) for k,v in cor.items()} }")
                return
            if escaped:
                conv.add(seed)
                cor=info['mdmm']['final_pred_corr']
                stamp(f"[CONVERGED {len(conv)}/{a.target}] seed={seed} "
                      f"indices={info['final_indices']} thr={[round(t,2) for t in info['final_thresholds']]} "
                      f"best_val={info['best_val_loss']:.1f}@{info['best_epoch']} "
                      f"corrs={ {k: round(v,3) for k,v in cor.items()} } ({info['wall_sec']}s)")
                if not a.extend:      # an extend worker sees only its own seeds
                    json.dump({'target':a.target,'converged_seeds':sorted(conv)},
                              open(os.path.join(a.out,'WINNERS.json'),'w'), indent=1)
            else:
                stamp(f"[stuck] seed={seed} best_val={info['best_val_loss']:.1f} "
                      f"(epochs={info['epochs']}) -- next seed")
        if len(conv) >= a.target:
            stamp(f"DONE: {len(conv)} converged seeds: {sorted(conv)}")
        else:
            stamp(f"EXHAUSTED candidates with {len(conv)}/{a.target} converged")
    except Exception:
        traceback.print_exc()
        open(os.path.join(a.out,'FAILED.txt'),'w').write(traceback.format_exc()); sys.exit(1)


main()
