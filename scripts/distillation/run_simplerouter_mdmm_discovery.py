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
import os, sys, json, time, glob, random, argparse, csv, re, traceback

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

from prepare_tfrecords import load_tfrecords
from train import create_model
from AnnealingScheduler import AnnealingScheduler
from loss import custom_loss
from mdmm import MDMM, MinCorrConstraint

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


class AbortOnStuck(tf.keras.callbacks.Callback):
    """Stuck = val_loss above `thr` AND not improving for `patience` epochs.

    thr=1e5 here, NOT the unconstrained study's 1e4: at init the NLL is fully
    clipped (val ~99,113 -- likelihoods underflow the loss's 1e-9 clip, zero NLL
    gradient) and under MDMM the escape off that plateau is constraint-DRIVEN
    and gradual (corr rises epoch by epoch), not the unconstrained study's
    few-epoch dropout lottery. With thr=1e4 the clipped plateau itself counts as
    'bad' and every seed would be killed mid-escape at `patience` epochs. 1e5
    sits ABOVE the plateau (Harshul's exact setting, proven 5/6 on his MDMM
    campaign), so only true divergence (>1e5 or non-finite) aborts."""
    def __init__(s, thr=1e5, patience=20, min_delta=1.0):
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
                 'mu_top5_idx','mu_top5_val','mu_entropy','visits_top5_idx'])
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
                 f'{ent:.6g}', j(int(k) for k in vidx)])
        if epoch % s.snap == 0:
            s.snap_epochs.append(epoch)
            s.snap_theta.append(th.astype(np.float32))
            s.snap_mu.append(mu.astype(np.float32))
            s.snap_visits.append(vis.astype(np.float32))
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
        if resume and os.path.exists(s.csv_path) and os.path.getsize(s.csv_path)>0:
            return                                    # append to the existing log
        with open(s.csv_path,'w',newline='') as f:
            csv.writer(f).writerow(
                ['epoch'] + [f'lmbda_{c.name}' for c in s.constraints] +
                [f'pred_corr_{n}' for n in names] + [f'pred_std_{n}' for n in names])
    def on_epoch_end(s, epoch, logs=None):
        stds, corrs = pred_stats(s.inner, s.x, s.y)
        lmb=[float(c.lmbda.numpy()) for c in s.constraints]
        with open(s.csv_path,'a',newline='') as f:
            csv.writer(f).writerow(
                [epoch] + [f'{v:.6g}' for v in lmb] +
                [f'{corrs[n]:.6g}' for n in MDMM_OUTPUT_COLUMNS] +
                [f'{stds[n]:.6g}' for n in MDMM_OUTPUT_COLUMNS])


def last_logged_epoch(seed_dir):
    """Last epoch present in history.csv, or -1 if this seed has never run."""
    p=os.path.join(seed_dir,'history.csv')
    if not os.path.exists(p): return -1
    try:
        rows=[r for r in csv.DictReader(open(p)) if r.get('epoch','').isdigit()]
        return int(rows[-1]['epoch']) if rows else -1
    except Exception:
        return -1


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
    resume_ck=ck_last if os.path.exists(ck_last) else (ck_best if os.path.exists(ck_best) else None)
    resume = prev_ep >= 0 and resume_ck is not None
    initial_epoch = prev_ep+1 if resume else 0
    if resume and initial_epoch >= epochs:
        stamp(f"[seed {seed}] already at epoch {prev_ep} >= target {epochs} -- nothing to do")
        return None, None
    if resume:
        stamp(f"[seed {seed}] RESUMING from epoch {initial_epoch} "
              f"({os.path.basename(resume_ck)}) -> {epochs} ep")
    else:
        stamp(f"[seed {seed}] thr0={[round(t,1) for t in thr0]}  -> fitting (max {epochs} ep, MDMM corr>={MDMM_MIN_CORR})")

    model_name = 'ViT_Max_SimpleRouterBeta' if beta else 'ViT_Max_SimpleRouter'
    vit=create_model(model_name, timeslices=N_SLICES, soft_quantize_layer=True,
                     initial_thresholds=thr0, threshold_offset=0.0, initial_levels=LEVELS)
    constraints=[
        MinCorrConstraint(column=MDMM_OUTPUT_COLUMNS[p], label_column=MDMM_LABEL_COLUMNS[p],
                          min_value=MDMM_MIN_CORR, scale=MDMM_SCALE, damping=MDMM_DAMPING,
                          name=f"corr_{p}")
        for p in MDMM_OUTPUT_COLUMNS
    ]
    model=MDMM(vit, constraints, constraint_samples=MDMM_CONSTRAINT_SAMPLES,
               constraint_pass='primary', name='mdmm_vit_router')
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3), loss=custom_loss)

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
    stuck=AbortOnStuck()
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
         # save_weights delegates to the inner ViT -> checkpoint loads into a
         # plain create_model for eval, exactly like the unconstrained study.
         tf.keras.callbacks.ModelCheckpoint(os.path.join(OUT,'best.weights.hdf5'),
                                            save_weights_only=True, monitor='val_loss',
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
                  'final_lambdas':{c.name: float(c.lmbda.numpy()) for c in constraints},
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
