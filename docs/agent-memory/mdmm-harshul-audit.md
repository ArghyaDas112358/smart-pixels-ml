---
name: mdmm-harshul-audit
description: "Audit of Harshul's MDMM repo (2026-07-27) — what works, the non-MDMM control, port notes for combining with SimpleRouter"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7a0a041e-b87d-47e5-ae22-8b5c100f4193
  modified: 2026-07-31T05:00:56.493Z
---

**Harshul's MDMM repo, audited 2026-07-27.** Public fork cloned (sparse: ADC_effect_training
+ models + losses + DG) to `/work/users/das214/SmartPixels/harshul-smart-pixels-ml`.
Source: https://github.com/guptaharshul24/smart-pixels-ml (dir `ADC_effect_training/mdmm/2ns5ns`).
Fun fact: `models/mdmm.py` header credits "das214's TF implementation" (the user's own
mdmm_tf_mnist code) as the original.

**His setup vs ours:** contained clusters + CORRELATED noise (ours: iid), slices [10,25]
= 2ns/5ns (ours [11,26]), same ViT/custom_loss lineage, means at output cols 0,2,4,6
(interleaved mean/diag — IDENTICAL to our repo, constraint columns port verbatim).
His env is Keras 3.14; ours Keras 2.15 → port needs: compiled_loss instead of
compute_loss, .ref() instead of id(), old add_weight signature (his docstring lists
exactly these — he ported FROM Keras 2 originally).

**The killer result (local minimum, NOT info ceiling):** on the SAME contained
corr-noise 2ns5ns data:
- non-MDMM Stage 1: 4/5 seeds stuck −25.8..−27.0K (collapsed); 1/5 (a43ed7b9) reached
  −37,833 healthy, angle corr 0.98-0.996 WITHOUT MDMM. The healthy basin exists;
  plain training is a ~20% lottery.
- MDMM (MinCorrConstraint corr≥0.5, scale 1e4, damping 1): 5/6 runs −36.7..−38.9K,
  cotA corr ~0.994. Lottery → default. Median thresholds [13.00, 21.90, 57.14] mV
  (close to our iid Part-1 [12.09, 22.72, 52.73]).
- Stage 1.5 (frozen hard thresholds): no-MDMM e7ddbec2 collapsed at −24.6K (corr ~0.03);
  MDMM 00bbfea6 −35,906 (corr 0.994/0.975). MDMM alone worth ~11K NLL at fixed everything.
→ REVISES our containment conclusion in [[threshold-opt-2_5-noise-contained]]: our
6/6 iid seeds at ~−28K and 3/3 clean-contained at ~−32.6K likely never found the healthy
basin. The "~20K containment cost" is substantially collapse-basin, not missing info.

**Constraint-gaming history (do NOT re-derive):** MinStd (std≥0.8·true) gamed by
outlier-salting (99.7% constant + 0.3% extremes, run 438bcf1c); MinMad gamed too
(dispersion with ZERO truth corr, runs eabfe9f3/4c28f1e8); MinCorr (Pearson≥0.5 per
param, truth-aware) forbids all observed cheats. Scale MUST be ~1e4: at scale 1 Nadam
caps lambda ascent at ~lr/step → lambda ratchets 5000 epochs, never enough pressure.
Constraints evaluated on a SECOND deterministic (training=False) forward pass —
dropout noise inflates output std 15-35% and silently satisfies spread constraints.
Lambdas NOT checkpointed (resume regrows them from 0). val_loss stays plain NLL
(comparable to non-MDMM runs).

**His QConv2D (Stage 2.5) stall — NOT lambda blowup:** 10/10 seeds stuck at the
IDENTICAL val ~98,980. Unquantized twin (Stage 2, same arch) converges attempt 1 to
−25,749. Leading hypothesis (his README, plausible): `quantized_bits(4,0,1,alpha=1)`
assumes weights fill [−1,1]; Glorot init is much smaller → crushed to ~0 at init, dead
network. Fix directions: trainable/auto alpha, warm-up before quantizing, or wider init.
Not our problem to fix unless asked.

**RUN 1 (2026-07-27 22:39, PID 4180194) FAILED-FROZEN, killed with user approval;
archived at `runs/simplerouter_mdmm_discovery/archive_run1_frozen/`.** Root cause,
two-layer: (1) `SoftQuantizeLayer.py:187` eval branch is `tf.stop_gradient(hard_q)` —
a hard gradient wall — so Harshul's deterministic (training=False) constraint pass
NEVER delivers gradient to anything above the quantizer → router theta exactly frozen
(uniform mu) for 250+ epochs; (2) the dropout-free train-batch corr was satisfied by
MEMORIZATION (train corr ≥0.5, true val corr 0.17) → all λs froze at epoch ~85 →
with the NLL clipped, total fixed point (loss byte-identical every epoch; AbortOnStuck
thr=1e5 correctly doesn't fire since plateau 99,113 < 1e5 — such a freeze must be
caught by eye/monitor, not the guard).

**RUN 2 (2026-07-28 02:35, user-approved) — the fix: `constraint_pass='primary'`** in
`mdmm.py` (new option; 'deterministic' kept for legacy/spread constraints): penalties
computed on the training=True y_pred of the main pass. For MinCorr this is strictly
safe (dropout DEcorrelates → floor un-gameable by memorization), the gradient reaches
theta through the STE + sampled pair (exact SIMPLE gradient of E[penalty]), measured
view == trained view, second pass eliminated (peak 4.87 GiB vs 7.56, ~46 s/ep).
Driver PID **1632784** on purdue-af-46 (A100), `--epochs 1000 --target 6`, out
`runs/simplerouter_mdmm_discovery/`. Smoke 11/11 incl. legacy-mode regression.
Sanity 3 ep: theta ALIVE and differentiating; early top slices [32,31,28,36,24] —
the TIMING REGION (vs unconstrained anchor 78) — suggestive but 3-epoch-only.
Readouts: (1) collapse broken (angle corr → ~0.99)? (2) μ migration 73-89 → ~11-31?
Kill by exact PID only.

**Run-2 course of events (2026-07-28 ~03:00): clip-death is SEED-DEPENDENT, campaign
self-corrects — no further intervention.** Seed 42 (clip-dead NLL from init): router
went straight to the timing region (top-5 [28,29,25,31,24], pair [28,29], corr cotA
0.30 climbing) but the corr constraint is SCALE-INVARIANT and with the NLL clipped
nothing bounds output scale → pred_std inflated to ~2.0 (4× true) → float32 NaN
cascade at epoch 20 → AbortOnStuck's NaN branch (patience 20) killed it at epoch 40.
That's the third clip-regime failure mode (gradient wall → memorization freeze →
unbounded growth). Seed 1042 drew thresholds [73.3,75.2,102.8] → NLL ALIVE at epoch 0
(loss_obj 28,248, val 20,866 by ep 1, already past the old −28K-plateau's frozen 99K
start); with a live NLL the scale runaway can't happen (NLL regularizes scale) and
the experiment runs in the Harshul regime. Doomed seeds cost ~30 min each then
self-abort; healthy seeds proceed. Early θ leaders on 1042: [70,61,74,68,84] (late,
NLL-driven) — the real contest is NLL-pulls-late vs λ_angle-pulls-early; θ's final
resting place is THE readout. If NaN attrition across seeds gets bad, the queued
hardening is: Nadam global_clipnorm=1.0 + MaxStdConstraint caps (2× label std) —
discussed, not applied, ask user first.

**FIRST CONVERGED SEED (2026-07-28 22:32, seed 1042, 19.4h/1000ep): indices [45,53]
— MID-WAVEFORM (9-10.6ns), thresholds [11.61, 33.66, 87.91] mV, best_val −27,494@983,
corrs x 0.994 / y 0.990 / cotA 0.594 / cotB 0.493 (cotB hovers at the 0.5 floor,
λ_cotB≈8.3 vs λ_cotA≈5).** Headline: with angles constrained, the router left the
unconstrained late window (73-89, anchor 78) for mid-pulse, at ~the same NLL as the
collapsed plateau (−28,004) but with angles tracking. θ profile = broad hump over
~35-63, negative before slice 10. Thresholds dove low-ish (T1≈11.6 — wing-pixel
sensitivity for cluster-shape/angle info). Driver auto-advanced to seed 2042.
NOTE: SoftQuantizeLayer thresholds = cumsum(EXPM1(threshold_deltas_raw)) (log1p
inverse, SoftQuantizeLayer.py:160-161), NOT softplus — the module header comment
(lines 10-12) says softplus and is WRONG; reconstruct from h5 with expm1.

**RUN 3 — 5000 EPOCHS, 3 PARALLEL WORKERS (2026-07-29 ~01:40, user asked "run them
for 5000 epochs + resume 1042"):** driver now supports RESUME (verified end-to-end
on a scratch 2ep→4ep test: continuous logs w/ single header, λs replayed from the
last mdmm_epochs.csv row, theta_mu.npz snapshots merged, new `last.weights.hdf5`
written every epoch = exact resume point; older runs fall back to best.weights).
New `HoldAnnealingScheduler` + `ANNEAL_EPOCHS=1000`: k anneals 1→67 over a FIXED
1000-epoch horizon then HOLDS — so 5000-ep runs share the 1000-ep schedule and a
resumed run never re-softens an annealed quantizer (the stock scheduler ties the
cosine to fit(epochs=), which would have reset seed 1042's k=67 back to ~7).
New `--extend` flag = resume even already-converged seeds, skip the target gate,
don't touch WINNERS.json. Workers (disjoint --seeds, same --out, shared
discovery.log): A `--seeds 1042 --extend` (1000→5000), B `--seeds 2042 --extend`
(resumed at ep 109), C `--seeds 3042,4042,...  --target 6` (fresh seeds).
Cost reality: ~70 s/ep solo → 5000 ep ≈ 4 days/seed; 6 seeds serial ≈ 1 month, hence
the parallelism (each run peaks 4.87 GiB on a 40 GB A100). Watch for GPU contention
slowing s/ep; killing extra workers is now cheap because resume works.

**SCALED TO 6 PARALLEL WORKERS 2026-07-29 04:45 (user: "add the remaining seeds
too").** Ownership rule: ONE seed per worker (`--seeds <s> --extend --target 1`),
never a shared queue — two workers must never touch the same seed dir. NOTE
`--extend` is REQUIRED even for fresh seeds: without it the shared-out-dir scan
finds converged 1042, sees len(conv) >= target, and exits with "target already met".
Workers: 1042(ext), 2042, 3042, 4042, 5042, 6042. Timing: 70 s/ep at 1 worker,
95 s/ep at 3 (2.2x throughput) — the job is input-bound (~517 MB/batch,
~16 GB/epoch), so extra runs fill idle GPU gaps. Host RAM 183/503 GB at 6 workers.
Beware `pgrep -f <pattern>` self-matching: filter args on the python path AND
remember the check string itself appears in your own shell's argv.

**HARD LIMIT = 4 CONCURRENT WORKERS on this A100 (learned 2026-07-29):** launching
6 OOM'd workers 5 and 6 on their FIRST train step — `OOM ... shape[5000,4,20,64]`
in MultiHeadAttention's einsum. The 4.87 GiB sanity figure is a SOLO steady-state
peak; transient attention tensors at batch 5000 push real usage well past it, so
6x4.87≈29 GB "fits in 40 GB" was wrong. 4 workers run at ~68 s/ep each — i.e.
SOLO speed, 4x throughput — because the job is input-bound; a 5th buys nothing.
Also: an OOM crash writes FAILED.txt into the SHARED --out root (confusing when
other workers are fine) — delete it after cleanup.

**WHY THE ANGLES STAY ~0.6 (diagnosed 2026-07-29, measured not guessed):** the
router NEVER COMMITS — participation ratio of μ = **55-64 effective slices** out of
101 even at epoch 2500 (top-2 μ mass only ~0.10 of 2.0). So one weight set + ONE
SoftQuantize threshold triple must serve ~60 amplitude regimes spanning 74→335 mV;
only the cluster centroid is invariant across that, hence x/y at 0.99 and angles
mean-reverting (perf plots: α/β residual slope ≈ +1, σα 55-62°, angle pulls ~1.3).
Proof it is NOT just slice choice: seed 3042 sits on [7,12] — the timing-rich early
region — and still only reaches 0.62. Thresholds are amplitude-locked to the μ
region: the 3 late seeds agree to ~1% ([12.2, 40.8, 98]) while early-slice 3042 sits
at [7.3, 15.6, 65.5]. Second cause: the NLL is position-dominated, so θ drifts late,
and a corr floor of 0.5 is CHEAP to satisfy there → no pressure to move early.
Harshul isn't beating us at the same task — fixed [10,25] is ONE amplitude regime
(strictly easier). Our all-101 set is `NOISE_MODE="iid"` (σ=4.64) per
make_tfrs_all101.py, vs his correlated set; T1≈12 mV ≈ 2.6σ so iid noise flips the
wing pixels that carry angle info. User owns iid; Harshul owns non-iid.

**Literature check (2026-07-29):** SIMPLE is ONLY a gradient estimator and
deliberately has no temperature — commitment is out of scope for it. The field with
this exact problem is one-shot NAS: "supernet-child gap"/rank inconsistency, and the
fast-signal bias (DARTS over-picks skip connections ≈ our position-over-angles).
Standard fixes = progressive search-space shrinking (P-DARTS), entropy-based
supernet shrinking (FX-DARTS 2504.20079), temperature annealing (Concrete
Autoencoders 1901.09346, τ≥2 → 0.001-0.1), supernet warm-up, and retrain-the-winner.
So warm-start + sharpening are NAMED practices, not hacks.

**O4 IMPLEMENTED 2026-07-29 (user picked O11 = leave AF runs alone + O4 on Gautschi):**
`SimpleRouterLayer(anneal_beta=True)` adds an OPT-IN `log_k` weight → p({a,b}) ∝
exp(β(θa+θb)), β=exp(log_k) driven by any AnnealingScheduler (duck-typed on log_k).
Opt-in matters: adding the weight unconditionally breaks load_weights for every
beta-less checkpoint. New model name `ViT_Max_SimpleRouterBeta`; driver flags
`--beta-final/--beta-start/--beta-epochs` (default OFF → byte-identical old
behaviour) + `DelayedHoldAnnealingScheduler` (hold β=1 until beta_start, ramp, then
hold — sharpening from epoch 0 would freeze the position-driven late ordering).
Smoke `scripts/distillation/smoke_beta_router.py` 9/9 incl. μ vs brute-force
enumeration and closed-form dθ vs finite-difference dE[L]/dθ.
**CRITICAL BUG FOUND+FIXED by that smoke test:** the algebraic marginals
`Z=0.5(S1²−S2)`, `μ=w(S1−w)/Z` are catastrophically unstable exactly in the
committed regime β aims for — with w=[1,1.5e-8,…] both S1² and S2 round to 1.0 in
float32 → Z=0 → **μ=NaN at β=60**, 5e-2 error by β=10. Replaced with a
cancellation-free O(T²) float64 form: `_pair_stats` builds the zero-diagonal outer
product, Z=0.5·Σ, P=num/Z, μ=ΣP, and the gradient is `dφ = μ·dz + P@dz − μ·Σ(μ·dz)`.
Verified exact to 7.5e-8 vs brute force. Calibration: top-2 μ mass 0.13/0.93/1.66/1.97
at β=1/5/20/60 → use **β_final≈30**. NOTE the chain rule dθ=β·dφ is applied by
autodiff itself (φ=β·θ is a traced op) — multiplying inside grad() double-counts.

**GAUTSCHI STAGING done 2026-07-29** (SSH from AF needs BoilerKey → user must log in
themselves; I can only prepare). `/depot/cms` here IS `datadepot.rcac.purdue.edu`,
the same Depot Gautschi mounts (26 TB free) → staged to
`/depot/cms/users/das214/SmartPixels_gautschi/`: the 19 GB all-101 TFR set (42 files,
verified by apparent size + count; `du` shows 31 G there purely from Depot block
inflation) + the repo. NOTE `rsync` does NOT exist on the AF node — use `cp -a`/tar.
Driver is now PATH-PORTABLE (repo-relative helpers, `SMARTPIX_TFR` /
`SMARTPIX_DATA_BASE` env overrides) — resolves identically on AF, so the running
seeds are unaffected. Scripts: `scripts/gautschi/setup_env_gautschi.sh` (pinned
mirror of the AF env, `tensorflow[and-cuda]==2.15.1` for H100/sm90) and
`submit_o4.sbatch` (self-chaining sbatch: resubmits until epoch 5000, since resume
is exact; user must EDIT `-A` allocation and `-p` partition after checking
`slist`/`sinfo`). Gautschi = 8× H100 80 GB/node, Slurm, 7 PB all-flash — and our job
is INPUT-bound (517 MB/batch, ~16 GB/epoch) so the flash FS should help.
Caveat: resume does not restore Nadam moments — fine for 1-2 chunk resumes, NOT for
the 4 h standby QoS (~10 resets).

**GAUTSCHI IS LIVE 2026-07-31 (O4 seed 8042, job 14497729).** User registered the
AF pubkey in Gautschi `~/.ssh/authorized_keys`, so `ssh gautschi.rcac.purdue.edu`
now works key-only from AF — I can drive it directly. Hard-won setup facts:
- **H100/`ai` partition is BLOCKED**: `slist` shows GPU Hours Balance 0.0 for both
  `cms` and `physics`; submits fail `AssocGrpBillingMinutes` under every QOS the
  user has (normal/preemptible; `standby` invalid there). Needs a PI allocation
  request to RCAC. Usable instead: **`smallgpu`, L40S 46 GB, 12 h cap, requires
  exactly 64 CPUs per GPU** (scheduler rejects other ratios; `ai` wants 14).
- **CONDA IS UNUSABLE HERE.** On Depot it saturated the NFS mount (every /depot
  stat on the login node hung for minutes); on Lustre scratch it died twice with
  `OSError: [Errno 5]` in "Verifying transaction". Use a **venv** on the module
  python. `python/3.11.9` is in **modtree/cpu**, NOT modtree/gpu (fine —
  tensorflow[and-cuda] ships its own CUDA 12, same env runs on L40 and H100).
- `pip install --upgrade pip` on Lustre produced a HALF-WRITTEN pip
  (`ModuleNotFoundError: pip._vendor.rich.align`) that `ensurepip` refused to fix
  ("already satisfied") — bootstrap with get-pip.py instead.
- Deps missing from my first pin list: **natsort, tqdm, pyarrow**.
- Slurm runs batch scripts in a NON-login shell → `module: command not found`
  (job 14497092 died in 1 s). Shebang must be **`#!/bin/bash -l`**.
- Layout: env + dataset on **Lustre scratch** `/scratch/gautschi/das214`
  (job reads ~16 GB/epoch — Depot NFS cannot sustain that); repo + scripts +
  run outputs on Depot. `/depot/cms` is the same mount on AF and Gautschi.
- **L40S = 15 s/epoch** vs 68 s/epoch on the AF A100 (4-way shared) — 4.5x faster
  because the job is input-bound and Lustre is fast. So 5000 ep ≈ 21 h ≈ only ONE
  12 h-cap restart, not the ~8 I feared.

**OPTIMIZER-STATE CHECKPOINTING ADDED 2026-07-31 (user-requested before the long
run).** `tf.train.Checkpoint(optimizer=..., constraints=...)` written every epoch
to `opt_state`, restored on resume (deferred restore applies at slot creation, so
no manual optimizer.build needed). Verified by
`scripts/distillation/smoke_optstate.py`: a warm restart reproduces an
uninterrupted run **exactly (max|ΔW| = 0.0)** vs 3.3e-3 for a cold restart.
TWO traps found while building it: (1) testing on the REAL model is useless —
router pair-sampling + ViT dropout make continuous vs restarted runs diverge from
RNG drift alone (warm 2.6e-3 vs cold 2.9e-3, i.e. no signal); test on a
deterministic MLP wrapped in the real MDMM. (2) `MDMM.save_weights` delegates to
the inner model so **the Lagrange multipliers are NOT in the .hdf5** — putting the
constraints in the same tf.train.Checkpoint is what took warm from "26x closer"
to exact (the CSV replay is only 6 sig figs, kept as fallback).

**EMERGING CROSS-SEED RESULT (all seeds, ~epoch 120-1100):** every MDMM-constrained
seed sits EARLY-to-MID, none near the unconstrained late anchor 78 — 1042 [53,58]
(drifted from [45,53] during the extension), 2042 [30,36] (from [30,98]), 3042
[23,24] (≈ the hand-picked [11,26]!). Direction reproduces; exact location not
converged (spread 23-58). Seed 1042's extension already improved best val
−27,494 → −27,898 by epoch 1107, so the NLL does keep descending past 1000 ep.

**Clip discovery (important, explains "stuck seeds" everywhere):** at random init the
predicted Cholesky diag can be ~1e-9 → 4D-Gaussian likelihood underflows for ALL
events → custom_loss's clip_by_value(1e-9) saturates → EXACTLY zero gradient for the
whole network (val NLL frozen at ~99,113 ≈ Harshul's QConv2D stuck 98,980). The
unconstrained studies escaped by dropout luck; under MDMM the corr penalty supplies
the only gradient at first (sanity: cotA corr 0→0.30 in 3 ep while NLL frozen) and
drags the model off the plateau deterministically. Consequently the driver's
AbortOnStuck uses thr=1e5/patience=20 (Harshul's numbers — plateau sits BELOW thr),
NOT the unconstrained study's 1e4/15, which would kill every seed mid-escape.
Router note: constraint pass runs training=False → hard top-2; theta keeps its exact
sampled-pair gradient; visits not double-counted (smoke-verified). See
[[soft-router-design]] for the SimpleRouter contract.
