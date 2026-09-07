---
name: handoff-deep-head-o17-o18
description: "READ FIRST. Live state ~2026-08-13: O17 deep-head breakthrough (sigma_alpha 13 deg at intact position), O18 launched, full campaign ledger O11-O18, deck at 33 slides, all operational traps"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7a0a041e-b87d-47e5-ae22-8b5c100f4193
  modified: 2026-08-18T05:22:39.639Z
---

**HANDOFF (pre-compaction). Supersedes [[handoff-2026-08-03]] where they conflict.**

**NAMING (user's conventions):** "O<n>" = Option/campaign number. **"ViT_d"** =
the DEEP-decoder ViT (head 1280->256->128->64->14, ~706K params, factories
`ViT_MaxDeep_*`) — user's name as of 2026-08-15; use it in all writing/slides.
Vanilla/shallow ViT = head 1280->64->14 (~419K, factories `ViT_Max_*`).
Everything from O17 onward (O17/O18/O20/O21/O21.v2) is ViT_d.

## THE BREAKTHROUGH — O17 (deep regression head, frozen [11,26])
The user suspected the decoder MLP was too shallow. They were RIGHT, in the way
that matters. Head was `Flatten(1280)->Dense(64)->14` (one hidden layer, 19.6%
of all params in one shared matrix). New: `1280->256->128->64->14`
(`ViT_MaxDeep_SimpleRouter`, 706,012 params vs 418,908).
- Frozen router at [11,26], uncapped anglenll constraints (cotA<=-0.15,
  cotB<=-2.20, clip=1), 2000 ep — IDENTICAL to the shallow baseline arm.
- **Deep-basin hit rate among escapees: 1/5 (shallow) -> 5/5 (deep).** Paired
  seed 41542 flips −25,750 -> −38,734 (init is fully seed-determined: thr0 +
  tf/np/random all seeded, so pairing is exact).
- **Full eval (37,919 events): σα 12.7–14.0°, σβ 8.3–9.3°, σx ≈7 µm, σy ≈2 µm,
  NLL −37.4K…−39.0K, thresholds ≈[14, 26, 61] mV.** Prior bests: σα 39.2°
  (O11 2042), 30.3° (O13 13342 but σx 56 µm). This is the FIRST config where
  angles AND position work simultaneously. Pulls 1.2–1.5 (slightly overconfident).
- Cost: plateau aborts 7/13 launches vs 2/7 shallow (paired 41042 flipped
  escape->abort). Aborts cost ~5 min (AbortOnStuck thr=9.9e4 @20 ep). Idea if
  needed: warm-up INF_CAP for first ~50 ep only.
- User has seen ~−50K on [11,26] with MDMM forcing historically -> headroom.
- Still finishing on Gautschi chains: deep 41242, 41642, then 41742 (jobs
  fxD_11_26_a/b, self-resubmitting, names must stay fxD_*).

## O18 — JUST LAUNCHED (deep head + O15 kernel + FREE router)
Hypothesis: O15 proved the free router finds whatever the objective prefers
(late window, −30.5K). O17 proved the deep head unlocks a −38–39K basin at
EARLY slices. If that basin is reachable, the plain objective may now prefer
early slices -> router should find them UNAIDED (no angle constraint).
- Factory `ViT_MaxDeep_SimpleRouterSmooth` (deep head + smooth_logits;
  706,013 params, verified built). Registered in train.py.
- Env: `SMARTPIX_MODEL_NAME=ViT_MaxDeep_SimpleRouterSmooth
  SMARTPIX_SMOOTH_SIGMA0=4 SMARTPIX_SMOOTH_EPOCHS=1500`, default corr
  constraints (O11's), 5000 ep, out `runs/simplerouter_o18deepsmooth`.
- Seed 18042 running; 18142/18242/18342 auto-launch via
  `$SC/o18_filler.sh` (background, fills AF slots as the 3 O17 extras finish;
  log $SC/o18_filler.log). AF cap = 4 workers.
- JUDGE ON: final slices (early [11,26]-ish vs late [69,73]) and NLL (deep
  basin < −35K vs late-window −30.5K). If early: the whole angle problem
  dissolves — objective was fine, capacity was the wall.
- **NaN failure mode (2026-08-13, seed 18042 died ep 4)**: deep head parks on
  the clip plateau; L diags on the 1e-9 floor; the triangular-solve cascade
  gives unclipped BOUNDARY events gradients ~ (1/L)^3 ~ 1e27; one such spike
  squared overflows Nadam's float32 second moment -> inf/inf = NaN weights.
  Measured in mdmm_epochs.csv ep 0: unclipped nll_x 2.4e18, nll_y 4.9e18,
  nll_cotA 1.4e38, nll_cotB 9.7e56. Corr constraints EXONERATED (pred stds
  healthy; also hardened to sqrt(var+1e-12) anyway, forward-identical).
  check_numerics can't help: benign clip-masked z^2=inf fires first at ep 1
  (hook SMARTPIX_CHECK_NUMERICS added to driver regardless). O15/O17 never
  NaN'd: shallow leaves plateau instantly / frozen router doesn't hop pairs.
  Policy: AbortOnStuck kills NaN runs in <=20 ep; abort-aware o18_filler.sh
  (queue 18142..18642, TARGET=4 escapees) auto-replaces. If NaN rate proves
  systemic, ASK USER about a high global_clipnorm NaN-guard on Nadam (touches
  optimizer config = breaks one-change-at-a-time vs O15; not done silently).
  Dead run archived at runs/simplerouter_o18deepsmooth/nan_18042_ep4.
- **O20 COMPLETE (5000 ep, 2026-08-14)**: both escaped. 20042 best −35,825
  @ep768, final readout [7,12] (late-phase decommit, mu entropy ~3.7 — best
  ckpt is the result). 20142 best −36,004 @ep2986, readout [12,13]. FINAL
  physics (evaluated, plots in runs/perf_plots_o20/): 20042 [11,12]
  T=[7.5,19.8,57.8]mV sigx 10.9um sigalpha 16.8deg sigbeta 11.9deg; 20142
  [12,14] T=[7.6,18.9,53.5]mV sigx 8.1um sigalpha 19.9deg sigbeta 14.1deg;
  pulls 1.02-1.13 (better calibrated than O17). AF mirror:
  runs/simplerouter_o20warm/.

## O20 — WARM-START PROBE (launched 2026-08-14, Gautschi jobs 15267569/70)
O18 ANSWERED ITS QUESTION: all 4 seeds (18242/18342/18442/18542) re-found the
LATE window (pairs ~72-81, three at 73/74) as sigma annealed — the deep head
alone does NOT flip the free router. Completes the 2x2: capacity is necessary
(O17 5/5) but not sufficient for discovery; the router-facing landscape points
late regardless. User's chosen follow-up: O20 = exact O18 recipe but backbone+
head+thresholds warm-started from O17 escapee 41942 (best −38,994), ROUTER
theta kept at fresh flat init (checkpoint theta carries the [11,26] answer —
must not leak; driver asserts this). If the developed representation flips the
router early -> landscape story proven; if the head re-adapts late anyway ->
the late attractor beats even a formed early representation.
- Driver: SMARTPIX_WARM_START=<ckpt> SMARTPIX_WARM_SRC_MODEL=<factory that
  wrote it> (fresh starts only — chunk resumes use their own last.weights;
  src/main matched BY POSITION, names uniquify across in-session builds).
  Verified on AF and on Gautschi: 27 layers moved, 706,013 params, 0 mismatch.
- Gautschi: scripts/gautschi/submit_o20.sbatch (self-resubmitting, seeds
  20042/20142, out runs/simplerouter_o20warm, ckpt at REPO/ckpt/). Depot
  tools/counts.sh now appends o20_<seed>=ep<n>@(pair) per seed.
- O18 CLOSED (user decision 2026-08-14): all 4 seeds killed at ~ep 2400-2600,
  AF slots left FREE. Final banked bests −28.3K…−28.8K, pairs deep-late
  (80,82)/(80,83)/(73,80)/(80,82) — late lock unambiguous, control role done.
  Curves live in runs/simplerouter_o18deepsmooth (no result.json — killed, not
  aborted; mu plots fall back to router CSVs). 18542 had ESCAPED the plateau
  via corr (no NaN) — final NaN tally 1/5 seeds.
- **O20 VERDICT (sigma locked to 0 at ep 1500, 2026-08-14): PROVEN EARLY.**
  20042 -> (12,13) best −35,825; 20142 -> (13,14) best −35,645. Both beat the
  entire late-window basin (−30.5K asymptote) by ~5K nats with NO angle
  constraints — same objective that sent every cold seed late. Chicken-and-egg
  confirmed: the objective prefers early slices ONCE the representation
  exists; cold free search can never bootstrap it. Notable second finding:
  the free warm router picks ADJACENT early pairs (12-14), not the wide
  [11,26]; gap to O17's −38.9K is ~3K — unclear if refinement (to ep 5000)
  closes it or the wide rise+later pairing is genuinely better but not a
  free-search attractor. Runs continue to 5000; final eval (sigma_alpha etc.)
  + deck slides when done.

## O21 — THREE DISCOVERY MECHANISMS vs the adjacency trap (launched 2026-08-14)
O20's residual: free warm search picks ADJACENT early pairs (12-14, best −36.0K),
never the wide [11,26] (−38.9K) — gradient routers can't cross the valley and
the slice-space kernel enforces contiguity. User picked brainstorm options
2/3/4, all warm-started from the SAME o17_41942 ckpt (only variable vs O20 =
search mechanism), one seed each on the AF:
- O21a seed 21042: PairLatticeRouterLayer — trainable psi over all 5050 pairs,
  annealed 2D Gaussian in PAIR space (wide pairs are neighbours of wide
  pairs), exact categorical-covariance gradient lifted from SIMPLE's dz.
  out runs/o21a_pairlattice, model ViT_MaxDeep_PairLattice (710,962 params).
- O21b seed 21142: TwoRouterLayer — independent thetaA/thetaB (one per slot),
  per-router 1D kernel (shared smooth_sigma), differentiable anti-overlap
  penalty via add_loss. out runs/o21b_tworouter, ViT_MaxDeep_TwoRouterSmooth
  (706,114).
- O21c seed 21242: UCBRouterLayer — discounted-UCB bandit over 5050 arms
  (gamma=0.9995, all arms tried once first = ~163 ep forced exploration; NO
  smoothing env), reward = -batch loss via hasattr-gated bandit_update hook I
  added to mdmm.py train_step. out runs/o21c_ucb, ViT_MaxDeep_UCBRouter
  (721,065).
All layers keep name 'simple_router_output' + full logger contract (theta is a
read-only property — FIX_PAIR pinning incompatible, irrelevant here). All
state in add_weight → chunk-resume exact. Smoke tests scripts/distillation/
smoke_o21{a,b,c}.py all passed; my end-to-end check: warm transfer 27 layers,
1-epoch MDMM fit incl. bandit hook, all three. JUDGE ON: which mechanism
finds a WIDE early pair (11,26-class, NLL < −37K) vs re-locking adjacent.
NOT synced to Gautschi yet (AF-only runs).
**O21 v1 COMPLETE (all three ran to ep 5000, finished 2026-08-15).** Final:
a −41,710 @1318 (readout [11,18] at end, [11,24] at best-eval); c −41,023
@1123 (readout [10,69] end); b −36,055 @4362, adjacent [12,14] to the end.
Scratchpad was WIPED by a harness restart mid-day — ckpt snapshots gone
(sources intact in runs/), monitor scripts rewritten fresh (Gautschi-focused).
**VERDICT (a locked at ep 1500): a AND c both smash the O17 gold standard;
b stays trapped adjacent (the clean internal control).**
- O21a (pair-lattice) LOCKED sigma=0 on (11,22); best **−41,710** @ep1318.
  Eval of that ckpt (readout [11,24], T=[9.7,21.9,50.1] mV): sigx 6.46um,
  sigy 1.84um, **sigalpha 11.13deg, sigbeta 7.85deg**, pulls 0.93-1.00.
- O21c (bandit) best **−41,023** @ep1123; eval ckpt readout [10,48],
  T=[10.5,25.8,77.7] mV: sigx 6.50um, **sigalpha 11.69deg, sigbeta 7.80deg**,
  pulls 0.93-1.04. Earlier records on [8,86] (−40,040, sigalpha 11.84).
- O21b (two-router) stuck adjacent (13,14), −35,791, sigalpha 25deg —
  per-router 1D kernels keep both softmaxes in one region; escape is
  pair-space geometry/exploration, NOT the warm start (a/c vs b isolates it).
- CONSISTENT PICTURE: anchor slice ~8-12 + ANY partner 22-86 = broad plateau
  (a:(11,22-24), c:(10,48)/(8,86), O17:(11,26)). sigbeta < 8deg for the first
  time. First eval snapshot ckpts in $SC/ckpt/. Deck slides 28-32 auto-refresh
  via atrain-slides scripts/add_o21_slides.mjs (idempotent, live standings).

## O21.v2 — LOSS V2 RETRAIN (launched 2026-08-15, Gautschi 15297325/6/7)
User adopted the Symb_ASIC side's softplus-diag loss into OUR tfp-free stack:
`conditional_nll.custom_loss_v2` = softplus diag (minval+softplus(p)), NO
prob clip, log-domain via triangular_solve, batch-SUM (their reference is
tfp+mean; ours differs by exactly the batch factor). Verified: ==tfp softplus
reference to 0 reldiff; v1 clip path regression-clean; COLD-INIT GRADIENT
v1=0 (the dead plateau) vs v2=139 — kills the plateau+NaN class at the root.
- DIALECT RULE (hard): relu-dialect ckpts (O11..O21) and softplus-dialect
  (O21.v2+) must NEVER be cross-scored (~7 nats/event artifact). Driver env
  SMARTPIX_LOSS_V2=1 selects the loss AND flips MDMMStateLogger's nll columns;
  eval script takes SMARTPIX_DIAG=softplus for pulls (mu-based resolutions
  are dialect-free). v2 NLLs not comparable with any prior campaign number.
- O21.v2 = exact O21 arms (same mechanisms, SAME seeds 21042/21142/21242 for
  pairing, same warm start from relu-dialect 41942 — expect early diag
  recalibration dip), only the loss changes. scripts/gautschi/submit_o21v2.sbatch
  (ARM=a|b|c), out runs/o21v2{a,b,c}_*, self-resubmitting chains. counts.sh
  emits v2a/v2b/v2c=ep@(pair). O21 v1 arms left finishing on the AF.

## O21.v2 RESULTS (v2a/v2b DONE 2026-08-15; v2c mid-flight)
**v2a (pair-lattice, loss v2): NEW PROJECT RECORDS ACROSS THE BOARD.**
Escaped, locked [10,20], best −40,994 (v2 dialect), eval (softplus pulls):
T=[8.9,21.5,49.7] mV, sigx 6.64um, sigy 1.96um, **sigalpha 8.24deg,
sigbeta 5.99deg**, pulls 0.88-1.01. vs v1 best (sigalpha 11.1): the UNCLIPPED
loss fixes the worst-fit tail the clip Huberised away — residual std is
tail-dominated, so angles jump ~30%. Even v2b ADJACENT [11,13] gets sigalpha
13.2deg (v1 adjacent was 19.8) — the loss effect is universal, not pair-luck.
v2b: −35,286, adjacent — trap replicates under v2, mechanism story loss-
independent. v2c: AbortOnStuck misfired (v2 has no val ceiling; UCB sweep
spikes past the 9.9e4 clip-plateau threshold) — fixed via new env
SMARTPIX_ABORT_THR (driver line ~601; sbatch sets 1e12 for v2), chain resumed.
Evals in runs/perf_plots_o21v2/ + AF mirrors runs/o21v2{a,b}_*.
PENDING: v2c finish; deck O21.v2 slides; artifact update; frozen-router
confirmation protocol on [10,20] under loss v2 = the SHIP config candidate.

## O21.v2a.2 / O21.v2c.2 — COLD ZERO-PRIOR PROOF (launched 2026-08-18)
User's bias audit question: could the O20/O21 results be an artifact of the
O17 warm start? Audit found: router NEVER seeded (theta/psi/q = zeros =
uniform over all 5,050 pairs; the checkpoint's [11,26]-pinned theta is
explicitly excluded + asserted), thresholds randomised per seed in cold runs
BUT **warm runs inherit O17's trained thresholds ~[12.3, 26.8, 63.9] mV**
(soft_quantizer IS transferred; decoded via offset+cumsum(expm1(raw))), and
the representation itself is the real prior (trained at frozen [11,26]).
Counter-evidence vs contamination: the bandit swept [11,26] explicitly and
REJECTED it for [10,48]/[8,86]; cold O18/O15 both went LATE.
**The runs:** ViT_d + loss v2, NO warm start, cold random init, router from
uniform, thresholds random uniform(25,160). 3 seeds each (22042/22142/22242),
10,000 epochs (v2a's best came at ep 4567/5000 — not converged), corr
constraints KEPT (one change at a time), kernel horizon 1500->3000 for arm a.
Gautschi: scripts/gautschi/submit_o21v2_cold.sbatch (ARM=a|c, SEED=...),
jobs 15334946-51, out runs/o21v2a2_pairlattice + runs/o21v2c2_ucb.
counts.sh emits cold{a,c}_<seed>=ep@(pair).
**AF GPU TRAP (cost 4 failed launches):** the AF session restarted
2026-08-17T20:27Z on a ~3 GiB GPU SLICE (was a full 40 GB A100) — a
batch-5000 ViT_d worker no longer fits AT ALL there. Probe before launching:
allocate 1 GiB tensors in a loop and count. nvidia-smi still prints
"A100-PCIE-40GB" and memory.used is permission-denied, so it does NOT reveal
the slice. Restarting the AF session to a full A100 would kill this Claude
session -> user's call.

## CAMPAIGN LEDGER (all conclusions locked)
- O11: corr>=0.5 baseline. Angles alive but weak (gameable by shrinkage: 4042
  passes at 6% of true range). 8 seeds, no pair consensus, drift late.
- O4 (beta): sharpening pair softmax never concentrates mu. Dead end.
- Ablation (frozen router, shallow): EARLY beats LATE for cotA: [19,20]
  −0.162±0.019 vs [66,67] +0.115±0.062. [11,26] bimodal: 4 seeds ~−0.2, 1 seed
  −1.878 (the deep basin, ~1-in-5 with shallow head).
- O12 (block anglenll): satisfied via cotB, cotA got worse. Wrong target.
- O13 (two-lambda cotA/cotB, caps 25/2.0): router goes EARLY (slice 7 all
  seeds) but pays with position (σx 10–56 µm). Uncapped variant diverges
  (all seeds collapse to [7,8], lambda ~30+).
- O15 (kernel σ4->0@1500, shallow): best search ever (churn 3–5, 3 seeds
  agree to 0.5 mV) — SAME late window [69–73] @ −30.5K. Objective, not search.
- O17/O18: above.
- conditional_nll.py: exact per-target split of the 4D Gaussian NLL
  (x, y|x, cotA|xy, cotB|xy,cotA; sums to joint; clip=True bit-identical to
  custom_loss). MaxBlockNLLConstraint in mdmm.py (blocks angle/position/cotA/
  cotB; float64 + clip inside, max_lambda/inf_cap kwargs).

## DECK (port 50696, weekly.atdeck) — 33 slides, rev ~711
Slide 2 = "The campaigns at a glance" (4 TLDR cards, 24px). Every campaign has:
summary + mu distribution + training-curve + per-seed plot slides. 24px body
sweep applied (71 boxes; titles/footers/tables untouched). ALL EDITS VIA CRDT
(scripts/atrain_op.mjs) — never write deck.json (see [[atrain-edit-through-crdt]]).
Slide scripts (all idempotent by title, in /work/users/das214/atrain-slides/scripts/):
- add_o15_o17_slides.mjs — O15/O17 summary slides; refresh after regenerating
  $SC/o15_summary.json, $SC/o17_af.json, $SC/o17_gautschi.json (fsjson.py on
  Gautschi Depot tools/ emits the gautschi JSON).
- add_seed_slides_generic.mjs <port> <plotDir> <spec.json> <anchorTitle>
- add_overview_histories.mjs, sweep_font24.mjs.
Plot scripts all parameterised by env: make_perf_plots_mdmm.py
(SMARTPIX_RUN_DIR/PLOT_DIR/MODEL/CKPT/USE_SNAP), make_mu_plot.py (MU_*),
make_history_plot.py (HIST_*), make_history_o17.py (shallow-vs-deep overlay).
Kernel sandbox demo: port 8899.

## DECK + ARTIFACT (both survive restarts now)
Deck server dies with the pod. Restart with an ABSOLUTE deck path — `npm run
-w` resolves a relative one against the package dir and silently opens an
EMPTY deck at packages/deckd/weekly.atdeck:
  cd /work/users/das214/atrain-slides && nohup env ATRAIN_PORT=50696 \
    npm run deckd -- /work/users/das214/atrain-slides/weekly.atdeck &
Deck = 40 slides. O21 v1 slides 28-32 (add_o21_slides.mjs), O21.v2 + cold-proof
slides 33-34 (add_o21v2_slides.mjs). Both idempotent, read live run dirs.
Artifact "The Adjacency Trap" (updated through loss v2 + the cold proof):
https://claude.ai/code/artifact/bcd335df-ce9f-4187-948e-c8df05b52537 — SOURCE
now at smart-pixels-ml/reports/adjacency_trap.html (was in /tmp and got wiped
by a restart once; republish with url=<above> to keep the same link).
make_perf_plots_mdmm.py now MERGES perf_summary_mdmm.json (it used to
overwrite, silently dropping seeds not named in that invocation).

## OPERATIONAL TRAPS (each cost real damage once)
1. **HDF5 lock on CephFS kills the TRAINER** when a reader opens a live
   last.weights.hdf5 (killed 13042 at 4871, checkpoint 0 bytes). ALWAYS
   snapshot-copy + h5py-validate first ($SC/ckpt/seed_X.hdf5 +
   SMARTPIX_USE_SNAP=1), or plot only finished runs.
2. **Partial sync to Gautschi crash-loops sbatch chains** (smooth_logits kwarg
   TypeError, 9s/cycle self-resubmit). Sync the WHOLE two_bit_optimization_helpers
   package and verify by BUILDING the model on Gautschi. rsync absent — use
   tar|ssh with retries; /tmp NOT shared between login nodes — put tools in
   Depot smart-pixels-ml/tools/ (counts.sh, fs1126.py, fsjson.py).
3. **Checkpoint compatibility**: Smooth/Beta/Deep variants each add weights;
   loading needs the right factory (SMARTPIX_MODEL for perf plots: O15 needs
   ViT_Max_SimpleRouterSmooth, O17/O18 need Deep variants) else
   "expects 2 weights, received 3".
4. Result.json is ALSO written by AbortOnStuck — count "escaped": true, not
   file existence (counts.sh does this).
5. scan_existing treats seed_<n>_* dirs as attempted — archive dead runs
   OUTSIDE the seed_* namespace (diverged_*, nan_*) to allow relaunch.
6. Monitor: $SC/watch_runs.sh via watch_wrapped.sh (banner filter), persistent
   Monitor tool; watches O17+O18 dirs, o1[78]*.log faults, Gautschi counts.sh.
   AF MOTD banner pollutes every shell — grep it away.
7. Slurm: scancel by ID (repeated --name flags don't accumulate); QOS cap 8
   jobs; one job per pair with seeds sequential (co-tenancy OOM-thrashed);
   GPU leak ~170 MB/ep on Gautschi -> chunks die + self-resubmit by design.
8. Driver env contract: SMARTPIX_CONSTRAINT anglenll|corr, NLL_BLOCK csv,
   ANGLE_NLL_TARGET csv, LOSS_CLIP, LAMBDA_CAP, INF_CAP, FIX_SLICES,
   MODEL_NAME, SMOOTH_SIGMA0/EPOCHS, BATCH. Defaults = O11 verbatim.

## NEXT STEPS
1. Watch O18 (slices early or late = the answer). 2. Refresh O17 slides when
Gautschi chains land (n grows past 5/5). 3. If O18 goes late anyway: O19 =
deep head + two-lambda anglenll + position guard (−log p(x,y) <= −3.6) + kernel.
4. Plateau-abort fix candidate: INF_CAP=2 for first 50 ep only. 5. Decoder
upgrade candidate beyond MLP: 2 learned cross-attention queries (position/
angles) + split mean/covariance branches (DETR-style) — discussed, not built.

## COLD-START RESULT — the campaign's answer (2026-08-21)
Pair-lattice + loss v2, COLD start (no warm start, nothing inherited). Three
mature Gautschi seeds ran the full 10,000 epochs and converged:
  22042 (10,21) T=[8.6,20.2,47.5] mV  NLL -41,217  sigx 6.64 sigy 1.93  **sigA 7.78 sigB 5.57**  pulls .88-.96
  22242 (10,23) T=[8.8,20.3,48.4] mV  NLL -40,912  sigx 6.49 sigy 1.85  **sigA 7.62 sigB 5.43**  pulls .94-1.03
  22142 (11,18) T=[7.5,17.8,44.9] mV  NLL -40,195  sigx 6.60 sigy 1.99  sigA 10.84 sigB 8.01   pulls .86-.98
BEATS the hand-imposed O17 [11,26] baseline (sigA 12.7-14.0, sigB 8.3-9.3) by
~40% on angles, position unchanged, calibration far better (O17 pulls 1.2-1.5).
SHIP CANDIDATE: slices ~(10,21), thresholds ~[8.6,20.2,47.5] mV.

AF cohort (o21v2a_phi0, 4 seeds, 5000 ep, FULL 200-frame phi_history from ep 0):
  23042 [11,21] sep10 NLL -40,463 sigA 12.05 sigB 7.95   <- the success
  23142 [30,31] sep 1 NLL -30,405 sigA 47.80 sigB 28.40
  23242 [53,56] sep 3 NLL -31,494 sigA 51.28 sigB 30.62
  23342 [65,73] sep 8 NLL -29,609 sigA 54.91 sigB 32.87
Cold hit rate 1/4 at 5000 ep; the mature seeds show mid-window is a WAYPOINT,
not a trap -- 22142/22242 escaped it after ~5000 ep. NOT separation alone:
23342 has sep 8 but sits at slices 65-73 (past the pulse) and still fails.
Needs BOTH an early anchor (~10-14) AND real separation (partner ~18-26).
Position is insensitive to pair choice (sigx 5.98-6.82 everywhere); angles are
what the readout choice buys.

## GOTCHAS FOUND (2026-08-20/21)
- Self-resubmit chains die SILENTLY: sbatch refused (QOSMaxSubmitJobPerUserLimit,
  smallgpu cap = 8 submits / 4 GPUs per user) and exit status was ignored.
  3 chains died in one night. FIXED: RESUBMIT GUARD in all 4 sbatch scripts
  (retry 5x, then log "RESUBMIT GUARD: GIVING UP -- CHAIN DEAD") + reaper
  tools/reap_chains.sh (freshness-based, .reaper_job prevents duplicate chains,
  bails at the 8-job cap) wired into the monitor each cycle.
- result.json is WRONG for chained runs (Keras checkpoint 'best' resets on each
  resume): 22142 reported -31,032 when history.csv said -40,195. Trust
  history.csv; best.weights.hdf5 = best-of-LAST-CHUNK, not global best.
- Source history.csv can contain NUL bytes after an OOM-killed chunk (22142 had
  666). awk copes; Python csv.DictReader raises "line contains NUL".
- Bandit arm RETIRED (user decision): ARM=c exits immediately in
  submit_o21v2_cold.sbatch (reversible - delete the guard block).
