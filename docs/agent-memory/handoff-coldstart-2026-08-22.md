---
name: handoff-coldstart-2026-08-22
description: "READ FIRST (supersedes handoff-deep-head-o17-o18 for live state). Cold-start pair-lattice + loss v2 CAMPAIGN ANSWER: discovered (10,21) beats the hand-imposed [11,26] by ~40% on angles. Live fleet state, deck/artifact state, all operational traps."
metadata:
  node_type: memory
  type: project
---

**HANDOFF 2026-08-22.** Read with [[handoff-deep-head-o17-o18]] (that file holds
the O11-O21 ledger + the loss-v2 adoption; this file holds the ANSWER and the
current live state).

## THE RESULT — cold search beat the hand-picked configuration
Pair-lattice router + loss v2 (softplus diag, unclipped), COLD start: no warm
start, no inherited weights/thresholds/slices. Three Gautschi seeds ran the full
10,000 epochs and converged on the same configuration:

| seed | pair | thresholds mV | NLL | sigx | sigy | **sigA** | **sigB** | pulls |
|---|---|---|---|---|---|---|---|---|
| 22042 | **(10,21)** | 8.6, 20.2, 47.5 | -41,217 | 6.64 | 1.93 | **7.78** | **5.57** | .88-.96 |
| 22242 | (10,23) | 8.8, 20.3, 48.4 | -40,912 | 6.49 | 1.85 | **7.62** | **5.43** | .94-1.03 |
| 22142 | (11,18) | 7.5, 17.8, 44.9 | -40,195 | 6.60 | 1.99 | 10.84 | 8.01 | .86-.98 |

vs the O17 hand-imposed [11,26]: sigA 12.7-14.0, sigB 8.3-9.3, pulls 1.2-1.5.
=> **~40% better angles, same position, far better calibration, DISCOVERED.**
SHIP CANDIDATE: slices ~(10,21), thresholds ~[8.6, 20.2, 47.5] mV.

AF cohort (runs/o21v2a_phi0, 4 seeds, 5000 ep, complete 200-frame phi_history
from epoch 0 -- the only runs with a full true-probability record):
  23042 [11,21] sep10 -40,463  sigA 12.05 sigB 7.95   <- success
  23142 [30,31] sep 1 -30,405  sigA 47.80 sigB 28.40
  23242 [53,56] sep 3 -31,494  sigA 51.28 sigB 30.62
  23342 [65,73] sep 8 -29,609  sigA 54.91 sigB 32.87
Cold hit rate 1/4 at 5000 ep, but the 10k seeds show mid-window is a WAYPOINT
not a trap (22142/22242 escaped it after ~5000 ep) -> LONGER RUNS RAISE THE HIT
RATE. Not separation alone: 23342 has sep 8 but sits at slices 65-73 (past the
pulse) and still fails. Needs an EARLY ANCHOR (~10-14) AND real separation
(partner ~18-26). Position is insensitive to pair choice (sigx 5.98-6.82
everywhere); the pair choice buys ANGLES.

## LIVE STATE
- AF A100: IDLE, all 4 seeds finished. Free for the next campaign.
- Gautschi: 3 v3 seeds (24042/24142/24242, runs/o21v3a_pairlattice_phi) at
  ~ep 8400-9900 of 10,000, all sitting at (10-11, 18-19) -- corroborating.
  Evaluate them when they finish.
- Monitor: $SC/watch_wrapped.sh -> watch_runs.sh (re-arm after every session
  restart; it dies with the session). It runs the chain reaper each cycle.
- Decks (restart with setsid so they survive session churn):
    cd /work/users/das214/atrain-slides
    setsid nohup node bin/atrain.mjs weekly.atdeck   --port=50696 --no-open &
    setsid nohup node bin/atrain.mjs o21v2cold.atdeck --port=8900  --no-open &
  weekly.atdeck = 45 slides; o21v2cold.atdeck = 19 slides (O21a.v2-cold story,
  GIFs, and 3 interactive embeds: sampled-pair scrubber log+linear, true-phi
  scrubber). Slides are STALE vs the result above.
- Artifact claude.ai/code/artifact/bcd335df-ce9f-4187-948e-c8df05b52537
  ("The Adjacency Trap") is STALE (pre-loss-v2). It is ours again (the twin that
  maintained it is dead). Repeated "republished elsewhere" notices all cite the
  SAME version 1787030547-bc47 = stale re-announcements, not real edits.
  User was offered a refresh; had not answered as of compaction.

## TRAPS (each cost real time)
1. **Self-resubmit chains die SILENTLY.** sbatch refused
   (QOSMaxSubmitJobPerUserLimit; smallgpu = 8 submits / 4 GPUs per user) and the
   exit status was ignored -> 3 chains sat dead for hours. FIXED: RESUBMIT GUARD
   in all 4 sbatch scripts (retry 5x, then log "RESUBMIT GUARD: GIVING UP") +
   tools/reap_chains.sh (freshness-based; .reaper_job marker prevents DUPLICATE
   chains on one seed -- which would corrupt shared checkpoints; bails at the
   8-job cap so it does not spam refusals), wired into the monitor.
2. **result.json is WRONG for chained runs.** Keras checkpoint 'best' resets on
   every resume: 22142 reported -31,032 when history.csv said -40,195. Trust
   history.csv. best.weights.hdf5 = best-of-LAST-CHUNK, not global best.
3. **history.csv can contain NUL bytes** after an OOM-killed chunk (22142: 666).
   awk copes; Python csv.DictReader raises "line contains NUL". Clean a local
   mirror before eval; keep the .orig.
4. **Never tar a run dir while it is training** -- torn CSVs. Mirror finished
   runs, or snapshot + validate.
5. Eval needs BOTH SMARTPIX_MODEL=ViT_MaxDeep_PairLattice and
   SMARTPIX_DIAG=softplus (v2 dialect); wrong dialect makes pulls meaningless.
6. Bandit arm RETIRED by user decision (ARM=c exits immediately in
   submit_o21v2_cold.sbatch; reversible by deleting the guard block).
7. scontrol does NOT expose the exported SEED/ARM, so jobs cannot be mapped to
   seeds that way -- use history.csv freshness instead.
8. Deck servers and the monitor die with the session; the widgets are embeds
   that only run via the canvas "Interact" chip or Present mode.

## NEXT STEPS (nothing is blocked; all of these need the user's word)
1. Evaluate the 3 v3 seeds when they hit 10k.
2. Refresh deck + artifact with the cold-start result (currently stale).
3. SHIP-NUMBER RUN: frozen-router confirmation on (10,21) under loss v2, ~6
   seeds on the idle AF A100 -- the mechanism-independent number for hardware.
4. Rebuild the true-phi widget/GIFs from the AF cohort: they now have complete
   200-frame records, so the animation can be REAL softmax(phi) from epoch 0
   (make_prob_widget.py auto-switches when phi_history.npz is present).

## UPDATE (same day, after the handoff was first written)
The 3 Gautschi v3 phi seeds FINISHED (10,000 ep) and the queue is now EMPTY --
the whole fleet is idle. Driver's own converged lines:
  24042 [10,18] thr=[8.91, 20.86, 48.33] corrs x.993 y.988 cotA.997 cotB.979
  24142 [11,18] thr=[8.23, 19.55, 47.54] corrs x.993 y.988 cotA.994 cotB.971
  24242 [10,19] thr=[8.93, 20.64, 48.15] corrs x.993 y.988 cotA.997 cotB.983
=> 6 of 6 mature cold seeds land in the (10-11, 18-23) band with thresholds
clustering at ~[8.5, 20.3, 48] mV. The result reproduces.
Each carries a COMPLETE 400-frame phi_history (6.3 MB/seed) -- richer than the
AF cohort's 200 frames.

TWO MORE TRAPS (both cost a wrong number here):
9. **A torn val_loss line POISONS the Keras 'best' monitor permanently.**
   seed 24142 logged val_loss = -37,265,446 once at ep5555; nothing can ever
   beat it, so best.weights.hdf5 FROZE at ep5555 for the remaining 4,400
   epochs and best_val in the driver log is that garbage number. ALWAYS
   evaluate SMARTPIX_CKPT=last for chained runs, and sanity-check any
   best_val outside the plausible band (roughly -30k..-45k for loss v2).
10. **Torn CSV lines can parse into PLAUSIBLE numbers**, not just absurd ones:
   seed 24242 showed a lone -45,408 (and -42,115) among neighbours at -39k.
   A range filter is not enough -- reject points far below the median of their
   +/-10-epoch window, or better, trust the driver's CONVERGED log line and
   the checkpoint evaluation over history.csv extrema.

## FULL 6-SEED COLD TABLE (all evaluated, ckpt=last, softplus dialect)
| seed | pair | sep | thr mV | sigx | sigy | sigA | sigB | pulls |
|---|---|---|---|---|---|---|---|---|
| 22242 | (10,23) | 13 | 8.8,20.3,48.4 | 6.49 | 1.85 | **7.62** | 5.43 | .94-1.03 |
| 22042 | (10,21) | 11 | 8.6,20.2,47.5 | 6.64 | 1.93 | **7.78** | 5.57 | .88-.96 |
| 24042 | (10,18) |  8 | 8.9,20.9,48.3 | 6.72 | 2.02 | **8.12** | 5.93 | .91-1.01 |
| 24242 | (10,19) |  9 | 8.9,20.6,48.2 | 6.68 | 1.98 | **8.14** | 5.97 | .96-1.04 |
| 22142 | (11,18) |  7 | 7.5,17.8,44.9 | 6.60 | 1.99 | 10.84 | 8.01 | .86-.98 |
| 24142 | (11,18) |  7 | 8.2,19.6,47.5 | 6.70 | 2.02 | 11.31 | 8.16 | .90-1.03 |

## NEW FINDING: THE ANCHOR MATTERS MORE THAN THE SEPARATION
Sorting the 6 mature cold seeds by sigA splits them CLEANLY by anchor slice:
  anchor 10 (4 seeds, sep 8-13): sigA 7.62, 7.78, 8.12, 8.14  -> mean 7.9
  anchor 11 (2 seeds, sep 7):    sigA 10.84, 11.31            -> mean 11.1
Moving the anchor 11 -> 10 buys ~3 deg of sigA. Within anchor 10, stretching
separation 8 -> 13 buys only ~0.5 deg. Two independent seeds agree at each
anchor, so this is reproducible, not seed noise. Position is flat across all
six (sigx 6.49-6.72) -- confirming again that the pair choice buys ANGLES only.
CAVEAT: the two anchor-11 seeds also have the smallest separation (7), so
anchor and separation are partly confounded here; AF seed 23042 = (11,21),
sep 10, sigA 12.05 breaks the confound in the anchor's favour, but it ran only
5,000 epochs so it is not a clean control. A frozen-router scan over
{(10,18),(11,18),(10,21),(11,21)} would separate the two cleanly.

## EPOCHS DRIVE THE COLD HIT RATE
  5,000 epochs (AF cohort, 4 seeds):    1/4 found the basin
  10,000 epochs (Gautschi, 6 seeds):    6/6 found the basin
The mid-window pairs (30,31)/(53,56)/(65,73) are a WAYPOINT, not a trap:
22142 and 22242 sat there past epoch 5,000 and still escaped. Cheapest test of
this: rerun the 3 failed AF seeds (23142/23242/23342) to 10,000 epochs.

## O22 -- RESIDUAL BIAS REMOVAL (launched 2026-08-27, running)
lgray's point on the 26 Aug thread: the network is systematically wrong by up
to 1.5 deg in a way that DEPENDS ON THE TRUE ANGLE. Bias does not average down
in a track fit the way resolution does, so a 1.5 deg effect on a plot whose
band is 5 deg matters more than it looks.

TWO ARMS on the AF A100, identical except two flags, both warm-started from
seed 22042 last.weights, pinned to (10,21), thresholds FROZEN at
[8.57, 20.23, 47.49], 2,000 epochs, same seed:
  runs/o22_bias_on   ZEROBIAS=1 BINBAL=byterm   (both fixes)
  runs/o22_bias_off  neither                     (CONTROL)
The control is load-bearing: without it "the constraint removed the bias"
cannot be told from "2,000 more epochs did".

THE LOSS (see conditional_nll.py + mdmm.py ZeroBiasConstraint):
  L = sum_k sum_e w_k(e)*nll_k(e)                          <- Fix 1
    + sum_t sum_b s*[max(lam_tb,0)*|rbar|/sig + (c/2)*(rbar/sig)^2]   <- Fix 2
* Fix 1 REPLACES the plain NLL; it is not an added term.
* nll_terms already returns the chain-rule decomposition (B,4), so each term is
  balanced in ITS OWN target's bins -- no single per-event weight has to serve
  four binnings. The terms are CONDITIONAL not marginal: perturbing x moves all
  four, perturbing beta moves only term 3.
* One multiplier PER BIN (4 targets x 15 = 60). lgray's "L1 with the vertex at
  y=0" IS mdmm's infeasibility measure -- the L1 and the lambda are one thing.
* Angles constrained in DEGREES, not cot: the plot is in degrees and the two
  differ once a bin is wide.
* Infeasibility is DIMENSIONLESS (bias / that target's residual spread) so one
  scale serves microns and degrees. Scale defaults to BATCH (5000).

EARLY RESULT at epoch ~45/2000 (make_bias_comparison.py):
  alpha max|bias|  6.42 (parent) -> 6.81 (control) -> 2.17 (constrained)
  worst bin        14.3 sigma    -> 15.1           -> 4.1
  beta             1.62          -> 0.81           -> 0.55
  cost: sigA +12.5%, sigB +4.4%, position ~+1%
The CONTROL did not improve alpha at all. Do not quote these -- 2% of the way in.

MORE TRAPS
11. plain_nll must be a compiled METRIC: mdmm's train_step builds its own return
    dict, so compile(metrics=...) is silently dropped. Patched to surface them.
    Training loss is a weighted composite; plain_nll/val_plain_nll are the
    comparable numbers. Use custom_loss_v2 not v1 -- these are softplus-dialect.
12. Vector lambdas break two writers that assumed a scalar per constraint
    (MDMMStateLogger row, result.json final_lambdas). Both patched.
13. SoftQuantizeLayer.thresholds is a PROPERTY, not a method.
14. The pair-lattice router ALSO answers to layer name 'simple_router_output',
    so the old FIX_SLICES path finds it and then assigns a 101-vector to a
    5050-entry psi. Find it by attribute (psi/ia/ib) instead.
15. atan(1/c) has an INFINITE derivative at cot=0 -- one such event NaNs the
    batch. Use atan2(1, c): same values, no division.
16. nohup alone is not enough for long runs launched from a tool call; the call
    timing out kills the child. Use setsid (same as the deck servers).
17. Warm start transfers the quantizer thresholds (not a 'simple_router' layer),
    so FREEZE_THR must run AFTER it or it reports thr0, a seed-derived random
    triple, instead of the inherited values.

## O22 UPDATE (2026-08-27 evening) -- SCAN DONE, CONFIRMATION RUNNING
Take 1 and take 2 both FAILED, each for a different reason, and both failures
are the useful part. Archived, not deleted:
  runs/_o22_onehot_overfit_{on,off}   take 1
  runs/_o22_runaway_notol_on          take 2
  runs/o22_control_final              the VALID control (1,685 ep) -- reusable
  runs/o22_tol0{10,17,30,50}          the 400-epoch tolerance scan
  runs/o22_final_tol017               3 seeds x 2,000 ep, IN FLIGHT

FAILURE 1 -- one-hot pin removed the sampler.
PairLatticeRouterLayer SAMPLES one pair per training step from softmax(phi) and
only uses argmax at eval. Seed 22042's final peak is p=0.0365, so training drew
the winning pair on 3.6% of steps and something ELSE on 96.4% (25 pairs cover
half the mass, 80 cover 90%). That is a huge augmentation. Spiking psi to
one-hot deleted it and BOTH arms overfit within 200 epochs -- control went
train -47,218 -> -49,415 while val went -39,581 -> -34,220.
FIX: SMARTPIX_PIN_MODE=inherit -- keep the source checkpoint's learned psi and
just set psi._trainable=False. Same (10,21) readout, sampling preserved. With
it the control held flat for 1,685 epochs (-40,4xx throughout).
=> ANY "frozen router" run must use inherit, never a one-hot spike.

FAILURE 2 -- zero-bias as an EQUALITY constraint cannot be satisfied.
An unbiased model still pays, because a bin mean fluctuates. Multipliers reached
361,073, outvoted the NLL, and the model escaped by WIDENING its predictions
(sigma +80..200%, val -15,341). Bias did fall (worst bin 10-38 sigma -> 3-5) so
the mechanism is sound; the dynamics were not.
FIX: tolerance band. inf = max(0, |bias|/sigma - tol), plus max_lambda/inf_cap
(which MaxBlockNLLConstraint already had and ZeroBiasConstraint did not).
tol MUST be set against the PER-BATCH noise, not the eval-set noise:
  batch 5,000 / 15 soft bins ~ 356 per bin -> bin mean fluctuates 0.056 sigma
  eval 37,919                ~ 2,712 per bin -> 0.021 sigma
My first tol=0.06 came from the eval figure = ~1 sigma of batch noise, so a
third of bins breach it by chance every step. 3x batch noise = 0.17.

SCAN RESULT (400 ep, vs parent; bias in um / deg):
  arm        x     y     a     b   | dsigx dsigy dsiga dsigb
  parent   3.60  1.14  6.42  1.62  |    -     -     -     -
  control  3.17  0.98  7.98  1.28  | -1.1  -1.1  +3.1  +0.4   <- bias UP, not down
  tol0.50  2.72  0.88  4.21  1.17  | -1.4  -0.5  +7.2 +13.6
  tol0.30  4.81  0.70  3.14  0.59  |+12.8  +5.9 +27.3 +28.7
  tol0.17  1.04  0.31  2.06  0.57  | +2.0  +3.8 +25.2 +31.2   <- OPERATING POINT
  tol0.10  1.16  0.21  3.04  0.98  | +8.7  +8.2 +33.9 +30.0
tol=0.17 is also the value derived from the noise floor BEFORE the scan ran.
The control's alpha bias going UP is the load-bearing row: training alone does
not remove this bias.
DO NOT rank adjacent tolerances -- epoch-to-epoch val swings reach 32,000 nats
on the loose arms, so each endpoint is one draw from a wide spread.

TRAPS 18-21
18. PIN_MODE=onehot silently removes the sampler (see FAILURE 1).
19. An equality constraint on a noisy statistic can never be satisfied; always
    give it a band sized from the noise the CONSTRAINT sees, not the noise the
    PLOT sees.
20. `ps ... | grep -c` matches its own command line. Use awk on ps output.
    Cost me a false "5 workers" reading twice.
21. A launcher script with `sleep N` between launches survives its own crashed
    child: relaunching it produced TWO trainers writing one checkpoint dir.
    Launch long runs directly, or verify with the awk inventory afterwards.
