---
name: handoff-o24-campaigns
description: "READ FIRST: O24 cold + O24b warm bias-loss campaigns in flight (10K epochs, user-mandated full budget), configs, results so far, and the three launch traps hit"
metadata:
  type: project
---

As of 2026-09-02, two 10,000-epoch campaigns run the O22 loss (loss v2 + byterm
bin-balance + 60-bin ZeroBias MDMM, tol 0.17, max_lambda 5, inf_cap 2) on
ViT_MaxDeep_PairLattice. **The user explicitly ordered the full 10K with no
early stops or mid-run opinions** — late grokking is the point (the parent
improved at ep ~4,999 after 768 stale epochs).

- **O24 cold** (Gautschi, seeds 40342/40442/40542/40642, `runs/o24_scratch_bias/s<SEED>`):
  parent recipe verbatim (free router, random trainable thresholds, smoothing
  sigma 4->0 over 3K) + the new loss from epoch 0. Chunks die of OOM every ~149
  epochs (constraint tripled the leak; SMARTPIX_GPU_MEM_MB raised 20000->42000
  on the L40S nodes) and self-resubmit via `submit_o24_scratch_bias.sbatch`.
  At ep ~1,250 the gap to the parent's matched-epoch curve closed 11K -> ~3K:
  the constraint tax is front-loaded and amortizing. Constraint NOT saturated.
- **O24b warm** (AF A100, seeds 41042/41142/41242, `runs/o24b_from_o22/s<SEED>`):
  warm start from `runs/o22_ep100/seed_22042/last.weights.hdf5`, router psi
  inherited AND trainable via the new `SMARTPIX_PIN_MODE=inherit_free`, NO
  smoothing schedule. Best plain NLL = the warm-start state (~-40.3/-40.4K,
  beats parent's -40,162) set in the first ~10 epochs and never beaten through
  ep ~4,800; medians plateau ~-38K. **Router migrated in all 3 seeds to the
  anchor-8 family (8,20)/(8,21)/(8,22)** — stable for thousands of epochs, but
  never concentrates (50% mass in ~27-32 pairs).

Traps hit this campaign (all archived with READMEs): (1) launching the driver
without `SMARTPIX_MODEL_NAME` silently trains shallow ViT_Max_SimpleRouter ->
`runs/_o24_wrongarch_cold`; (2) constraint-from-epoch-0 on a COLD model with
scale 5000 saturates (pen pinned at 900,000/target vs NLL 40K) — on the wrong
arch; correct arch does NOT saturate; (3) copying SMOOTH_SIGMA0=4 into a warm
start erases the inherited incumbent at ep 0 (adjacency trap (13,14)) ->
`runs/_o24b_smoothtrap`. Warm starts must keep the source's sigma≈0.

Still pending when runs finish: CPU bias-eval of the O24b best checkpoints
(may already beat the shipped O22 operating point), full perf plots, artifact
update ([[handoff-deep-head-o17-o18]], the "Bias and Ceiling" artifact
61401283-d679-4dd3-8fe4-ad2d2f0961f4 documents O22+O23). New driver knobs:
SMARTPIX_BIAS_START (gate ZeroBias until epoch N), SMARTPIX_BIAS_SCALE,
PIN_MODE=inherit_free. O23 verdict: router-less ceiling arms overfit on
regularization, not epochs ([[handoff-coldstart-2026-08-22]]).

**(historical) A100 PAUSED (2026-09-03 evening) for the FastML26 hackathon until Friday
afternoon; no new GPU launches until told to resume. CPU-only work is fine.**
State at pause: O24b seeds 41142 and 41242 COMPLETE (final readouts (8,21) and
(8,19); thresholds ~[5.3,15.6,38.9] / [5.4,16.1,39.5], i.e. 35-40% below the
inherited [8.57,20.23,47.49]); seed 41042 stopped cleanly at epoch 9,388 with
last.weights.hdf5 + opt_state in runs/o24b_from_o22/s41042/seed_41042/. Resume
= relaunch the identical O24b env with `--extend` (driver resumes from
last.weights). The Gautschi cold chains (~ep 4,400-4,530) were NOT paused (not
on the AF). The cold->A100 migration script is staged at
scratchpad/migrate_cold_to_a100.sh and is HELD until the pause lifts.
Pending after resume: full O24b evaluation (bias plots, pulls, best-ckpt vs
parent/O22), then migrate the cold campaign to the A100.

**2026-09-07 RESUMED.** A100 now runs warm 41042 (from ep 9,389) + cold
40342/40442/40542 (migrated from Gautschi by copying run dirs over the shared
Depot mount; resumed at 7,505/7,455/7,341). Cold 40642 still chains on Gautschi
(ep 7,352). O24b evaluation done: anchor-8 / lowered-threshold readout trades
angle resolution for position resolution -- not a dominating improvement; the
-40.4K plain-NLL state was never checkpointed (recover with a ~10-epoch warm
start + SMARTPIX_CKPT_MONITOR=val_plain_nll once a GPU slot frees, then migrate
40642). Trap: qkeras needs `networkx`; the env lost it with the inherited
PYTHONPATH after a Claude restart -> pip-installed. The durable handoff is in
the repo: docs/handoff/HANDOFF_2026-09-07.md, routed by CLAUDE.md/AGENTS.md.
