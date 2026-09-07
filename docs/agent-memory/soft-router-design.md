---
name: soft-router-design
description: "Soft router for time-slice discovery: chosen penalty-free slot-softmax design (SoftRouterLayer), key constraints, and the code-verified normalization rules"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7a0a041e-b87d-47e5-ae22-8b5c100f4193
  modified: 2026-07-27T20:05:37.121Z
---

New project thread (2026-07-02): a **SoftRouterLayer** that learns which 2 of the 101 time slices
to read out (replacing hand-picked [11,26]). Framing: like SoftQuantize, it's an **offline
bootstrapping tool, never on the ASIC** — SoftQuantize discovers ADC thresholds; the router
discovers the 2 readout indices. k=2 is a hard ASIC budget.

**Chosen design (user-driven, supersedes the earlier gate/L0 draft):** 2 selector slots, each a
softmax over 101 slice logits, annealed soft→hard by the SAME cosine k:1→67 AnnealingScheduler,
with the SoftQuantize STE (stop_gradient(hard−soft)+soft). **No penalty terms** — user explicitly
rejected an additional loss (free gates would keep all 101 slices open; SoftQuantize precedent:
cardinality must be structural). Duplicate collapse prevented architecturally (slot-2 logits
masked by log(1−a1)). Forward pass is always the hard 2-slice pick, so training NLL is honest.

**Plan docs (v3 — fully rewritten fresh, visual-first per user request; user is a visual
learner):** `smart-pixels-ml/docs/soft_router_plan.md` + `docs/soft_router_research.md`
(42-agent verified SOTA survey; recommendation updated A→B). Figures: 3 publication-grade
TikZ towers (discovery model / router internals / production model) built with the
tikz-diagrams skill at `/work/users/das214/SmartPixels/.claude/skills/tikz-diagrams/`
(sources in `docs/tikz/*.tex`, PNG+PDF in `docs/figs/fig_{a,b,c}_*`), plus 4 dataviz-skill
plots via `docs/make_router_plots.py` (`docs/figs/plot_*`). TinyTeX at ~/.TinyTeX has
standalone+pgf+pgf-blur installed (tlmgr). ViT backbone is PRE-LN (verified
transformer_model_nonquantized.py:54-68): patch (3,4)→20 patches, embed 64, 4 heads,
FF 128, 4 blocks, head LN→Flatten(1280)→Dense64→Dense14.

**Code-verified pipeline facts (byte-checked against production TFRs):**
- Pixels are RAW mV + baked noise — no standardization/log active (flags False; byte-proof 6e-5).
  Keep `to_standardize=False`; its standardize() is aggressive (sign-asymmetric rescale + clip).
- Labels ALWAYS divided by auto-computed `labels_scale` (v3 :623, no flag) — and train/test scales
  DIFFER slightly ([123.73,30.94,6.59,1.85] vs [123.61,31.01,6.51,1.85]) → pin `labels_scale=`
  explicitly across discovery/validation/baseline.
- Generator batch plan counts rows BEFORE its dropna → silent row loss (737 rows on the 2_5 set);
  fix by dropna in the prefilter. Generator files are sorted LEXICOGRAPHICALLY (part.0,1,10,...).
- **Zero changes to OptimizedDataGenerator_v3/prepare_tfrecords needed** — all-101 via
  `time_stamps_override=list(range(101))`, noise via the NoisyGen subclass pattern.
- Only repo edit: AnnealingScheduler isinstance guard (:34) → `hasattr(layer,'log_k')`.

**JOINT optimization (user decision 2026-07-02, overrides the earlier staged design):** the
discovery model is Input(16,16,101) → SoftRouterLayer → SoftQuantizeLayer → backbone, with
BOTH layers annealed together (two AnnealingSchedulers, same cosine) and thresholds starting
from the random [25,160] window — NOT from the previously-found optimum [12.09,22.72,52.73].
Rationale: thresholds are slice-dependent (different slices, different amplitudes), so fixed
thresholds would bias the router. Discovery outputs BOTH [i1,i2] and thresholds; the
validation retrain re-learns thresholds on fixed [i1,i2] as a consistency check. Fallback if
joint runs stick repeatedly: staggered anneals (ask the team first).

**Noise decision (2026-07-03):** i.i.d. per sample for THIS study (user); a colleague is
separately working on the correlated-noise variant. `make_tfrs_all101.py` has a
`NOISE_MODE="corr200ps"` stub if that ever needs merging.

**Generator bug found (2026-07-03):** `OptimizedDataGenerator_v3.process_file_parallel`
collects per-file row counts via `as_completed()` (v3.py:331) — completion order, not file
order — so with max_workers>1 the counts land on the WRONG files and tail rows are silently
truncated (this, not NaNs, caused the 737-row loss in the production 2_5 TFRs; data written
is internally consistent, only tails dropped). Workaround in make_tfrs_all101.py:
max_workers=1. Proper 1-line fix (iterate `futures` in submission order) proposed to user,
not yet applied (generator frozen).

**Pipeline state (2026-07-02):** SoftRouterLayer built + smoke 9/9; full 80+20 all-101 TFRs
built (152,037+37,919 events, row-exact, pinned labels_scale) at
`.../shuffled_3d/TFR_files_all101_noise_contained_discovery/`. A detached ORCHESTRATOR
(`scripts/distillation/orchestrate_discovery.sh`) owns the chain: verify → GPU sanity →
launch `run_softrouter_discovery.py --epochs 1000 --target 4` (user chose 1000 ep; noise
kills double-descent, NLL plateaus ~-30K) → auto-report via `make_softrouter_study.py`.
Status file: `runs/softrouter_discovery/orchestrator.log`. Do NOT launch discovery
manually — the orchestrator owns it (idempotent pgrep guard). ~14h/seed est.

**Open before launch:** GPU memory at batch 5000 with 101-channel input (ask before
rebatching, per [[ask-before-launching-runs]]). Validation = standard 2-slice retrain [i1,i2]
vs [11,26]. See [[threshold-opt-2_5-noise-contained]], [[new-3srb-dataset-conventions]].

**TEAM DECISION (2026-07-11 brainstorm, deck in [[softrouter-brainstorm-deck]]; user first said
option C — slip of tongue, corrected to D):** try **Option D = SIMPLE (Ahmed et al., ICLR 2023)**
next and compare against the slot-softmax (Option A) discovery. Motivation: A's best-vs-final
checkpoint pair DRIFT (3/4 seeds) + plateau-hopping = argmax tie-breaking pathology; D samples
the pair exactly and back-props through exact marginals, so ties are resolved by loss evidence
with deterministic gradients. Design: ONE theta vector (101); pair distribution
p({a,b}) prop. exp(theta_a+theta_b) over the 5,050 pairs (exact categorical sampling — trivial
at k=2); mu_i = w_i(S1-w_i)/Z closed form (sum mu = 2 invariant); backward via custom gradient
d(theta) = Cov(z) @ dL/dz where Cov diag = mu_i(1-mu_i), off-diag = p_ij - mu_i mu_j.
NO selection anneal, NO tau, NO duplicate mask — AnnealingScheduler drives only SoftQuantize.
Slot-ordering convention for the (16,16,2) readout: ascending index (early slice = channel 0,
matches production 2-slice semantics); dL/dz for unselected slices = mean of the two slot
sensitivities (implementation choice — document it). Inference = top-2 of theta; mu profile
reported as the calibrated importance map (directly answers pair-vs-window). Compare vs A on:
NLL (-28.7..-29.1K), cross-seed pair/window agreement, best-vs-final drift, pair-visit
concentration, mu landscape. Option C (Gumbel top-k) remains the fallback if D misbehaves.

**OPTION D BUILD + LAUNCH (2026-07-11):** built via 5-agent workflow (sonnet spec / fable impl /
opus adversarial tests / sonnet study script / fable verify), SMOKE 9/9 PASS. Files:
`two_bit_optimization_helpers/SimpleRouterLayer.py` (exact pair sampling, closed-form mu,
Cov(z)@dz custom gradient, visits counter, NO annealer), model `ViT_Max_SimpleRouter` registered
(NOTE: models file + train.py live under two_bit_optimization_helpers/, not repo root),
`scripts/distillation/{run_simplerouter_discovery.py, orchestrate_simplerouter.sh,
make_simplerouter_study.py, smoke_simplerouter.py}`. Orchestrator launched detached 2026-07-11
(~02:47): sanity gate -> 1000 ep x TARGET 6 converged seeds -> auto-report per seed.
Status: `runs/simplerouter_discovery/orchestrator.log` + `reseed.log`; REPORT.md auto-updates.
~14 h/seed => ~3.5 days. Do NOT relaunch manually (pgrep-guarded orchestrator owns it).
result.json key `final_slot_commitment` = [mu[i1], mu[i2]] (SIMPLE analog).

**FIRST D SEED + TIMING AUTOPSY (2026-07-11):** seed 1042: pair [78,82], best NLL -28,236@913,
mu stayed SPREAD (top5 all ~5.3-5.8%, late window 77-88) — honest degeneracy measurement, but
also means the backbone trained as a weight-shared multi-pair supernet -> per-pair NLL likely
UNDERSTATES a committed retrain (A's committed runs: -28.7..-29.1K). Timing autopsy
(`docs/figs/fig_timing_autopsy.png`, seed-pixel shape-variation/noise): CHARGE (3sr, 80e) has
timing SNR ~15 at earliest slices; VOLTAGE (3srb 200ps-shaped, 4.64mV) peaks only ~5 at
t~0.25 (~slice 25 — the hand-picked [11,26] region was the ANALOG-timing optimum!) and <1
past t~0.6. Chain: 200ps shaping erodes timing 3x AND the 2-bit ADC (thresholds tuned to late
amplitudes) crushes the few-% shape differences -> router (optimizing POST-quantization NLL)
rationally goes late. Explains why charge dataset could plausibly reach -50K (user memory;
NO on-disk run confirms it — archaeology found only 3srb no-noise at -52K; charge-era
vit_max_run best was -35.6K@30ep, thresholds in electrons [286,674,1559]).
**HANDOFF (2026-07-27, pre-compaction):**
- Option D FINAL: 6/6 seeds, slice 78 in ALL (pairs [78,82]x2,[78,80],[73,78],[78,89]x2), NLL
  -28,154..-29,124, zero best-vs-final drift. Best seed 4042 [73,78]: perf sx 6.19um sy 1.72um
  (best of any router seed), angles STILL collapse (cot pulls 1.37) — late slices buy position
  not angle. Perf driver: `make_perf_plots_simplerouter.py`, figs `runs/perf_plots_router/
  {summary,pull}_simple_seed4042.png` (landed there, not in own dir — cosmetic).
- Deck: 33 slides live :8900 (softrouter.atdeck). Slide 23 = mu profile + per-seed table;
  24 = training history; 25 = best-seed perf. mu = MARGINAL inclusion prob per slice (sums
  to 2), NOT joint pair prob. Ports 8787 (jetbuster) + 8792 (smartpix-advanced) are OTHER
  projects' servers — never pkill by pattern, kill by exact PID only (killed 8787 once).
- atrain-slides repo at ab92a14 + UNCOMMITTED local patch (P0 in ATRAIN_FEEDBACK.md): CRDT
  WS auth is origin-pinned -> permanently offline behind port-forwards + silent out-of-band
  clobber (user LOST edits; may be recoverable via editor History panel). Local fix: deckd
  GET /session-token (ACAO stripped) + editor fetchSessionToken() on the WS URL
  (http.ts/ws.ts/deckdClient.ts/deckdSync.ts). Verified by handshake test. A git pull may
  conflict with these 4 files. Rule: use the running server/CRDT for deck edits, never raw
  deck.json writes while user edits.
- Published artifact (whole-project dossier, real waveform+mu data):
  https://claude.ai/code/artifact/a66cd1ec-7ce3-47a0-8a26-41d55285d56e
- NEXT: user is sharing HARSHUL'S MDMM REPO after compaction -> combine MDMM (constrained
  training, e.g. pull-width==1 on cot-alpha/beta) with SoftTimeRouter. Key hypothesis to
  test: slice choice is CONDITIONAL on the objective — an angle-constrained loss may shift
  mu toward the timing region (~slices 11-31, where analog timing SNR peaks ~5) instead of
  the amplitude plateau (70-90). AUDIT DONE 2026-07-27, see [[mdmm-harshul-audit]] — TWO
  pre-audit hypotheses were WRONG: (a) the corr>=0.5 constraint is SATISFIABLE on contained
  data (Harshul: 5/6 MDMM runs -36.7..-38.9K, angle corr 0.99 — no lambda blowup), so the
  collapse is a basin problem, not an info ceiling; (b) his QConv2D stall is a quantizer-init
  pathology (10/10 seeds stuck at IDENTICAL ~98,980; alpha=1 4-bit crushes Glorot init),
  not lambda blowup.
- Still unrun: the validation retrain [78,82]-vs-[11,26] (standard 2-slice pipeline).

**CLEAN-CONTAINED CONTROL launched 2026-07-11 ~20:49** (user-approved): TFR_files_2_5_clean_contained
(sigma=0 clone of the noise build) + run_part1_long_2_5_clean_contained.py (5000ep x 3 seeds)
via orchestrate_2_5_clean.sh (hardened driver_running guard). Prediction: recovers most of
-52K; gap vs -52K = containment cost, gap vs -28K = noise cost. Report:
runs/part1_long_2_5_clean_contained/REPORT.md.
