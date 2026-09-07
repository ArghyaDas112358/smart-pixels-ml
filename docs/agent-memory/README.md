# Agent memory (ported from the Claude session, 2026-09-07)

One fact per file. Frontmatter gives type (user / feedback / project / reference).
Newest project state: HANDOFF in ../handoff/. Files marked *handoff-* are dated snapshots; the newest wins.

- [Handoff cold-start 2026-08-22](handoff-coldstart-2026-08-22.md) — READ FIRST: cold search found (10,21), beats the hand-imposed pair by ~40% on angles; live fleet + traps
- [SmartPixels repo layout](smartpixels-repo-layout.md) — the new 2bit notebook repo vs the legacy package fork, and how they differ
- [SmartPixels data locations](smartpixels-data-locations.md) — where the parquet datasets live and which one the new notebook uses
- [SmartPixels fresh env](smartpixels-fresh-env.md) — the conda env for the new notebook and how to launch training
- [SmartPixels distillation findings](smartpixels-distillation-findings.md) — why soft-KL ViT->tiny-MoE distillation loses to standalone, and the TBR/DeiT direction now running
- [Use full datasets](use-full-datasets.md) — never subset a dataset without explicit permission; full statistics by default
- [Act on ops decisions](act-on-ops-decisions.md) — cancel/resubmit/restructure jobs without asking; still ask on experiment MEANING
- [Ask before launching runs](ask-before-launching-runs.md) — consult the user before launching/killing runs or changing data/params beyond the explicit ask
- [New 3srb dataset conventions](new-3srb-dataset-conventions.md) — batch 5000, time samples [11,26] of 101, 80/20 split
- [Threshold optimization 3srb archive](threshold-optimization-3srb-archive.md) — where Part-1 threshold results live + recommended thresholds [4.15, 13.01, 47.24]
- [Threshold opt 2_5 noise-contained](threshold-opt-2_5-noise-contained.md) — Part-1 thresholds [12.09, 22.72, 52.73] for the new mV baked-noise contained 2_5 dataset; AbortOnStuck 1e5→1e4 fix
- [Threshold opt 1_6 iid](threshold-opt-1-6-iid.md) — [6,31] slices: [12.64, 28.17, 69.89] (5-seed median); completes the three-regime T-vs-slice picture
- [Soft router design](soft-router-design.md) — penalty-free slot-softmax SoftRouterLayer for time-slice discovery; code-verified normalization rules; generator stays frozen
- [MDMM Harshul audit](mdmm-harshul-audit.md) — his MinCorr MDMM breaks the angle collapse on contained data (local minimum, not info ceiling); port notes + agreed router experiment
- [Handoff deep-head O17/O18](handoff-deep-head-o17-o18.md) — READ FIRST: the σα-13° deep-head breakthrough, O18 in flight, full O11–O18 ledger, all ops traps
- [Handoff 2026-08-03](handoff-2026-08-03.md) — older: O11 five-seed results, the "satisfied not stuck" finding, O4 in flight, two memory leaks, Gautschi facts, weekly deck
- [Reports stay in repo](reports-stay-in-repo.md) — never write reports/archives to /work/projects/SmartPixML unless explicitly asked; default to the repo
- [Git push policy](git-push-policy.md) — push freely to the user's own fork (myfork); never to a collaborator's repo without their per-push say-so
- [Full image paths](full-image-paths.md) — always print absolute paths for figures so they're clickable in VS Code
- [atrain WS port-forward fix](atrain-ws-port-forward-fix.md) — why decks only updated on refresh; the /session-token fix pushed to GitHub
- [Edit atrain decks through the CRDT](atrain-edit-through-crdt.md) — never rewrite deck.json on a live deck; use atrain_op.mjs so we can co-edit
- [SoftRouter brainstorm deck](softrouter-brainstorm-deck.md) — the interactive atrain-slides deck (port 8900), its 4 widgets, and the upstream textColor/autoplay fixes
- [AF loadavg is the host's](af-loadavg-is-the-host.md) — /proc/loadavg shows the Geddes node, not your pod; use cgroup cpu.stat to judge contention
- [Handoff O24 campaigns](handoff-o24-campaigns.md) — READ FIRST: cold+warm bias-loss 10K campaigns in flight, anchor-8 migration, the three launch traps
- [Run full epoch budgets](run-full-epoch-budgets.md) — user-set budgets run to completion; no mid-run early-stop opinions
