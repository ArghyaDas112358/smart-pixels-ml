# smart-pixels-ml — agent router

You are in the SmartPixels ML repo (CMS smart-pixel ASIC: 2 of 101 time slices,
2-bit readout, ViT regression of track parameters). Route yourself:

1. **State of the project** → `docs/handoff/HANDOFF_2026-09-07.md` (newest
   file in `docs/handoff/`). Read it before touching runs, drivers, or results.
   It has the campaign ledger (O21 parent → O22 → O23 → O24), what is running,
   exact resume commands, the code map, and the traps.
2. **Running the campaign driver** → `scripts/distillation/run_simplerouter_mdmm_discovery.py`.
   Always set `SMARTPIX_MODEL_NAME=ViT_MaxDeep_PairLattice` (unset = wrong
   architecture, silently). One `--out` dir per seed. `--extend` resumes.
3. **Evaluating checkpoints** → `scripts/distillation/make_bias_comparison.py`
   (`SMARTPIX_ARMS`, `SMARTPIX_OUT`) and `make_perf_plots_o22.py` (pulls).
   Compare NLLs only as `val_plain_nll`; the training loss is a weighted composite.
4. **Gautschi batch** → `scripts/gautschi/submit_o24_scratch_bias.sbatch`
   (chunk chain, 42 GB cap, resubmit guard). Depot is mounted on both machines.
5. **Figures** → `runs/perf_plots_*`; scripts `scripts/distillation/make_*.py`.
   Print absolute paths for every figure.
6. **Live status without any session state** → `bash scripts/status_o24.sh`
   (both machines, all seeds). Agent-independent follow-ons (the −40.4K
   recovery job, then migrating cold seed 40642 to the A100) run from
   `scripts/o24_followon_watch.sh` under `setsid`; its log is
   `runs/o24_followon.log`. Do not launch duplicates of those two jobs.
7. **Durable facts from earlier agent sessions** → `docs/agent-memory/`
   (one fact per file; `README.md` is the index). The Claude "Bias and
   Ceiling" artifact URL in the handoff is read-only for non-Claude agents;
   the underlying figures are in `runs/perf_plots_*`.
8. **Platform** → the Purdue AF pod: `/proc/loadavg` and `free` show the host;
   use cgroup files. A100 fits 3–4 trainers. Launch long jobs with
   `setsid nohup … &` and wait on the log, not the pid.

Rules from the user: run full epoch budgets (no early-stop opinions mid-run);
never subset a dataset; ask before changing an experiment's meaning, act on
ops (restart/resubmit/kill broken runs) without asking; push to `myfork`
freely, never to `origin`; keep reports in this repo.
