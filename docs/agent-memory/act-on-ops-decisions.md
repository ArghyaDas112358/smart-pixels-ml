---
name: act-on-ops-decisions
description: "Don't ask permission for operational/infrastructure fixes — cancel, resubmit, restructure jobs and report after"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7a0a041e-b87d-47e5-ae22-8b5c100f4193
  modified: 2026-08-05T17:08:48.605Z
---

**Act on operational decisions; don't wait for a go-ahead.** User, 2026-08-04:
*"you shodul do this things autmatically and nto ait for e"* — after I had twice
asked "say go and I'll cancel the thrashing jobs and resubmit".

**Why:** these are reversible infrastructure calls (Slurm job restructuring,
memory caps, killing a thrashing or diverging run, resubmitting). Waiting burns
GPU-hours while a broken configuration keeps running, and the user has to babysit.

**How to apply:** fix it, then report what was done and why. Covers cancelling /
resubmitting Slurm jobs, changing memory caps and concurrency, killing dead or
diverging seeds, restarting damaged runs, monitor tuning.

**Still ask first for:** changing the scientific setup — constraint quantity,
targets, loss definition, datasets, epoch budgets — i.e. anything that changes
what the experiment MEANS rather than how it is scheduled. That distinction is
the point; see [[ask-before-launching-runs]], which this narrows rather than
replaces.
