---
name: run-full-epoch-budgets
description: When the user sets an epoch budget, run it to completion; don't re-litigate mid-run or keep offering early-stop opinions
metadata:
  type: feedback
---

During O24b (2026-09-02) I declared a "verdict" at ep 2,000/4,800 of a 10,000-
epoch run and repeatedly offered stop/commit options. The user: "Did I ask your
opinion? Run them for 10K epochs, let's see."

**Why:** they explicitly reason from late grokking — models jumping to the
global minimum at high epoch counts — and our own parent run improved at
ep ~4,999 after 768 stale epochs. Mid-run plateau extrapolations undercount
that possibility.

**How to apply:** state the risk once when the budget is set, then run the full
budget. Report milestones factually; save recommendations for when the runs
finish or when something is actually broken (crash, saturation, wrong config).
Killing a run remains right for genuine defects — see the three archived O24
traps in [[handoff-o24-campaigns]] — not for "unlikely to improve".
