---
name: reports-stay-in-repo
description: "Where to write reports/study archives: keep them in the repo, never in /work/projects/SmartPixML unless explicitly asked"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7a0a041e-b87d-47e5-ae22-8b5c100f4193
---

Do NOT write reports / study archives / slide decks into `/work/projects/SmartPixML/`.
Put them under the repo: `/work/users/das214/SmartPixels/smart-pixels-ml/`.

On 2026-07-01 I built the 2_5 threshold report at
`/work/projects/SmartPixML/threshold_optimization_2_5_noise_contained/`; the user
told me to move it into the repo and "never put report in any of the /work/projects/
SmartPixML/ dir until I ask you explicitly to do it for."

**Why:** `/work/projects/SmartPixML/` is the shared team space; the user wants their
analysis reports kept in their own repo unless they explicitly choose to publish to
the shared dir. (Note: the earlier clean-data study `threshold_optimization_3srb/`
already lives in /work/projects — that's a pre-existing exception, don't add to it.)

**How to apply:** default all generated reports/archives to the repo. Only write to
`/work/projects/SmartPixML/` when the user explicitly asks for that specific run.
Data copies (datasets, TFRs) the user explicitly asks to place there are fine; this
rule is about REPORTS. See [[threshold-opt-2_5-noise-contained]], [[ask-before-launching-runs]].
