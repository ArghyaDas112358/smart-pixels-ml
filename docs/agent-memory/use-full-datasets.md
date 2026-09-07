---
name: use-full-datasets
description: Never train/generate on a subset of a dataset without explicit permission — default to the FULL dataset
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7a0a041e-b87d-47e5-ae22-8b5c100f4193
---

Do NOT use a subset of any dataset (files, events, splits) without the user's explicit
permission — default to the FULL dataset every time.

On 2026-07-03 I defaulted the all-101 discovery TFRs to a 20+5-file subset (of 80+20) to
save IO/epoch time. The user corrected firmly: "don't use a subset of any dataset without
my permission, this is very important — we have this large dataset because we want enough
stats; a lot of computation resources have been spent on having those datasets and you are
not using those, that is a shame."

**Why:** the datasets are expensive products (simulation + shuffling + uploads); statistics
are the whole point. Saving wall-clock by silently cutting stats is a false economy to them.

**How to apply:** any pipeline step (TFR generation, training, validation) uses ALL
available files/events by default. If compute/IO makes a subset genuinely attractive,
present the trade-off (time/disk vs stats) and ASK first — subsetting is the user's call,
never mine. Smoke tests on 1-2 files are fine (they verify machinery, not physics).
See [[ask-before-launching-runs]], [[soft-router-design]].
