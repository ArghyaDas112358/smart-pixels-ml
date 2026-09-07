---
name: threshold-opt-1-6-iid
description: "Part-1 thresholds for the 1ns/6ns slices [6,31], iid noise: [12.64, 28.17, 69.89] (5-seed median) — completes the three-regime picture"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7a0a041e-b87d-47e5-ae22-8b5c100f4193
---

Part-1 threshold study for the **1 ns / 6 ns slice pair = time indices [6, 31]**
(mapping: index = 5*ns + 1) is DONE (2026-07-08). Same recipe as the 2_5 study: full
80+20 contained dataset, i.i.d. N(0, 4.64 mV) baked noise, ViT_Max + SoftQuantize,
5000 ep x 5 converged seeds, cosine k 1->67.

**RECOMMENDED thresholds (5-seed median): [12.64, 28.17, 69.89]** — std [0.21, 0.32, 1.01],
median best NLL -28,303/batch. Converged seeds [1042, 2042, 5042, 6042, 7042]; seeds
42/3042/4042 stuck at the ~99,113 ceiling and were auto-skipped in 16 ep each (stuck rate
3/8 attempts — much higher than 2_5's 0/5; [6,31] has more dead init basins).

**Three-regime picture (all iid, same recipe):** T0 is universal (~12-14 = noise floor);
T1/T2 scale with slice amplitude:
- [11,26] (2ns/5ns):  [12.18, 22.94, 52.75]  NLL ~ -28.0K
- [6,31]  (1ns/6ns):  [12.64, 28.17, 69.89]  NLL ~ -28.3K
- router late (~59-97): [13.78, 44.47, 105.05] NLL ~ -28.7..-29.1K
Later/larger-amplitude slices carry slightly more info (NLL ordering).

**Locations:** TFRs `.../shuffled_3d/TFR_files_1_6_iid_contained/` (das214; distinct from
Harshul's TFR_files_1_6_noise_corr_contained). Run + REPORT.md + figs:
`runs/part1_long_1_6_iid_contained/`. Scripts: `make_tfrs_contained_1_6_noise.py`,
`run_part1_long_1_6_iid_contained.py`, `make_1_6_study.py`, `orchestrate_1_6.sh`.
See [[threshold-opt-2_5-noise-contained]], [[soft-router-design]], [[use-full-datasets]].
