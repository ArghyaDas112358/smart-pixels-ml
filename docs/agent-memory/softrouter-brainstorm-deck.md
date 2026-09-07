---
name: softrouter-brainstorm-deck
description: "The interactive SoftTimeRouter brainstorm deck (atrain-slides) — where it lives, how to serve, what the widgets are"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7a0a041e-b87d-47e5-ae22-8b5c100f4193
  modified: 2026-07-28T16:13:23.030Z
---

For the 2026-07-11 SoftTimeRouter team brainstorm I built an interactive ATrain-Slides deck:
- Deck: `/work/users/das214/atrain-slides/softrouter.atdeck` (final: 24 slides, 6 widgets incl. the
  W9 "watch the router learn" full-loop toy on the real waveform; option C=Gumbel top-k and option
  D=SIMPLE each get a concept slide built around `docs/figs/fig_router_dataflow.png` — a TikZ tower
  (source: SmartPixels/.claude/skills/tikz-diagrams/build/router_dataflow.tex) defining theta/xi/p(S)/mu
  and drawing both backward routes). Builder: `scripts/build_softrouter_deck.mjs` (regenerates all).
  Key pedagogy rule learned: DEFINE every symbol in plain words on-slide; ground toys in the real
  waveform + "weights and gradient" language, not abstract logit bars.
- Serve (PORTS MOVED 2026-07-28 at user request): `npm run atrain -- softrouter.atdeck --port 8910
  --no-open`; threshold-talk deck `smartpix.atdeck` on 8911. (Old 8900/8899 servers killed.)
  Offline backup: `softrouter_backup/{index,reveal}.html` in the repo.
- Widgets (script embeds, theme-aware, autoplay on `atrain-autoplay` postMessage): W1 slice-picker on the
  REAL mean seed-pixel waveform (extracted from 3srb test parquet: pulse saturates ~335 mV at slice ~90;
  74 mV @11, 245 @26, 335 @76 — the router's "go late" is visible in raw data); W2 slot-softmax anneal toy;
  W3 SoftQuantize threshold playground; W4 three-regime noise-floor explorer (T0 = 2.6 sigma).
- The upstream atrain-slides repo (github ArghyaRanjanDas/atrain-slides) fixed our ATRAIN_FEEDBACK.md batch
  (commit b720b5d: textColor mark heal, edit data-loss traps, sync defer, undo toast) — canonical colour mark
  is `textColor`, legacy `color` marks are auto-healed by migrate(). Embeds autoplay via the a69a992 convention.
See [[soft-router-design]], [[smartpixels-distillation-findings]].
