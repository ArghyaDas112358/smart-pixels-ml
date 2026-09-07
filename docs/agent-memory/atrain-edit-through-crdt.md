---
name: atrain-edit-through-crdt
description: Never rewrite deck.json on a live atrain deck — edit through the CRDT op API so the user and Claude can work on the same slides
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7a0a041e-b87d-47e5-ae22-8b5c100f4193
  modified: 2026-08-03T16:38:23.862Z
---

**Edit running atrain decks through the CRDT, never by writing `deck.json`.**
User's ask (2026-08-03): "please make the changes only through the crdt thingy
not directly into the deck ... that way we both can work on the same slides".

**Why:** writing `deck.json` takes deckd's chokidar out-of-band path, which
re-seeds the CRDT from the file and broadcasts a full resync — last-writer-wins,
so anything the user edited in the browser since the last build is destroyed.
`build_weekly_deck.mjs` is especially dangerous: it mints a FRESH element id for
every element on every run, so a rebuild replaces the entire deck.

**How to apply:** use `/work/users/das214/atrain-slides/scripts/atrain_op.mjs`
(written for this). `getDeck(port, deckDir)` reads deckd's live in-memory deck;
`ops(port, [...], deckDir)` POSTs to `/op` → `store.apply` → `crdt.applyLocal`,
which merges and lets deckd persist deck.json itself. Look elements up BY ID in
the live deck — never regenerate ids. Op vocabulary is in
`packages/ops/src/ops.ts` (`add_element`, `update_element`, `delete_element`,
`set_cell`, `table_structure`, `set_slide_props`, `add_slide`, …).
Verified: two authors editing different elements both survive; rev increments.
Build scripts are SEED-ONLY, for creating a deck that nobody is editing yet.

**Serve from the NEWER checkout:** `/home/das214/atrain-slides` (not
`/work/users/das214/atrain-slides`, whose `packages/editor/dist` was stale from
07-27 and was why live updates needed a manual refresh). Decks themselves live
in `/work/.../atrain-slides/*.atdeck` and are passed as an argument.
Tables have a native `table` element — use it, don't hand-roll shapes+text.
See [[softrouter-brainstorm-deck]], [[handoff-2026-08-03]].
