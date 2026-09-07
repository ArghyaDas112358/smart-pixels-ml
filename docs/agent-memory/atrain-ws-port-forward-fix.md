---
name: atrain-ws-port-forward-fix
description: "Why atrain decks only updated on refresh behind a port forward, and the /session-token fix that is now on GitHub"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7a0a041e-b87d-47e5-ae22-8b5c100f4193
  modified: 2026-08-03T19:14:29.890Z
---

**atrain-slides live sync died behind a port forward — fixed 2026-08-03.**

**Symptom:** the deck loaded and looked healthy, but Claude's edits only appeared
after a manual browser refresh.

**Cause:** deckd checks `Origin` on the WebSocket upgrade. A tab reached through a
VS Code port forward / tunnel sends an Origin that is not in the trust list, so
the upgrade was refused **4401 — silently, on both ends**. `GET /deck` has no
Origin check, so HTTP kept working; only the live channel was dead.

**Fix** (branch `fix/ws-auth-behind-port-forward`, commit 5ca9f11, pushed to
`ArghyaRanjanDas/atrain-slides`):
- deckd `GET /session-token` returns the bearer token with
  `Access-Control-Allow-Origin` **stripped** — a cross-origin page still cannot
  read it, so the trust boundary is unchanged.
- editor `fetchSessionToken()` + `wsUrlFor(slug, token)`; `connect()` awaits it
  and re-checks `currentSlug`. A null token falls back to Origin-only auth.
- `ws.ts` now logs a refused upgrade once per origin, naming the cause.
- Tests in `packages/deckd/src/http.test.ts` (token returned, ACAO absent, no
  auth needed). Verified: untrusted origin without token → 4401, with token →
  accepted.

**After changing editor source you MUST `npm run build:editor`** — the served
bundle is `packages/editor/dist`, and a stale dist was the earlier red herring.
Then restart the deck servers.

**Diagnosis technique worth reusing:** playwright-core is vendored in the repo, so
the served editor can be driven headless — load `http://127.0.0.1:<port>/`, watch
`websocket` frames and console, issue a `POST /op` from outside, and assert the
DOM changed without a reload. That is what proved the server was fine.

See [[atrain-edit-through-crdt]] for the op API itself.
