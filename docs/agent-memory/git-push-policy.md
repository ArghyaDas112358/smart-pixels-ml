---
name: git-push-policy
description: "Push freely to the user's own fork; NEVER to a collaborator's repo without explicit per-instance approval"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7a0a041e-b87d-47e5-ae22-8b5c100f4193
  modified: 2026-08-02T00:20:09.172Z
---

**Push/commit only to the user's OWN remote. Never to anyone else's repo without
asking first, every time.**

In `smart-pixels-ml` that means:
- `myfork` = https://github.com/ArghyaRanjanDas/smart-pixels-ml.git — the user's
  own fork. **Free to commit and push here without asking.**
- `origin` = https://github.com/davidgjiang/smart-pixels-ml.git — a collaborator's
  repo (David Jiang). **Never push here** unless the user says so for that
  specific push. Same rule for any other collaborator remote (Harshul's fork,
  Shiqi's `symbolic` branch repo, etc.).

**Why:** a push into someone else's repository is outward-facing and lands in
another person's workspace — it is theirs to authorize, not the assistant's.
Confirmed by the user 2026-08-02 after a two-branch push to `myfork`:
"yes absolutely dont ever commit to other repo wihtout my call ...you are free
to commit to my repo".

**How to apply:** when a repo has multiple remotes, check `git remote -v` and
identify the user's own before pushing. Push to it, then TELL the user what was
pushed and explicitly offer the collaborator remote as a separate decision rather
than doing it. Branch layout established on 2026-08-02: `softtimerouter` for the
router/MDMM line, `symbolic_distillation` for the distillation line, both on a
shared base commit carrying the data-generator fix. See
[[smartpixels-repo-layout]] and [[reports-stay-in-repo]].
