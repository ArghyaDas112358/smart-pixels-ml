---
name: full-image-paths
description: "Always print full absolute paths for generated images/figures so they're clickable in VS Code"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: bab40003-6e1e-4266-ad76-90edb4e81a2c
---

When reporting any generated image/figure/plot, write the **full absolute path** (e.g. `/work/users/das214/SmartPixels/smart-pixels-ml/runs/.../fig.png`), not a relative path.

**Why:** the user opens figures by clicking the path in the VS Code terminal; a full path opens the image directly, a relative path does not.

**How to apply:** any time I save or mention a .png/.pdf/figure, give its absolute path on its own line so it's click-to-open. Applies to all deliverables. See [[ask-before-launching-runs]].
