"""
Build the interactive lattice-scrubber embed (template + script + inlined data)
for the O21a.v2-cold deck, as JSON for scripts/add_lattice_widget.mjs to push.

The embed renders inside atrain's sandboxed iframe (allow-scripts, CSP
default-src 'none'), so EVERYTHING must be inline: the per-epoch sampled pairs
are packed into one compact string per seed and the occupancy surface is
computed in-browser per frame. No network, no assets.

Data packing: one "i.j" per epoch, joined by ";" — epochs are consecutive from
the first logged epoch, so only the pair needs storing.

  python make_lattice_widget.py            # writes lattice_widget.json
"""
import os, csv, json

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
RUN = os.path.join(R, "runs", "o21v2a2_pairlattice")
OUT = os.path.join(R, "runs", "perf_plots_o21v2", "lattice_widget.json")
SEEDS = [22042, 22142, 22242]
COLOR = {22042: "#7c3aed", 22142: "#0e7490", 22242: "#b45309"}


def pack(seed):
    p = os.path.join(RUN, f"seed_{seed}", "router_epochs.csv")
    rows = []
    for r in csv.DictReader(open(p)):
        try:
            rows.append((int(r["epoch"]), int(r["i1"]), int(r["i2"])))
        except (ValueError, KeyError):
            continue
    rows.sort()
    e0 = rows[0][0]
    return e0, ";".join(f"{min(a,b)}.{max(a,b)}" for _, a, b in rows)


data = {}
for s in SEEDS:
    e0, packed = pack(s)
    data[str(s)] = {"e0": e0, "n": packed.count(";") + 1, "pairs": packed, "color": COLOR[s]}
print({k: (v["e0"], v["n"]) for k, v in data.items()})

TEMPLATE = """
<style>
  /* The sandbox sets color-scheme:light dark on :root, so in a dark-mode
     browser the UA paints form-control TEXT white. Every colour here is
     therefore explicit -- nothing inherits, and the widget pins itself to the
     light scheme so the range track/thumb stay visible too. */
  :root{color-scheme:light}
  html,body{margin:0;padding:0;background:transparent;
    font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:#111318}
  .wrap{display:flex;flex-direction:column;height:100vh;box-sizing:border-box;padding:6px 10px 10px}
  .row{display:flex;gap:14px;flex:1;min-height:0}
  .cell{flex:1;display:flex;flex-direction:column;align-items:center;min-width:0}
  canvas{width:100%;height:auto;image-rendering:pixelated;border:1px solid #dfe4e9;border-radius:6px;background:#fff}
  .cap{font-size:15px;font-weight:700;margin-top:6px}
  .ctl{display:flex;align-items:center;gap:12px;margin-top:10px}
  button{font:inherit;font-size:15px;padding:5px 14px;border-radius:7px;border:1px solid #c8d0d7;
    background:#ffffff;color:#111318;cursor:pointer;-webkit-appearance:none;appearance:none}
  button:hover{background:#f2f5f7;color:#111318}
  input[type=range]{flex:1;accent-color:#7c3aed;height:22px}
  .ep{font-variant-numeric:tabular-nums;font-weight:700;font-size:16px;min-width:190px;text-align:right;color:#111318}
  .hint{font-size:12px;color:#5b6470;margin-top:2px}
  .cbar{display:flex;align-items:center;gap:8px;margin-top:8px}
  .cbar canvas{width:260px;height:12px;border:1px solid #dfe4e9;border-radius:3px;image-rendering:auto}
  .cbl{font-size:12px;color:#111318;font-weight:600}
  .cbnote{font-size:12px;color:#5b6470}
  .ctl{background:rgba(255,255,255,.92);border-radius:8px;padding:6px 10px}
</style>
<div class="wrap">
  <div class="row" id="row"></div>
  <div class="cbar">
    <span class="cbl">low</span>
    <canvas id="cb" width="256" height="12"></canvas>
    <span class="cbl" id="cbmax">high</span>
    <span class="cbnote" id="cbnote"></span>
  </div>
  <div class="ctl">
    <button id="play">Pause</button>
    <input type="range" id="sl" min="0" max="100" value="0" step="1">
    <span class="ep" id="ep">epoch 0</span>
  </div>
  <div class="hint" id="hint">One slider drives all three seeds. Playing automatically — drag it anywhere; playback continues from there.</div>
</div>
"""

SCRIPT = """
const DATA = __DATA__;
const T = 101, BIN = 2, NB = Math.ceil(T / BIN), WIN = 400;
const SCALE = "__SCALE__";   // "log" | "linear"
const seeds = Object.keys(DATA);
// unpack once
for (const s of seeds) {
  const parts = DATA[s].pairs.split(";");
  const lo = new Int16Array(parts.length), hi = new Int16Array(parts.length);
  for (let k = 0; k < parts.length; k++) {
    const d = parts[k].indexOf(".");
    lo[k] = +parts[k].slice(0, d); hi[k] = +parts[k].slice(d + 1);
  }
  DATA[s].lo = lo; DATA[s].hi = hi;
}
const NEP = Math.max(...seeds.map(s => DATA[s].n));

// --- magma-ish ramp (inline: no libraries in the sandbox) ---
const RAMP = [[0,0,4],[28,16,68],[79,18,123],[129,37,129],[181,54,122],
              [229,80,100],[251,135,97],[254,194,135],[252,253,191]];
function magma(t){
  t = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(t)), f = t - i;
  const a = RAMP[i], b = RAMP[i+1];
  return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f];
}
// separable box blur ~ gaussian (3 passes), radius r on an NBxNB grid
function blur(g, r){
  if (r < 1) return g;
  const n = NB, tmp = new Float32Array(n*n), out = Float32Array.from(g);
  for (let pass = 0; pass < 3; pass++){
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++){
      let s = 0, c = 0;
      for (let k = -r; k <= r; k++){ const xx = x+k; if (xx>=0 && xx<n){ s += out[y*n+xx]; c++; } }
      tmp[y*n+x] = s/c;
    }
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++){
      let s = 0, c = 0;
      for (let k = -r; k <= r; k++){ const yy = y+k; if (yy>=0 && yy<n){ s += tmp[yy*n+x]; c++; } }
      out[y*n+x] = s/c;
    }
  }
  return out;
}

const row = document.getElementById("row");
const cells = {};
for (const s of seeds){
  const d = document.createElement("div"); d.className = "cell";
  const cv = document.createElement("canvas"); cv.width = NB; cv.height = NB;
  const cap = document.createElement("div"); cap.className = "cap";
  cap.textContent = "seed " + s; cap.style.color = DATA[s].color;
  d.appendChild(cv); d.appendChild(cap); row.appendChild(d);
  cells[s] = { cv, ctx: cv.getContext("2d"), img: cv.getContext("2d").createImageData(NB, NB), cap };
}

function draw(epIdx){
  for (const s of seeds){
    const D = DATA[s], c = cells[s];
    const last = Math.min(epIdx, D.n - 1);
    const first = Math.max(0, last - WIN);
    const g = new Float32Array(NB*NB);
    for (let k = first; k <= last; k++){
      const x = Math.floor(D.lo[k]/BIN), y = Math.floor(D.hi[k]/BIN);
      g[y*NB + x] += 1;
    }
    const b = blur(g, 2);
    let mx = 0; for (let i = 0; i < b.length; i++) if (b[i] > mx) mx = b[i];
    const px = c.img.data;
    for (let y = 0; y < NB; y++) for (let x = 0; x < NB; x++){
      const v = mx > 0 ? b[y*NB+x]/mx : 0;
      // log-ish stretch so the tail is visible, like the static magma figures
      const col = magma(SCALE === "log" ? (v > 0 ? Math.log10(1 + 99*v)/2 : 0) : v);
      const o = 4*((NB-1-y)*NB + x);          // flip y so t_j grows upward
      px[o] = col[0]; px[o+1] = col[1]; px[o+2] = col[2]; px[o+3] = 255;
    }
    c.ctx.putImageData(c.img, 0, 0);
    c.cap.textContent = "seed " + s + "  (" + D.lo[last] + ", " + D.hi[last] + ")";
  }
  document.getElementById("ep").textContent =
    "epoch " + (DATA[seeds[0]].e0 + epIdx) + " / " + (DATA[seeds[0]].e0 + NEP - 1);
}

// --- playback: one GLOBAL clock for all three panels ------------------------
// Autoplays; while the user drags, the clock follows the slider; when they let
// go, playback simply CONTINUES from wherever they left it.
const sl = document.getElementById("sl"), btn = document.getElementById("play");
sl.max = String(NEP - 1);
const STEP = Math.max(1, Math.round(NEP / 240));   // ~240 frames per loop
let clock = 0;              // the single position driving all three canvases
let scrubbing = false;
let paused = false;

function tick(){
  if (!paused && !scrubbing) {
    clock = (clock + STEP) % NEP;
    sl.value = String(clock);
    draw(clock);
  }
  setTimeout(tick, 90);
}
sl.addEventListener("input", () => {
  scrubbing = true;
  clock = +sl.value;        // the clock IS the slider — no snapping back
  draw(clock);
  document.getElementById("hint").textContent =
    "Scrubbing — playback continues from here when you let go.";
});
const release = () => {
  if (!scrubbing) return;
  scrubbing = false;
  clock = +sl.value;        // resume from exactly where the user stopped
  document.getElementById("hint").textContent =
    "Playing automatically — drag the slider anywhere; playback continues from there.";
};
sl.addEventListener("change", release);
sl.addEventListener("pointerup", release);
sl.addEventListener("pointercancel", release);
sl.addEventListener("blur", release);
btn.addEventListener("click", () => {
  paused = !paused;
  btn.textContent = paused ? "Play" : "Pause";
  document.getElementById("hint").textContent = paused
    ? "Paused — drag the slider to explore any epoch."
    : "Playing automatically — drag the slider anywhere; playback continues from there.";
});
// --- colour bar (same ramp + same stretch as the panels) --------------------
(function paintCbar(){
  const cb = document.getElementById("cb"), cx = cb.getContext("2d");
  const im = cx.createImageData(cb.width, cb.height);
  for (let x = 0; x < cb.width; x++){
    const v = x / (cb.width - 1);
    const col = magma(SCALE === "log" ? (v > 0 ? Math.log10(1 + 99*v)/2 : 0) : v);
    for (let y = 0; y < cb.height; y++){
      const o = 4*(y*cb.width + x);
      im.data[o] = col[0]; im.data[o+1] = col[1]; im.data[o+2] = col[2]; im.data[o+3] = 255;
    }
  }
  cx.putImageData(im, 0, 0);
  document.getElementById("cbnote").textContent = SCALE === "log"
    ? "log stretch — structure everywhere, each panel scaled to its own busiest bin"
    : "linear — mass in proportion, each panel scaled to its own busiest bin";
})();

draw(0);
tick();
"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
packed = json.dumps(data, separators=(",", ":"))
for scale in ("log", "linear"):
    payload = {"template": TEMPLATE,
               "script": SCRIPT.replace("__DATA__", packed).replace("__SCALE__", scale)}
    path = OUT.replace(".json", f"_{scale}.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    print("wrote", path, f"({os.path.getsize(path)/1024:.0f} kB)")
