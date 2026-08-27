"""
Interactive TRUE-PROBABILITY scrubber: softmax(phi) per epoch, log colour.

Unlike make_lattice_widget.py (which animates the empirical occupancy of
SAMPLED pairs), every frame here is the router's actual pair distribution,
read from the phi_history.npz snapshots the driver now writes every SNAP
epochs. Probabilities are absolute, so the colour bar carries real units.

PACKING. A frame is 5050 probabilities; inlining them raw would bloat the deck.
Each frame is therefore stored SPARSELY: the top-K pairs by probability, as
(pair index in base36, log-quantised value byte). Everything outside the top-K
renders at the frame floor -- on a log scale that tail is visually flat anyway.
K and the frame count are chosen so the payload stays a few hundred kB.

  python make_prob_widget.py
"""
import os, json
import numpy as np

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
RUN = os.path.join(R, "runs", "o21v2a2_pairlattice")
OUT = os.path.join(R, "runs", "perf_plots_o21v2", "prob_widget.json")
SEEDS = [22042, 22142, 22242]
COLOR = {22042: "#7c3aed", 22142: "#0e7490", 22242: "#b45309"}
T = 101
MAX_FRAMES = 48             # subsample if more snapshots exist (payload budget)
IA, IB = np.triu_indices(T, k=1)
FLOOR = 1e-7                # log-scale floor, also the colour-bar minimum


import base64, csv, h5py


def best_ckpt_frame(seed):
    """psi from best.weights.hdf5 + the epoch it belongs to.

    The ONLY true-probability record that predates the phi logger: Keras kept
    this checkpoint from the best-val epoch, so it is a real earlier sample of
    the learned distribution (one per seed, not a history).
    """
    d = os.path.join(RUN, f"seed_{seed}")
    best, bep = float("inf"), None
    try:
        for r in csv.DictReader(open(os.path.join(d, "history.csv"))):
            try:
                v = float(r["val_loss"])
            except (ValueError, KeyError, TypeError):
                continue
            if np.isfinite(v) and v < best:
                best, bep = v, int(r["epoch"])
    except OSError:
        return None
    hit = {}
    def visit(name, obj):
        if hasattr(obj, "shape") and tuple(obj.shape) == (len(IA),):
            hit["psi"] = np.array(obj)
    try:
        with h5py.File(os.path.join(d, "best.weights.hdf5"), "r") as f:
            f.visititems(visit)
    except OSError:
        return None
    return (bep, hit["psi"]) if "psi" in hit and bep is not None else None


data = {}
for s in SEEDS:
    p = os.path.join(RUN, f"seed_{s}", "phi_history.npz")
    if not os.path.exists(p):
        print(f"seed {s}: no phi_history yet — skipped"); continue
    z = np.load(p)
    eps, phis = np.asarray(z["epochs"]), np.asarray(z["phi"]).astype(np.float32)
    n = min(len(eps), len(phis))                       # tolerate a torn write
    eps, phis = eps[:n], phis[:n]
    if n == 0:
        print(f"seed {s}: empty phi_history — skipped"); continue
    keep = np.linspace(0, n - 1, min(MAX_FRAMES, n)).astype(int)
    snaps = [(int(eps[k]), phis[k]) for k in keep]
    bc = best_ckpt_frame(s)
    if bc and all(bc[0] != e for e, _ in snaps):
        snaps.append(bc)                       # the one pre-logger true sample
        print(f"  seed {s}: + best.weights frame at epoch {bc[0]}")
    snaps.sort(key=lambda t: t[0])
    frames = []
    for ep_k, phi_k in snaps:
        phi = phi_k.astype(np.float64)
        w = np.exp(phi - phi.max()); pr = w / w.sum()
        vmax = float(pr.max()); peak = int(np.argmax(pr))
        # EVERY pair, log-quantised to one byte against [FLOOR, vmax]; base64 of
        # 5050 bytes is ~6.7 kB per frame, so the whole lattice fits the deck
        # and the widget can render the same texture the static figure shows.
        lo, hi = np.log10(FLOOR), np.log10(max(vmax, FLOOR * 10))
        q = np.clip(np.round(255 * (np.log10(np.maximum(pr, FLOOR)) - lo) / (hi - lo)), 0, 255)
        frames.append({
            "e": int(ep_k),
            "vmax": vmax,
            "pk": peak,
            "b": base64.b64encode(q.astype(np.uint8).tobytes()).decode("ascii"),
        })
    data[str(s)] = {"color": COLOR[s], "frames": frames}
    print(f"seed {s}: {len(frames)} frames (epochs {frames[0]['e']}..{frames[-1]['e']}), "
          f"p_max last {frames[-1]['vmax']*100:.2f}%")

if not data:
    raise SystemExit("no phi_history anywhere yet — nothing to build")

# pair index -> (i, j) lookup, shared by all seeds, packed once
PAIRS = ";".join(f"{a}.{b}" for a, b in zip(IA.tolist(), IB.tolist()))

TEMPLATE = """
<style>
  :root{color-scheme:light}
  html,body{margin:0;padding:0;background:transparent;
    font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:#111318}
  .wrap{display:flex;flex-direction:column;height:100vh;box-sizing:border-box;padding:6px 10px 10px}
  .row{display:flex;gap:14px;flex:1;min-height:0}
  .cell{flex:1;display:flex;flex-direction:column;align-items:center;min-width:0}
  canvas.map{width:100%;height:auto;image-rendering:pixelated;border:1px solid #dfe4e9;
    border-radius:6px;background:#000}
  .cap{font-size:15px;font-weight:700;margin-top:4px;color:#111318}
  .ax{display:flex;justify-content:space-between;width:100%;font-size:11px;color:#5b6470;margin-top:3px}
  .cbar{display:flex;align-items:center;gap:8px;margin-top:8px}
  .cbar canvas{width:280px;height:12px;border:1px solid #dfe4e9;border-radius:3px}
  .cbl{font-size:12px;color:#111318;font-weight:600;font-variant-numeric:tabular-nums}
  .cbnote{font-size:12px;color:#5b6470}
  .ctl{display:flex;align-items:center;gap:12px;margin-top:8px;
    background:rgba(255,255,255,.92);border-radius:8px;padding:6px 10px}
  button{font:inherit;font-size:15px;padding:5px 14px;border-radius:7px;border:1px solid #c8d0d7;
    background:#ffffff;color:#111318;cursor:pointer;-webkit-appearance:none;appearance:none}
  button:hover{background:#f2f5f7;color:#111318}
  input[type=range]{flex:1;accent-color:#7c3aed;height:22px}
  .ep{font-variant-numeric:tabular-nums;font-weight:700;font-size:16px;min-width:150px;
    text-align:right;color:#111318}
  .hint{font-size:12px;color:#5b6470;margin-top:2px}
</style>
<div class="wrap">
  <div class="row" id="row"></div>
  <div class="cbar">
    <span class="cbl">1e-7</span>
    <canvas id="cb" width="280" height="12"></canvas>
    <span class="cbl" id="cbmax">p max</span>
    <span class="cbnote">true pair probability, log scale — each panel scaled to its own peak</span>
  </div>
  <div class="ctl">
    <button id="play">Pause</button>
    <input type="range" id="sl" min="0" max="10" value="0" step="1">
    <span class="ep" id="ep">epoch —</span>
  </div>
  <div class="hint" id="hint">One slider drives all three seeds. Playing automatically — drag it anywhere; playback continues from there.</div>
  <div class="hint" id="range" style="font-weight:600"></div>
</div>
"""

SCRIPT = """
const DATA = __DATA__, PAIRSTR = "__PAIRS__";
const T = 101, FLOOR = 1e-7;
const seeds = Object.keys(DATA);
// pair index -> i,j
const PS = PAIRSTR.split(";");
const PI = new Int16Array(PS.length), PJ = new Int16Array(PS.length);
for (let k = 0; k < PS.length; k++){
  const d = PS[k].indexOf(".");
  PI[k] = +PS[k].slice(0, d); PJ[k] = +PS[k].slice(d + 1);
}
for (const s of seeds) for (const f of DATA[s].frames){
  const bin = atob(f.b);                       // one byte per pair, all 5050
  const a = new Uint8Array(bin.length);
  for (let k = 0; k < bin.length; k++) a[k] = bin.charCodeAt(k);
  f.val = a;
}
const NF = Math.max(...seeds.map(s => DATA[s].frames.length));

const RAMP = [[0,0,4],[28,16,68],[79,18,123],[129,37,129],[181,54,122],
              [229,80,100],[251,135,97],[254,194,135],[252,253,191]];
function magma(t){
  t = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(t)), f = t - i;
  const a = RAMP[i], b = RAMP[i+1];
  return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f];
}

const row = document.getElementById("row");
const cells = {};
for (const s of seeds){
  const d = document.createElement("div"); d.className = "cell";
  const cv = document.createElement("canvas"); cv.className = "map"; cv.width = T; cv.height = T;
  const cap = document.createElement("div"); cap.className = "cap";
  cap.style.color = DATA[s].color; cap.textContent = "seed " + s;
  const ax = document.createElement("div"); ax.className = "ax";
  ax.innerHTML = "<span>t<sub>i</sub> = earlier slice &rarr;</span>" +
                 "<span>&uarr; t<sub>j</sub> = later slice</span>";
  d.appendChild(cv); d.appendChild(ax); d.appendChild(cap); row.appendChild(d);
  const ctx = cv.getContext("2d");
  cells[s] = { ctx, img: ctx.createImageData(T, T), cap };
}

function draw(fi){
  let anyMax = 0;
  for (const s of seeds){
    const D = DATA[s], c = cells[s];
    const f = D.frames[Math.min(fi, D.frames.length - 1)];
    const px = c.img.data;
    const base = magma(0);
    for (let k = 0; k < px.length; k += 4){
      px[k] = base[0]; px[k+1] = base[1]; px[k+2] = base[2]; px[k+3] = 255;
    }
    for (let k = 0; k < f.val.length; k++){       // EVERY pair
      const col = magma(f.val[k] / 255);
      const i = PI[k], j = PJ[k];
      let o = 4*((T-1-j)*T + i);                  // flip y so t_j grows upward
      px[o] = col[0]; px[o+1] = col[1]; px[o+2] = col[2]; px[o+3] = 255;
      o = 4*((T-1-i)*T + j);                      // symmetric half
      px[o] = col[0]; px[o+1] = col[1]; px[o+2] = col[2]; px[o+3] = 255;
    }
    c.ctx.putImageData(c.img, 0, 0);
    // furniture to match the static figure: dotted adjacency diagonal + peak star
    const g = c.ctx;
    g.save();
    g.strokeStyle = "rgba(154,164,173,.75)"; g.lineWidth = 0.6; g.setLineDash([2, 2]);
    g.beginPath(); g.moveTo(0, T); g.lineTo(T, 0); g.stroke();
    g.restore();
    const pi = PI[f.pk], pj = PJ[f.pk];
    g.save();
    g.strokeStyle = "#00e5ff"; g.lineWidth = 1.1;
    g.beginPath(); g.arc(pi + 0.5, (T-1-pj) + 0.5, 3.2, 0, 7); g.stroke();
    g.restore();
    c.cap.textContent = "seed " + s + "  peak (" + pi + ", " + pj + ")  p=" +
                        (f.vmax*100).toFixed(2) + "%";
    anyMax = Math.max(anyMax, f.vmax);
    if (s === seeds[0]) document.getElementById("ep").textContent =
      "epoch " + f.e + "  (" + (fi+1) + "/" + NF + ")";
  }
  document.getElementById("cbmax").textContent = (anyMax*100).toFixed(2) + "%";
}

(function paintCbar(){
  const cb = document.getElementById("cb"), cx = cb.getContext("2d");
  const im = cx.createImageData(cb.width, cb.height);
  for (let x = 0; x < cb.width; x++){
    const col = magma(x / (cb.width - 1));
    for (let y = 0; y < cb.height; y++){
      const o = 4*(y*cb.width + x);
      im.data[o] = col[0]; im.data[o+1] = col[1]; im.data[o+2] = col[2]; im.data[o+3] = 255;
    }
  }
  cx.putImageData(im, 0, 0);
})();

const sl = document.getElementById("sl"), btn = document.getElementById("play");
sl.max = String(NF - 1);
// Be explicit about the covered range: phi snapshots only start where the
// logger was switched on, so this slider does NOT span the whole run.
(function(){
  const es = seeds.flatMap(s => DATA[s].frames.map(f => f.e));
  const lo = Math.min(...es), hi = Math.max(...es);
  document.getElementById("range").textContent =
    "Slider covers epochs " + lo + "–" + hi + " (" + NF + " snapshots, every 25 epochs). " +
    "phi logging began mid-run, so earlier epochs have no true-probability record.";
})();
let clock = 0, scrubbing = false, paused = false;
function tick(){
  if (!paused && !scrubbing && NF > 1){
    clock = (clock + 1) % NF; sl.value = String(clock); draw(clock);
  }
  setTimeout(tick, 260);
}
sl.addEventListener("input", () => { scrubbing = true; clock = +sl.value; draw(clock);
  document.getElementById("hint").textContent = "Scrubbing — playback continues from here when you let go."; });
const release = () => { if (!scrubbing) return; scrubbing = false; clock = +sl.value;
  document.getElementById("hint").textContent =
    "One slider drives all three seeds. Playing automatically — drag it anywhere; playback continues from there."; };
sl.addEventListener("change", release);
sl.addEventListener("pointerup", release);
sl.addEventListener("pointercancel", release);
btn.addEventListener("click", () => { paused = !paused; btn.textContent = paused ? "Play" : "Pause"; });
draw(0);
tick();
"""

payload = {"template": TEMPLATE,
           "script": (SCRIPT.replace("__DATA__", json.dumps(data, separators=(",", ":")))
                            .replace("__PAIRS__", PAIRS))}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(payload, f)
print("wrote", OUT, f"({os.path.getsize(OUT)/1024:.0f} kB)")
