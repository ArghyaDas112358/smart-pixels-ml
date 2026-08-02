"""
ONE long Part-1 run (ViT_Max + SoftQuantize) on the NEW mV CLEAN (no-noise) CONTAINED
2_5 dataset, to watch 2-bit threshold + loss CONVERGENCE over many epochs.

Data: /work/.../shuffled_3d/TFR_files_2_5_clean_contained/{TFR_train,TFR_test}
  - time samples [11,26], NO noise (clean control) -> load with noise=-1
  - contained = original_atEdge==False

Recipe (ported from run_part1_long.py; data scale matches the prior new-3srb set,
max ~644, so the same window/offset apply):
  offset=0, random-init window [25,160], levels [0,1,2,3], cosine k-anneal 1->67
  stretched over the full run, Nadam(1e-3), custom_loss (NLL), no early stopping.

Logs epoch/loss/val_loss/T0/T1/T2 EVERY epoch to threshold_loss_epochs.csv and
keeps only the best checkpoint. AbortOnStuck kills a bad-seed stuck init fast.

Usage:
  python run_part1_long_2_5_noise_contained.py --seed 42 --epochs 5000
  SMOKE: --epochs 2  (quick end-to-end + OOM check)
"""
import os, sys, json, time, random, argparse, csv, traceback, gc, re, subprocess, glob
HELPERS = "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers"
sys.path.insert(0, HELPERS)
import numpy as np, tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(g, True)
    except Exception: pass
from prepare_tfrecords import load_tfrecords
from train import create_model
from AnnealingScheduler import AnnealingScheduler
from loss import custom_loss

BASE = "/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d"
TFR = os.path.join(BASE, "TFR_files_2_5_clean_contained")
TFR_TRAIN = os.path.join(TFR, "TFR_train")
TFR_TEST  = os.path.join(TFR, "TFR_test")

OFFSET, LOW, HIGH, TS = 0.0, 25.0, 160.0, [11, 26]
LEVELS = np.array([0., 1., 2., 3.], dtype=np.float32)


class AbortOnStuck(tf.keras.callbacks.Callback):
    """Abort a stuck seed. Stuck = val_loss still ABOVE `thr` AND not improving (by
    > min_delta) for `patience` consecutive epochs. The no-improvement test is what
    makes thr=1e4 safe: a HEALTHY run starts ~98k and descends through 10k-98k while
    improving every epoch, so it never trips; a stuck run is flat at ~98,886 (a single
    repeated value) and trips within `patience`. Once val_loss drops below thr the
    counter resets, so a converged run plateauing low is never aborted."""
    def __init__(s, thr=1e4, patience=15, min_delta=1.0):
        super().__init__(); s.thr=thr; s.pat=patience; s.min_delta=min_delta
        s.best=np.inf; s.bad=0; s.aborted=False
    def on_epoch_end(s, e, logs=None):
        v=(logs or {}).get("val_loss", np.inf)
        if not np.isfinite(v):
            s.bad+=1
        elif v < s.best - s.min_delta:    # genuine improvement -> healthy, reset
            s.best=v; s.bad=0
        elif v > s.thr:                    # not improving AND stuck above floor
            s.bad+=1
        else:                               # below floor (converging) -> never abort
            s.bad=0
        if s.bad>=s.pat:
            print(f"[AbortOnStuck] val_loss {v:.1f} stuck (> {s.thr:g}, no improve {s.pat} ep) -- aborting seed.")
            s.aborted=True; s.model.stop_training=True


class ThresholdLogger(tf.keras.callbacks.Callback):
    """Append epoch / loss / val_loss / T0 / T1 / T2 every epoch."""
    def __init__(s, path, layer='soft_quantizer_output'):
        super().__init__(); s.path=path; s.lname=layer
        with open(s.path,'w',newline='') as f:
            csv.writer(f).writerow(['epoch','loss','val_loss','T0','T1','T2'])
    def on_epoch_end(s, epoch, logs=None):
        logs=logs or {}
        thr=[float(t) for t in np.array(s.model.get_layer(s.lname).thresholds).ravel()]
        with open(s.path,'a',newline='') as f:
            csv.writer(f).writerow([epoch, logs.get('loss',''), logs.get('val_loss',''), thr[0], thr[1], thr[2]])


ESCAPE_BELOW = 5e4   # a seed that reaches best val_loss below this genuinely learned (escaped the ceiling)


def run_one_seed(seed, epochs, out_root, tg, vg, stamp):
    """Train one seed. Returns (escaped: bool, info: dict). A stuck seed aborts within
    ~AbortOnStuck.patience epochs; an escaping seed runs to `epochs`."""
    OUT=os.path.join(out_root, f'seed_{seed}'); os.makedirs(OUT, exist_ok=True)
    tf.keras.backend.clear_session(); gc.collect()   # free GPU graph from the previous seed
    thr0=sorted(np.random.default_rng(seed).uniform(LOW,HIGH,3).tolist())
    stamp(f"[seed {seed}] init={[round(t,2) for t in thr0]}  -> fitting (max {epochs} ep)")
    tf.random.set_seed(seed); np.random.seed(seed); random.seed(seed)
    model=create_model('ViT_Max', timeslices=2, soft_quantize_layer=True, initial_thresholds=thr0,
                       threshold_offset=OFFSET, initial_levels=LEVELS)
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3), loss=custom_loss)
    csv_path=os.path.join(OUT,'threshold_loss_epochs.csv')
    stuck_cb=AbortOnStuck(thr=1e4, patience=15)
    cbs=[ThresholdLogger(csv_path),
         tf.keras.callbacks.CSVLogger(os.path.join(OUT,'history.csv')),
         AnnealingScheduler(schedule='cosine', target_layer_name='soft_quantizer_output',
                            initial_k=1.0, final_k=67.0, verbose=0),
         tf.keras.callbacks.ModelCheckpoint(os.path.join(OUT,'best.weights.hdf5'),
                                            save_weights_only=True, monitor='val_loss', save_best_only=True),
         stuck_cb]
    t0=time.time()
    h=model.fit(tg, validation_data=vg, epochs=epochs, callbacks=cbs, shuffle=False, verbose=2)
    vl=h.history.get('val_loss',[np.inf]); best=float(min(vl)); fin=float(vl[-1]); best_ep=int(np.argmin(vl))
    escaped=(not stuck_cb.aborted) and (best < ESCAPE_BELOW)
    last=list(csv.reader(open(csv_path)))[-1]
    best_thr=[float(t) for t in np.array(model.get_layer('soft_quantizer_output').thresholds).ravel()]
    info={'seed':seed,'epochs':len(vl),'max_epochs':epochs,'best_epoch':best_ep,'escaped':escaped,
          'aborted_stuck':stuck_cb.aborted,'offset':OFFSET,'init_thresholds':thr0,
          'final_thresholds':[float(last[3]),float(last[4]),float(last[5])],'best_thresholds':best_thr,
          'best_val_loss':best,'final_val_loss':fin,'wall_sec':round(time.time()-t0),'csv':csv_path,'data':TFR}
    json.dump(info, open(os.path.join(OUT,'result.json'),'w'),indent=1)
    del model; gc.collect()
    if not escaped:   # mark the dead dir so it's obvious
        try: os.rename(OUT, OUT+'_STUCK')
        except OSError: pass
    return escaped, info


def scan_existing(out_root):
    """Classify existing seed dirs. Returns (converged:set, attempted:set)."""
    conv=set(); attempted=set()
    for d in glob.glob(os.path.join(out_root,'seed_*')):
        base=os.path.basename(d)
        m=re.match(r'seed_(\d+)$', base)          # a clean run dir
        if m:
            s=int(m.group(1)); attempted.add(s)
            rj=os.path.join(d,'result.json')
            if os.path.exists(rj):
                try:
                    if json.load(open(rj)).get('escaped'): conv.add(s)
                except Exception: pass
            continue
        m2=re.match(r'seed_(\d+)_', base)          # seed_N_STUCK / seed_N_STUCK_DEAD
        if m2: attempted.add(int(m2.group(1)))
    return conv, attempted


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=5000)
    ap.add_argument('--target', type=int, default=5, help='total CONVERGED seeds to collect (incl. existing)')
    ap.add_argument('--out', default='/work/users/das214/SmartPixels/smart-pixels-ml/runs/part1_long_2_5_clean_contained')
    ap.add_argument('--seeds', default=None, help='comma-sep candidate seeds; default = 40 spaced seeds')
    a=ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log=open(os.path.join(a.out,'reseed.log'),'a',buffering=1)
    stamp=lambda m: log.write(f"[{time.strftime('%H:%M:%S')}] {m}\n") or log.flush() or print(m, flush=True)
    candidates=[int(s) for s in a.seeds.split(',')] if a.seeds else [42+1000*i for i in range(40)]
    report_py=os.path.join(os.path.dirname(os.path.abspath(__file__)),'make_2_5_clean_study.py')

    def regen_report():
        try:
            subprocess.run([sys.executable, report_py], timeout=900,
                           env={**os.environ,'CUDA_VISIBLE_DEVICES':''},
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            stamp("  report regenerated (REPORT.md + figs)")
        except Exception as e:
            stamp(f"  report regen failed: {e}")

    try:
        gpus=tf.config.list_physical_devices("GPU")
        conv, attempted = scan_existing(a.out)
        stamp(f"COLLECT target={a.target} converged seeds: epochs={a.epochs} offset={OFFSET} "
              f"window[{LOW},{HIGH}] GPUs={len(gpus)} AbortOnStuck(1e4,15,flatline) ESCAPE_BELOW={ESCAPE_BELOW:g}")
        stamp(f"resume: {len(conv)} converged already ({sorted(conv)}); "
              f"{len(attempted-conv)} stuck/attempted; need {max(0,a.target-len(conv))} more")
        stamp(f"data: {TFR_TRAIN}  (noise=-1, already baked)")
        if len(conv) >= a.target:
            stamp("target already met -- regen report and exit"); regen_report(); return
        tg,vg=load_tfrecords(TFR_TRAIN, TFR_TEST, noise=-1, seed=42)
        for seed in candidates:
            if len(conv) >= a.target: break
            if seed in attempted:        # already converged or already tried (stuck)
                continue
            attempted.add(seed)
            escaped, info = run_one_seed(seed, a.epochs, a.out, tg, vg, stamp)
            if escaped:
                conv.add(seed)
                stamp(f"[CONVERGED {len(conv)}/{a.target}] seed={seed}: best_val={info['best_val_loss']:.1f}"
                      f"@{info['best_epoch']} best_thr={[round(x,2) for x in info['best_thresholds']]} "
                      f"({info['wall_sec']}s)")
                json.dump({'target':a.target,'converged_seeds':sorted(conv),'updated':time.strftime('%Y-%m-%d %H:%M:%S')},
                          open(os.path.join(a.out,'WINNERS.json'),'w'),indent=1)
                regen_report()
            else:
                stamp(f"[stuck] seed={seed} aborted={info['aborted_stuck']} best_val={info['best_val_loss']:.1f} "
                      f"(epochs={info['epochs']}) -- next seed")
        if len(conv) >= a.target:
            stamp(f"DONE: {len(conv)} converged seeds collected: {sorted(conv)}")
        else:
            stamp(f"EXHAUSTED candidates with {len(conv)}/{a.target} converged -- pass more --seeds")
        regen_report()
    except Exception:
        traceback.print_exc()
        open(os.path.join(a.out,'FAILED.txt'),'w').write(traceback.format_exc()); sys.exit(1)


main()
