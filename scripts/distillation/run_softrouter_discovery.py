"""
SoftRouter JOINT slice+threshold DISCOVERY driver (docs/soft_router_plan.md §4 step 3).

Model:   ViT_Max_SoftRouter = Input(16,16,101) -> SoftRouterLayer -> SoftQuantizeLayer
         -> pre-LN ViT backbone (14 outputs, full-covariance NLL).
Anneal:  TWO AnnealingSchedulers, same cosine k:1->67 over the full run
         (router 'soft_router_output' + quantizer 'soft_quantizer_output').
Init:    thresholds random in [25,160] per seed (NOT the prior optimum — joint
         re-discovery); router logits ~N(0,0.5) under the seed.
Data:    TFR_files_all101_noise_contained_discovery (20+5 files, 38k+9.6k events,
         noise baked -> load with noise=-1, labels_scale pinned).
Collect: like the threshold study — run seeds until --target CONVERGED seeds;
         stuck seeds auto-abort (AbortOnStuck thr=1e4 + 15-epoch flat-line) and
         are skipped. Resumable: re-run continues from existing seed dirs.
Logs:    per epoch CSV: loss, val_loss, k, [i1,i2], slot commitment, T0/T1/T2;
         slot-weight snapshots every SNAP epochs -> slot_weights.npz
         (feeds the real version of the plan's convergence heatmap).

Usage:
  python run_softrouter_discovery.py --sanity          # 3-epoch GPU/memory check
  python run_softrouter_discovery.py --epochs 5000 --target 4
"""
import os, sys, json, time, glob, random, argparse, csv, re, traceback

HELPERS = "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers"
sys.path.insert(0, HELPERS)

import numpy as np
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(g, True)
    except Exception: pass

from prepare_tfrecords import load_tfrecords
from train import create_model
from AnnealingScheduler import AnnealingScheduler
from loss import custom_loss

BASE = "/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d"
TFR = os.path.join(BASE, "TFR_files_all101_noise_contained_discovery")
TFR_TRAIN, TFR_TEST = os.path.join(TFR, "TFR_train"), os.path.join(TFR, "TFR_test")

N_SLICES = 101
THR_LOW, THR_HIGH = 25.0, 160.0          # random threshold window (threshold-study recipe)
LEVELS = np.array([0., 1., 2., 3.], dtype=np.float32)
ESCAPE_BELOW = 5e4                        # best val_loss below this = genuinely learned
SNAP = 25                                 # slot-weight snapshot cadence (epochs)


class AbortOnStuck(tf.keras.callbacks.Callback):
    """Stuck = val_loss above `thr` AND not improving for `patience` epochs.
    (Same guard that went 5-for-5 on the threshold study; see that file for why
    thr=1e4 is safe: a healthy run descends while improving, so it never trips.)"""
    def __init__(s, thr=1e4, patience=15, min_delta=1.0):
        super().__init__(); s.thr=thr; s.pat=patience; s.min_delta=min_delta
        s.best=np.inf; s.bad=0; s.aborted=False
    def on_epoch_end(s, e, logs=None):
        v=(logs or {}).get("val_loss", np.inf)
        if not np.isfinite(v): s.bad+=1
        elif v < s.best - s.min_delta: s.best=v; s.bad=0
        elif v > s.thr: s.bad+=1
        else: s.bad=0
        if s.bad>=s.pat:
            print(f"[AbortOnStuck] val_loss {v:.1f} stuck -- aborting seed.")
            s.aborted=True; s.model.stop_training=True


class RouterLogger(tf.keras.callbacks.Callback):
    """Per-epoch CSV of the joint state + periodic slot-weight snapshots."""
    def __init__(s, csv_path, npz_path, snap=SNAP):
        super().__init__(); s.csv_path=csv_path; s.npz_path=npz_path; s.snap=snap
        s.snap_epochs=[]; s.snap_weights=[]
        with open(s.csv_path,'w',newline='') as f:
            csv.writer(f).writerow(
                ['epoch','loss','val_loss','k_router','k_quant',
                 'i1','i2','w1_max','w2_max','T0','T1','T2'])
    def on_epoch_end(s, epoch, logs=None):
        logs=logs or {}
        r=s.model.get_layer('soft_router_output'); q=s.model.get_layer('soft_quantizer_output')
        W=r.slot_weights_numpy()                       # (2, 101) masked softmax weights
        idx=r.selected_indices()
        thr=[float(t) for t in np.array(q.thresholds).ravel()]
        with open(s.csv_path,'a',newline='') as f:
            csv.writer(f).writerow(
                [epoch, logs.get('loss',''), logs.get('val_loss',''),
                 float(np.exp(r.log_k.numpy()[0])), float(np.exp(q.log_k.numpy()[0])),
                 idx[0], idx[1], float(W[0].max()), float(W[1].max()),
                 thr[0], thr[1], thr[2]])
        if epoch % s.snap == 0:
            s.snap_epochs.append(epoch); s.snap_weights.append(W.astype(np.float32))
    def on_train_end(s, logs=None):
        if s.snap_epochs:
            np.savez_compressed(s.npz_path,
                epochs=np.array(s.snap_epochs),
                weights=np.stack(s.snap_weights))       # (n_snap, 2, 101)


def run_one_seed(seed, epochs, out_root, tg, vg, stamp):
    OUT=os.path.join(out_root, f'seed_{seed}'); os.makedirs(OUT, exist_ok=True)
    tf.keras.backend.clear_session()
    thr0=sorted(np.random.default_rng(seed).uniform(THR_LOW, THR_HIGH, 3).tolist())
    tf.random.set_seed(seed); np.random.seed(seed); random.seed(seed)
    stamp(f"[seed {seed}] thr0={[round(t,1) for t in thr0]}  -> fitting (max {epochs} ep)")

    model=create_model('ViT_Max_SoftRouter', timeslices=N_SLICES, soft_quantize_layer=True,
                       initial_thresholds=thr0, threshold_offset=0.0, initial_levels=LEVELS)
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3), loss=custom_loss)

    stuck=AbortOnStuck()
    cbs=[RouterLogger(os.path.join(OUT,'router_epochs.csv'), os.path.join(OUT,'slot_weights.npz')),
         tf.keras.callbacks.CSVLogger(os.path.join(OUT,'history.csv')),
         AnnealingScheduler('cosine', target_layer_name='soft_router_output',
                            initial_k=1.0, final_k=67.0, verbose=0),
         AnnealingScheduler('cosine', target_layer_name='soft_quantizer_output',
                            initial_k=1.0, final_k=67.0, verbose=0),
         tf.keras.callbacks.ModelCheckpoint(os.path.join(OUT,'best.weights.hdf5'),
                                            save_weights_only=True, monitor='val_loss',
                                            save_best_only=True),
         stuck]
    t0=time.time()
    h=model.fit(tg, validation_data=vg, epochs=epochs, callbacks=cbs, shuffle=False, verbose=2)
    vl=h.history.get('val_loss',[np.inf]); best=float(min(vl)); best_ep=int(np.argmin(vl))
    escaped=(not stuck.aborted) and (best < ESCAPE_BELOW)

    router=model.get_layer('soft_router_output'); quant=model.get_layer('soft_quantizer_output')
    info={'seed':seed,'epochs':len(vl),'max_epochs':epochs,'best_epoch':best_ep,
          'escaped':escaped,'aborted_stuck':stuck.aborted,
          'init_thresholds':thr0,
          'final_indices':router.selected_indices(),
          'final_slot_commitment':[float(v) for v in router.slot_weights_numpy().max(axis=1)],
          'final_thresholds':[float(t) for t in np.array(quant.thresholds).ravel()],
          'best_val_loss':best,'final_val_loss':float(vl[-1]),
          'wall_sec':round(time.time()-t0),'data':TFR}
    json.dump(info, open(os.path.join(OUT,'result.json'),'w'), indent=1)
    del model
    if not escaped:
        try: os.rename(OUT, OUT+'_STUCK')
        except OSError: pass
    return escaped, info


def scan_existing(out_root):
    conv=set(); attempted=set()
    for d in glob.glob(os.path.join(out_root,'seed_*')):
        base=os.path.basename(d)
        m=re.match(r'seed_(\d+)$', base)
        if m:
            s=int(m.group(1)); attempted.add(s)
            rj=os.path.join(d,'result.json')
            if os.path.exists(rj):
                try:
                    if json.load(open(rj)).get('escaped'): conv.add(s)
                except Exception: pass
            continue
        m2=re.match(r'seed_(\d+)_', base)
        if m2: attempted.add(int(m2.group(1)))
    return conv, attempted


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=5000)
    ap.add_argument('--target', type=int, default=4, help='converged seeds to collect')
    ap.add_argument('--out', default='/work/users/das214/SmartPixels/smart-pixels-ml/runs/softrouter_discovery')
    ap.add_argument('--seeds', default=None, help='comma-sep candidate seeds')
    ap.add_argument('--sanity', action='store_true',
                    help='3-epoch single-seed GPU/memory/timing check, then exit')
    a=ap.parse_args()
    if a.sanity:
        a.epochs=3; a.target=1; a.out=os.path.join(a.out,'sanity')
    os.makedirs(a.out, exist_ok=True)
    log=open(os.path.join(a.out,'discovery.log'),'a',buffering=1)
    stamp=lambda m: log.write(f"[{time.strftime('%H:%M:%S')}] {m}\n") or log.flush() or print(m, flush=True)
    seeds=[int(s) for s in a.seeds.split(',')] if a.seeds else [42+1000*i for i in range(40)]
    try:
        gpus=tf.config.list_physical_devices("GPU")
        conv, attempted = scan_existing(a.out)
        stamp(f"DISCOVERY collect target={a.target}: epochs={a.epochs} thr window[{THR_LOW},{THR_HIGH}] "
              f"GPUs={len(gpus)} JOINT anneal x2 cosine 1->67  ESCAPE_BELOW={ESCAPE_BELOW:g}")
        stamp(f"resume: {len(conv)} converged ({sorted(conv)}); {len(attempted-conv)} stuck/attempted")
        stamp(f"data: {TFR_TRAIN} (noise baked -> noise=-1)")
        tg,vg=load_tfrecords(TFR_TRAIN, TFR_TEST, noise=-1, seed=42)
        for seed in seeds:
            if len(conv) >= a.target: break
            if seed in attempted: continue
            attempted.add(seed)
            escaped, info = run_one_seed(seed, a.epochs, a.out, tg, vg, stamp)
            if a.sanity:
                try:
                    mem=tf.config.experimental.get_memory_info('GPU:0')
                    stamp(f"SANITY GPU peak memory: {mem['peak']/2**30:.2f} GiB")
                except Exception as e:
                    stamp(f"SANITY GPU memory query unavailable: {e}")
                stamp(f"SANITY done: {info['wall_sec']}s for {info['epochs']} epochs "
                      f"(~{info['wall_sec']/max(info['epochs'],1):.1f}s/ep) "
                      f"indices={info['final_indices']} thr={[round(t,1) for t in info['final_thresholds']]}")
                return
            if escaped:
                conv.add(seed)
                stamp(f"[CONVERGED {len(conv)}/{a.target}] seed={seed} "
                      f"indices={info['final_indices']} thr={[round(t,2) for t in info['final_thresholds']]} "
                      f"best_val={info['best_val_loss']:.1f}@{info['best_epoch']} ({info['wall_sec']}s)")
                json.dump({'target':a.target,'converged_seeds':sorted(conv)},
                          open(os.path.join(a.out,'WINNERS.json'),'w'), indent=1)
            else:
                stamp(f"[stuck] seed={seed} best_val={info['best_val_loss']:.1f} "
                      f"(epochs={info['epochs']}) -- next seed")
        if len(conv) >= a.target:
            stamp(f"DONE: {len(conv)} converged seeds: {sorted(conv)}")
        else:
            stamp(f"EXHAUSTED candidates with {len(conv)}/{a.target} converged")
    except Exception:
        traceback.print_exc()
        open(os.path.join(a.out,'FAILED.txt'),'w').write(traceback.format_exc()); sys.exit(1)


main()
