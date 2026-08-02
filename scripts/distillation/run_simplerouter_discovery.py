"""
SIMPLE (Option D, Ahmed et al. ICLR 2023) JOINT slice+threshold DISCOVERY driver
-- the exact-gradient sibling of run_softrouter_discovery.py.

Model:   ViT_Max_SimpleRouter = Input(16,16,101) -> SimpleRouterLayer
         -> SoftQuantizeLayer -> pre-LN ViT backbone (14 outputs, full-cov NLL).
Anneal:  ONE AnnealingScheduler only (quantizer 'soft_quantizer_output',
         cosine k:1->67). The SIMPLE router needs NO annealer: it samples one
         slice pair per step from p({a,b}) ~ exp(theta_a+theta_b) and
         backpropagates the exact Cov(z)@dz gradient into theta.
Init:    thresholds random in [25,160] per seed (NOT the prior optimum -- joint
         re-discovery); theta zeros (uniform over all C(101,2) pairs).
Data:    TFR_files_all101_noise_contained_discovery (noise baked -> load with
         noise=-1, labels_scale pinned) -- same data as the SoftRouter study,
         so the A-vs-D comparison is apples-to-apples.
Collect: run seeds until --target CONVERGED seeds; stuck seeds auto-abort
         (AbortOnStuck thr=1e4 + 15-epoch flat-line) and are skipped.
         Resumable: re-run continues from existing seed dirs.
Logs:    per epoch CSV: [i1,i2] + theta/mu/visits top-5 + mu entropy;
         theta/mu/visits snapshots every SNAP epochs -> theta_mu.npz.
Report:  make_simplerouter_study.py auto-regenerated per converged seed.

Usage:
  python run_simplerouter_discovery.py --sanity          # 3-epoch GPU/memory check
  python run_simplerouter_discovery.py --epochs 1000 --target 6
"""
import os, sys, json, time, glob, random, argparse, csv, re, traceback, subprocess

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
SNAP = 25                                 # theta/mu/visits snapshot cadence (epochs)


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


class SimpleRouterLogger(tf.keras.callbacks.Callback):
    """Per-epoch CSV of the SIMPLE router state + periodic theta/mu/visits snapshots."""
    def __init__(s, csv_path, npz_path, snap=SNAP):
        super().__init__(); s.csv_path=csv_path; s.npz_path=npz_path; s.snap=snap
        s.snap_epochs=[]; s.snap_theta=[]; s.snap_mu=[]; s.snap_visits=[]
        with open(s.csv_path,'w',newline='') as f:
            csv.writer(f).writerow(
                ['epoch','i1','i2','theta_top5_idx','theta_top5_val',
                 'mu_top5_idx','mu_top5_val','mu_entropy','visits_top5_idx'])
    def on_epoch_end(s, epoch, logs=None):
        r=s.model.get_layer('simple_router_output')
        th=r.theta.numpy().astype(np.float64)
        mu=r.mu_numpy().astype(np.float64)                 # exact marginals, sum=2
        vis=r.visits.numpy().astype(np.float64)
        i1,i2=r.selected_indices()
        tidx=np.argsort(th)[::-1][:5]
        midx=np.argsort(mu)[::-1][:5]
        vidx=np.argsort(vis)[::-1][:5]
        q=mu/mu.sum()                                       # normalized importance
        ent=float(-(q*np.log(q+1e-12)).sum())               # ln(101)=4.615 at uniform
        j=lambda v: ';'.join(str(x) for x in v)
        with open(s.csv_path,'a',newline='') as f:
            csv.writer(f).writerow(
                [epoch, i1, i2,
                 j(int(k) for k in tidx), j(f'{th[k]:.6g}' for k in tidx),
                 j(int(k) for k in midx), j(f'{mu[k]:.6g}' for k in midx),
                 f'{ent:.6g}', j(int(k) for k in vidx)])
        if epoch % s.snap == 0:
            s.snap_epochs.append(epoch)
            s.snap_theta.append(th.astype(np.float32))
            s.snap_mu.append(mu.astype(np.float32))
            s.snap_visits.append(vis.astype(np.float32))
    def on_train_end(s, logs=None):
        if s.snap_epochs:
            np.savez_compressed(s.npz_path,
                epochs=np.array(s.snap_epochs),             # (S,)
                theta=np.stack(s.snap_theta),               # (S, 101)
                mu=np.stack(s.snap_mu),                     # (S, 101)
                visits=np.stack(s.snap_visits))             # (S, 101)


def run_one_seed(seed, epochs, out_root, tg, vg, stamp):
    OUT=os.path.join(out_root, f'seed_{seed}'); os.makedirs(OUT, exist_ok=True)
    tf.keras.backend.clear_session()
    thr0=sorted(np.random.default_rng(seed).uniform(THR_LOW, THR_HIGH, 3).tolist())
    tf.random.set_seed(seed); np.random.seed(seed); random.seed(seed)
    stamp(f"[seed {seed}] thr0={[round(t,1) for t in thr0]}  -> fitting (max {epochs} ep)")

    model=create_model('ViT_Max_SimpleRouter', timeslices=N_SLICES, soft_quantize_layer=True,
                       initial_thresholds=thr0, threshold_offset=0.0, initial_levels=LEVELS)
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3), loss=custom_loss)

    stuck=AbortOnStuck()
    cbs=[SimpleRouterLogger(os.path.join(OUT,'router_epochs.csv'), os.path.join(OUT,'theta_mu.npz')),
         tf.keras.callbacks.CSVLogger(os.path.join(OUT,'history.csv')),
         # ONE annealer: the quantizer only. The SIMPLE router has no k.
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

    router=model.get_layer('simple_router_output'); quant=model.get_layer('soft_quantizer_output')
    mu=router.mu_numpy().astype(np.float64)
    idx=router.selected_indices()                        # ascending top-2 of theta
    mu_top5=[[int(k), float(mu[k])] for k in np.argsort(mu)[::-1][:5]]
    info={'seed':seed,'epochs':len(vl),'max_epochs':epochs,'best_epoch':best_ep,
          'escaped':escaped,'aborted_stuck':stuck.aborted,
          'init_thresholds':thr0,
          'final_indices':idx,
          'final_mu_top5':mu_top5,
          # commitment analog of the SoftRouter field: mu of the two picked slices
          'final_slot_commitment':[float(mu[idx[0]]), float(mu[idx[1]])],
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
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--target', type=int, default=6, help='converged seeds to collect')
    ap.add_argument('--out', default='/work/users/das214/SmartPixels/smart-pixels-ml/runs/simplerouter_discovery')
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
    report_py=os.path.join(os.path.dirname(os.path.abspath(__file__)),'make_simplerouter_study.py')

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
        stamp(f"SIMPLE DISCOVERY collect target={a.target}: epochs={a.epochs} thr window[{THR_LOW},{THR_HIGH}] "
              f"GPUs={len(gpus)} quantizer-only anneal cosine 1->67 (router: exact SIMPLE gradient) "
              f"ESCAPE_BELOW={ESCAPE_BELOW:g}")
        stamp(f"resume: {len(conv)} converged ({sorted(conv)}); {len(attempted-conv)} stuck/attempted")
        stamp(f"data: {TFR_TRAIN} (noise baked -> noise=-1)")
        if not a.sanity and len(conv) >= a.target:
            stamp("target already met -- regen report and exit"); regen_report(); return
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
                regen_report()
            else:
                stamp(f"[stuck] seed={seed} best_val={info['best_val_loss']:.1f} "
                      f"(epochs={info['epochs']}) -- next seed")
        if len(conv) >= a.target:
            stamp(f"DONE: {len(conv)} converged seeds: {sorted(conv)}")
        else:
            stamp(f"EXHAUSTED candidates with {len(conv)}/{a.target} converged")
        regen_report()
    except Exception:
        traceback.print_exc()
        open(os.path.join(a.out,'FAILED.txt'),'w').write(traceback.format_exc()); sys.exit(1)


main()
