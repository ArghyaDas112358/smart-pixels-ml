"""
One LONG Part-1 run for the late-epoch check: ViT_Max + SoftQuantize, offset=0,
1000 epochs, NO early stopping, k-anneal stretched over the full 1000 epochs.
Logs epoch / loss / val_loss / T0 / T1 / T2 EVERY epoch to a CSV (to spot any
late-epoch monkey business), and keeps only the best checkpoint (disk-safe for
20 parallel SLURM jobs). Seed from --seed (or SLURM_ARRAY_TASK_ID).

Usage: python run_part1_long.py --seed 42 --epochs 1000 --out runs/part1_long
"""
import os, sys, json, time, random, argparse, csv, traceback
HELPERS = "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers"
sys.path.insert(0, HELPERS)
import numpy as np, tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(g, True)
    except Exception: pass
from prepare_tfrecords import generate_tfrecords, load_tfrecords
from train import create_model
from AnnealingScheduler import AnnealingScheduler
from loss import custom_loss

NEW = '/depot/cms/users/das214/datasets/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d'
OFFSET, LOW, HIGH, TS = 0.0, 25.0, 160.0, [11, 26]
LEVELS = np.array([0., 1., 2., 3.], dtype=np.float32)


class AbortOnStuck(tf.keras.callbacks.Callback):
    def __init__(s, thr=1e5, pat=25): super().__init__(); s.thr=thr; s.pat=pat; s.bad=0
    def on_epoch_end(s, e, logs=None):
        v=(logs or {}).get("val_loss", np.inf)
        if v>s.thr or not np.isfinite(v):
            s.bad+=1
            if s.bad>=s.pat: print("[AbortOnStuck] aborting"); s.model.stop_training=True
        else: s.bad=0


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


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--out', default='/work/users/das214/SmartPixels/smart-pixels-ml/runs/part1_long')
    ap.add_argument('--init-thresholds', default=None,
                    help='comma-separated FIXED init e.g. "40,40.1,40.2"; if unset, sample random from --seed')
    ap.add_argument('--patience', type=int, default=0,
                    help='EarlyStopping patience on val_loss (restore_best_weights). 0 = no early stopping.')
    a=ap.parse_args()
    OUT=os.path.join(a.out, f'seed_{a.seed}'); os.makedirs(OUT, exist_ok=True)
    log=open(os.path.join(OUT,'run.log'),'a',buffering=1)
    stamp=lambda m: log.write(f"[{time.strftime('%H:%M:%S')}] {m}\n") or log.flush()
    try:
        if a.init_thresholds:
            thr0=[float(x) for x in a.init_thresholds.split(',')]
            init_kind='fixed'
        else:
            thr0=sorted(np.random.default_rng(a.seed).uniform(LOW,HIGH,3).tolist())
            init_kind='random'
        stamp(f"LONG run seed={a.seed} epochs={a.epochs} offset={OFFSET} init={[round(t,2) for t in thr0]} ({init_kind}) (anneal stretched over {a.epochs})")
        _,_,tr,va=generate_tfrecords(dataset_dir=NEW, model_type='ViT_Max', train_batch_size=5000,
            val_batch_size=5000, select_contained=False, timeslices=2, tfrecords_exist=True, seed=42,
            time_stamps_override=TS)
        tg,vg=load_tfrecords(tr,va,noise=-1,seed=42)
        tf.random.set_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
        model=create_model('ViT_Max', timeslices=2, soft_quantize_layer=True, initial_thresholds=thr0,
                           threshold_offset=OFFSET, initial_levels=LEVELS)
        model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3), loss=custom_loss)
        csv_path=os.path.join(OUT,'threshold_loss_epochs.csv')
        cbs=[ThresholdLogger(csv_path),
             tf.keras.callbacks.CSVLogger(os.path.join(OUT,'history.csv')),
             AnnealingScheduler(schedule='cosine', target_layer_name='soft_quantizer_output',
                                initial_k=1.0, final_k=67.0, verbose=0),
             tf.keras.callbacks.ModelCheckpoint(os.path.join(OUT,'best.weights.hdf5'),
                                                save_weights_only=True, monitor='val_loss', save_best_only=True),
             AbortOnStuck()]
        if a.patience>0:
            cbs.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=a.patience,
                                                        restore_best_weights=True, verbose=1))
        stamp(f"fit: epochs={a.epochs} early_stop_patience={a.patience if a.patience>0 else 'none'}")
        h=model.fit(tg, validation_data=vg, epochs=a.epochs, callbacks=cbs, shuffle=False, verbose=2)
        vl=h.history.get('val_loss',[np.inf]); best=float(min(vl)); fin=float(vl[-1])
        best_ep=int(np.argmin(vl))
        # thresholds at the final logged epoch (CSV last row)
        last=list(csv.reader(open(csv_path)))[-1]
        # restored-best thresholds (restore_best_weights leaves the model on its best epoch)
        best_thr=[float(t) for t in np.array(model.get_layer('soft_quantizer_output').thresholds).ravel()]
        json.dump({'seed':a.seed,'epochs':len(vl),'max_epochs':a.epochs,'patience':a.patience,
                   'stopped_early':len(vl)<a.epochs,'best_epoch':best_ep,
                   'offset':OFFSET,'init_kind':init_kind,'init_thresholds':thr0,
                   'final_thresholds':[float(last[3]),float(last[4]),float(last[5])],
                   'best_thresholds':best_thr,
                   'best_val_loss':best,'final_val_loss':fin,'csv':csv_path},
                  open(os.path.join(OUT,'result.json'),'w'),indent=1)
        stamp(f"DONE epochs={len(vl)}/{a.epochs} stopped_early={len(vl)<a.epochs} best_val={best:.0f}@{best_ep} best_thr={[round(x,2) for x in best_thr]}")
    except Exception:
        traceback.print_exc()
        open(os.path.join(OUT,'FAILED.txt'),'w').write(traceback.format_exc()); sys.exit(1)


main()
