"""
Re-run the BEST offset=10 Part-1 model (seed 7042) to regenerate its checkpoints
(deleted when we switched to offset=0), for the offset=10 vs offset=0 performance
comparison. Same config as the offset=10 collection: window [25,160], offset=10,
NO noise, 300 epochs, samples [11,26].
"""
import os, sys, json, time, random, traceback
HELPERS = "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers"
sys.path.insert(0, HELPERS)
import numpy as np, tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(g, True)
    except Exception: pass
from prepare_tfrecords import generate_tfrecords, load_tfrecords
from train import create_model, train, get_best_thresholds

NEW = '/depot/cms/users/das214/datasets/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d'
OUT = '/work/users/das214/SmartPixels/smart-pixels-ml/runs/new_dataset_part1_off10best'
SEED = 7042; OFFSET = 10.0; LOW, HIGH = 25.0, 160.0; EPOCHS = 300; TS = [11, 26]
LEVELS = np.array([0., 1., 2., 3.], dtype=np.float32)
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT, 'train.log'), 'a', buffering=1)
stamp = lambda m: log.write(f"[{time.strftime('%H:%M:%S')}] {m}\n") or log.flush()

class AbortOnStuck(tf.keras.callbacks.Callback):
    def __init__(s, thr=1e5, pat=20): super().__init__(); s.thr=thr; s.pat=pat; s.bad=0
    def on_epoch_end(s, e, logs=None):
        v=(logs or {}).get("val_loss", np.inf)
        if v>s.thr or not np.isfinite(v):
            s.bad+=1
            if s.bad>=s.pat: print("[AbortOnStuck] aborting"); s.model.stop_training=True
        else: s.bad=0

try:
    thr0 = sorted(np.random.default_rng(SEED).uniform(LOW, HIGH, 3).tolist())
    stamp(f"offset=10 best re-run: seed={SEED} offset={OFFSET} init_thr={[round(t,1) for t in thr0]}")
    _,_,tr,va = generate_tfrecords(dataset_dir=NEW, model_type='ViT_Max', train_batch_size=5000,
        val_batch_size=5000, select_contained=False, timeslices=2, tfrecords_exist=True, seed=42,
        time_stamps_override=TS)
    tg,vg = load_tfrecords(tr, va, noise=-1, seed=42)
    tf.random.set_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    model = create_model('ViT_Max', timeslices=2, soft_quantize_layer=True,
                         initial_thresholds=thr0, threshold_offset=OFFSET, initial_levels=LEVELS)
    ckpt_dir, fp, hist = train(model=model, model_type='ViT_Max',
        weights_directory=os.path.join(OUT, 'weights'), training_generator=tg, validation_generator=vg,
        timeslices=2, train_type='soft_quantize_layer', epochs=EPOCHS, seed=SEED, verbose=2,
        extra_callbacks=[AbortOnStuck()])
    best_val = float(min(hist.history.get('val_loss', [np.inf])))
    thr, lv = get_best_thresholds(checkpoints=ckpt_dir, model_type='ViT_Max', timeslices=2,
        initial_thresholds=thr0, threshold_offset=OFFSET, initial_levels=LEVELS)
    out = {'seed': SEED, 'offset': OFFSET, 'thresholds': [float(t) for t in np.array(thr).ravel()],
           'best_val_loss': best_val, 'ckpt_dir': ckpt_dir, 'init_thresholds': thr0, 'time_stamps': TS}
    json.dump(out, open(os.path.join(OUT, 'result.json'), 'w'), indent=1)
    stamp(f"DONE: thresholds={out['thresholds']} best_val={best_val:.0f} ckpts={ckpt_dir}")
except Exception:
    traceback.print_exc()
    open(os.path.join(OUT, 'FAILED.txt'), 'w').write(traceback.format_exc()); sys.exit(1)
