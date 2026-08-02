"""
Part-1 threshold COLLECTION for the NEW dataset (3srb, 2t = samples [11,26]).

Runs ViT_Max + SoftQuantize (cosine k-anneal 1 -> 67) repeatedly with a fresh
seed + fresh random initial thresholds each time, and APPENDS every successful
run's learned 2-bit thresholds to threshold_runs.jsonl (one row per run). These
are the data points; the MEDIAN of each threshold across runs is computed later.

Morphed from legacy/smart_pixels_ml/train_loop.py: the seed-retry + AbortOnStuck
machinery escapes the bad-seed stuck-init (val_loss pinned at the ~103,616
zero-likelihood ceiling). AbortOnStuck is actually wired into the callbacks here
(the legacy file created it but forgot to add it), so a stuck seed dies fast.

Per-run epochs: the k-anneal scales with epochs (total_epochs = fit epochs), so
fewer epochs still completes the 1->67 anneal. 300 balances convergence vs
collecting many runs. Resumable: re-running continues appending to the JSONL.
"""
import os, sys, json, time, random, traceback

HELPERS = "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers"
sys.path.insert(0, HELPERS)

import numpy as np
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

from prepare_tfrecords import generate_tfrecords, load_tfrecords
from train import create_model, train, get_best_thresholds

NEW_DATASET = '/depot/cms/users/das214/datasets/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d'
OUT = '/work/users/das214/SmartPixels/smart-pixels-ml/runs/new_dataset_part1'
COLLECT = os.path.join(OUT, 'threshold_runs.jsonl')     # the data points (one run per line)
EPOCHS = 300                 # k-anneal completes at any epoch count; 300 = convergence vs many-runs balance
TARGET_RUNS = 25             # collect up to this many successful runs (kill earlier whenever you have enough)
# NEW dataset scale: charge values ~25-150 (units changed, no longer mV/e-).
THRESHOLD_OFFSET = 0.0        # SoftQuantize floor (user: 0 so first threshold is free to go low); assert requires offset < smallest threshold
SAMPLE_LOW = 25.0            # random initial thresholds drawn in [25,160] -> land around [35,60,150]
SAMPLE_HIGH = 160.0
INITIAL_LEVELS = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
NOISE1 = -1                  # NO noise (user: dataset is different; old [0,80] would swamp a 25-150 signal)
TIME_STAMPS = [11, 26]
SEEDS = [42 + 1000 * i for i in range(80)]      # plenty of distinct seeds
STUCK_THRESHOLD = 1e5        # val_loss above this = at the zero-likelihood ceiling
STUCK_PATIENCE = 20          # generous: a healthy seed escapes well before 20 epochs
ESCAPE_BELOW = 5e4           # best val_loss below this = genuinely learned (escaped)

os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT, 'train.log'), 'a', buffering=1)
stamp = lambda m: log.write(f"[{time.strftime('%H:%M:%S')}] {m}\n") or log.flush()


def sample_thresholds(seed, low, high, num):           # legacy train_loop.py
    rng = np.random.default_rng(seed)
    return sorted(rng.uniform(low=low, high=high, size=num).tolist())


class AbortOnStuck(tf.keras.callbacks.Callback):       # legacy, now actually wired in
    def __init__(self, threshold=1e5, patience=20):
        super().__init__(); self.thr = threshold; self.pat = patience; self.bad = 0
    def on_epoch_end(self, epoch, logs=None):
        v = (logs or {}).get("val_loss", np.inf)
        if v > self.thr or not np.isfinite(v):
            self.bad += 1
            if self.bad >= self.pat:
                print(f"[AbortOnStuck] val_loss {v:.1f} > {self.thr} for {self.pat} epochs -- bad seed, aborting.")
                self.model.stop_training = True
        else:
            self.bad = 0


def load_collected():
    if not os.path.exists(COLLECT):
        return []
    return [json.loads(l) for l in open(COLLECT) if l.strip()]


def running_median():
    rows = load_collected()
    if not rows:
        return None, 0
    arr = np.array([r['thresholds'] for r in rows])     # (n_runs, 3)
    return np.median(arr, axis=0).tolist(), len(rows)


def main():
    stamp(f"Part-1 COLLECTION on NEW dataset  samples={TIME_STAMPS}  epochs/run={EPOCHS}  target={TARGET_RUNS}")
    _, _, tfr_tr, tfr_val = generate_tfrecords(
        dataset_dir=NEW_DATASET, model_type='ViT_Max',
        train_batch_size=5000, val_batch_size=5000,
        select_contained=False, timeslices=2,
        tfrecords_exist=True, seed=42, time_stamps_override=TIME_STAMPS)
    tg, vg = load_tfrecords(tfr_tr, tfr_val, noise=NOISE1, seed=42)
    stamp(f"TFRecords loaded (noise={NOISE1} -> {'NONE' if NOISE1 == -1 else NOISE1})")

    done_seeds = {r['seed'] for r in load_collected()}
    n_success = len(done_seeds)
    if n_success:
        med, _ = running_median()
        stamp(f"resuming: {n_success} data points already collected, running median={med}")

    for seed in SEEDS:
        if n_success >= TARGET_RUNS:
            break
        if seed in done_seeds:
            continue
        tf.random.set_seed(seed); np.random.seed(seed); random.seed(seed)
        thr0 = sample_thresholds(seed, SAMPLE_LOW, SAMPLE_HIGH, 3)
        stamp(f"[run seed={seed}] random init thresholds={[round(t,1) for t in thr0]}")

        model = create_model('ViT_Max', timeslices=2, soft_quantize_layer=True,
                             initial_thresholds=thr0, threshold_offset=THRESHOLD_OFFSET,
                             initial_levels=INITIAL_LEVELS)
        ckpt_dir, fingerprint, hist = train(
            model=model, model_type='ViT_Max',
            weights_directory=os.path.join(OUT, 'weights'),
            training_generator=tg, validation_generator=vg,
            timeslices=2, train_type='soft_quantize_layer',
            epochs=EPOCHS, seed=seed, verbose=2,
            extra_callbacks=[AbortOnStuck(STUCK_THRESHOLD, STUCK_PATIENCE)])

        best_val = float(min(hist.history.get('val_loss', [np.inf])))
        ep_run = len(hist.history.get('val_loss', []))
        if best_val >= ESCAPE_BELOW:
            stamp(f"[run seed={seed}] STUCK (epochs={ep_run}, best_val={best_val:.1f}) -- skipping")
            continue

        thresholds, levels = get_best_thresholds(
            checkpoints=ckpt_dir, model_type='ViT_Max', timeslices=2,
            initial_thresholds=thr0, threshold_offset=THRESHOLD_OFFSET, initial_levels=INITIAL_LEVELS)
        rec = {'seed': seed, 'thresholds': [float(t) for t in np.array(thresholds).ravel()],
               'levels': [float(l) for l in np.array(levels).ravel()],
               'best_val_loss': best_val, 'epochs': ep_run, 'init_thresholds': thr0,
               'time_stamps': TIME_STAMPS}
        with open(COLLECT, 'a') as f:
            f.write(json.dumps(rec) + "\n")
        n_success += 1
        med, n = running_median()
        stamp(f"[COLLECTED {n}/{TARGET_RUNS}] seed={seed} thr={rec['thresholds']}  ||  running median={med}")

    med, n = running_median()
    summary = {'n_runs': n, 'median_thresholds': med,
               'levels': [0.0, 1.0, 2.0, 3.0], 'time_stamps': TIME_STAMPS,
               'note': 'median over collected runs; raw data points in threshold_runs.jsonl'}
    json.dump(summary, open(os.path.join(OUT, 'median_thresholds.json'), 'w'), indent=1)
    stamp(f"COLLECTION DONE: {n} runs, median thresholds={med}  (old dataset: [226.04, 601.01, 1456.94])")


try:
    main()
except Exception:
    traceback.print_exc()
    with open(os.path.join(OUT, 'FAILED.txt'), 'w') as f:
        f.write(traceback.format_exc())
    sys.exit(1)
