"""
Background training run: ViT_Max (transformer) through the full 2-bit pipeline.
Part 1 (soft-quantize threshold optimization) -> Part 2 (2-bit-digitized training)
-> evaluation. 30 epochs per part on the 3sr centeredIncidence dataset.

Outputs (under runs/vit_max_run_1000ep/):
  history_part1_soft_quantize.{json,csv}   training history of Part 1
  history_part2_2bit_optimized.{json,csv}  training history of Part 2
  optimized_thresholds.json                thresholds learned in Part 1
  summary.json                             run summary (losses, paths)
Performance parquet (preds + uncertainties) lands in runs/processed_parquets/.
"""
import os, sys, json, csv, time, traceback
sys.path.insert(0, "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers")
import numpy as np
import tensorflow as tf

# GPU: enable memory growth so we don't grab the whole A100
for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("memory_growth warning:", e, flush=True)
print("GPUs:", tf.config.list_physical_devices("GPU"), flush=True)

from prepare_tfrecords import generate_tfrecords, load_tfrecords
from train import (create_model, train, get_best_thresholds,
                   cleanup_models_and_generators, save_performance_parquet)

DATASET = "/depot/cms/users/das214/datasets/largerWindowPreliminary/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets"
RUNS = "/work/users/das214/SmartPixels/smart-pixels-ml/runs"
OUT = f"{RUNS}/vit_max_run_1000ep"
WEIGHTS = f"{RUNS}/weights"
PERF = f"{RUNS}/processed_parquets/test_3src/2bit_optimized"
for d in (OUT, WEIGHTS, PERF):
    os.makedirs(d, exist_ok=True)

MODEL_TYPE = "ViT_Max"
TIMESLICES = 2
SEED = 42
EPOCHS = 1000
INIT_THR = [247.8, 668.4, 1662.9]
THR_OFF = 80.0
INIT_LVL = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)


def save_hist(h, name):
    json.dump(h.history, open(f"{OUT}/{name}.json", "w"), indent=1)
    keys = list(h.history.keys())
    with open(f"{OUT}/{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + keys)
        for i in range(len(h.history[keys[0]])):
            w.writerow([i + 1] + [h.history[k][i] for k in keys])


def stamp(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


try:
    t0 = time.time()
    stamp("=== generate / load TFRecords (one-time conversion) ===")
    dtrain, dval, tfr_tr, tfr_val = generate_tfrecords(
        dataset_dir=DATASET, model_type=MODEL_TYPE,
        train_batch_size=5000, val_batch_size=5000,
        select_contained=False, timeslices=TIMESLICES,
        tfrecords_exist=True, seed=SEED,
    )
    stamp(f"TFRecords ready ({time.time()-t0:.0f}s). train={tfr_tr}")

    stamp(f"=== PART 1: {MODEL_TYPE} soft-quantize, {EPOCHS} epochs ===")
    tg1, vg1 = load_tfrecords(tfr_tr, tfr_val, noise=[0, 80], seed=SEED)
    m1 = create_model(MODEL_TYPE, timeslices=TIMESLICES, soft_quantize_layer=True,
                      initial_thresholds=INIT_THR, threshold_offset=THR_OFF, initial_levels=INIT_LVL)
    ckpt1, fp1, h1 = train(m1, MODEL_TYPE, WEIGHTS, tg1, vg1, timeslices=TIMESLICES,
                           train_type="soft_quantize_layer", epochs=EPOCHS, seed=SEED, verbose=1)
    save_hist(h1, "history_part1_soft_quantize")
    thr, lvl = get_best_thresholds(ckpt1, MODEL_TYPE, timeslices=TIMESLICES,
                                   initial_thresholds=INIT_THR, threshold_offset=THR_OFF, initial_levels=INIT_LVL)
    json.dump({"thresholds": [float(x) for x in thr], "levels": [float(x) for x in lvl]},
              open(f"{OUT}/optimized_thresholds.json", "w"), indent=1)
    stamp(f"Part 1 done. optimized thresholds={list(map(float, thr))}")
    cleanup_models_and_generators([m1, tg1, vg1])

    stamp(f"=== PART 2: {MODEL_TYPE} 2bit-optimized, {EPOCHS} epochs ===")
    tg2, vg2 = load_tfrecords(tfr_tr, tfr_val, noise=-1, digitize=True,
                              digitize_levels=INIT_LVL, digitize_thresholds=thr, seed=SEED)
    m2 = create_model(MODEL_TYPE, timeslices=TIMESLICES, soft_quantize_layer=False)
    ckpt2, fp2, h2 = train(m2, MODEL_TYPE, WEIGHTS, tg2, vg2, timeslices=TIMESLICES,
                           train_type="2bit_optimized", epochs=EPOCHS, seed=SEED, verbose=1)
    save_hist(h2, "history_part2_2bit_optimized")
    stamp("Part 2 done.")

    stamp("=== EVALUATE: performance parquet on 3sr test split ===")
    save_performance_parquet(checkpoints=ckpt2, output_directory=PERF, test_generator=vg2,
                             model_type=MODEL_TYPE, train_type="2bit_optimized", fingerprint=fp2,
                             timeslices=TIMESLICES, soft_quantize_layer=False)

    summary = {
        "model_type": MODEL_TYPE, "epochs_each_part": EPOCHS,
        "part1_final_val_loss": float(h1.history["val_loss"][-1]),
        "part1_best_val_loss": float(min(h1.history["val_loss"])),
        "part2_final_val_loss": float(h2.history["val_loss"][-1]),
        "part2_best_val_loss": float(min(h2.history["val_loss"])),
        "optimized_thresholds": [float(x) for x in thr],
        "part1_checkpoints": ckpt1, "part2_checkpoints": ckpt2,
        "performance_dir": PERF, "total_seconds": round(time.time() - t0),
    }
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
    stamp("TRAINING COMPLETE")
    print(json.dumps(summary, indent=1), flush=True)

except Exception:
    with open(f"{OUT}/FAILED.txt", "w") as f:
        f.write(traceback.format_exc())
    traceback.print_exc()
    sys.exit(1)
