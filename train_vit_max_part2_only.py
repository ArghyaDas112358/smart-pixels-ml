"""
ViT_Max Part-2 (2-bit-optimized) training, 1000 epochs, picking up the
charge thresholds from the partially-completed Part-1 of the 1000-epoch
run (Part-1 was stopped at epoch 522/1000 with the user's blessing
because val_loss + thresholds were already in a healthy regime).

Reads thresholds from runs/vit_max_run_1000ep/optimized_thresholds.json
(written by the partial extractor) and writes summary.json + history
that match the v1 train_vit_max script's layout so the downstream
chained watcher can pick up.
"""
import os, sys, json, csv, time, traceback
sys.path.insert(0, "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers")
import numpy as np
import tensorflow as tf

for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass
print("GPUs:", tf.config.list_physical_devices("GPU"), flush=True)

import shutil, random
from prepare_tfrecords import generate_tfrecords, load_tfrecords
from train import create_model, cleanup_models_and_generators, save_performance_parquet
from loss import custom_loss

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
INIT_LVL = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)


def save_hist(h, name):
    json.dump(h.history, open(f"{OUT}/{name}.json", "w"), indent=1)
    keys = list(h.history.keys())
    with open(f"{OUT}/{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + keys)
        for i in range(len(h.history[keys[0]])):
            w.writerow([i + 1] + [h.history[k][i] for k in keys])


def stamp(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


try:
    t0 = time.time()
    thr_json = json.load(open(f"{OUT}/optimized_thresholds.json"))
    thresholds = np.array(thr_json["thresholds"], dtype=np.float32)
    levels = np.array(thr_json["levels"], dtype=np.float32)
    stamp(f"loaded thresholds from {thr_json.get('source_checkpoint')}: {thresholds.tolist()}")

    _, _, tfr_tr, tfr_val = generate_tfrecords(
        dataset_dir=DATASET, model_type=MODEL_TYPE,
        train_batch_size=5000, val_batch_size=5000,
        select_contained=False, timeslices=TIMESLICES,
        tfrecords_exist=True, seed=SEED,
    )
    stamp("TFRecords ready")

    stamp(f"=== PART 2: {MODEL_TYPE} 2bit-optimized, up to {EPOCHS} epochs with EarlyStopping(patience=50) ===")
    tg2, vg2 = load_tfrecords(tfr_tr, tfr_val, noise=-1, digitize=True,
                              digitize_levels=levels, digitize_thresholds=thresholds,
                              seed=SEED)
    m2 = create_model(MODEL_TYPE, timeslices=TIMESLICES, soft_quantize_layer=False)
    m2.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3), loss=custom_loss)
    fp2 = '%08x' % random.randrange(16 ** 8)
    ckpt2 = f"{WEIGHTS}/weights-2t-{MODEL_TYPE}-2bit_optimized-{fp2}-checkpoints"
    if os.path.exists(ckpt2):
        shutil.rmtree(ckpt2)
    os.makedirs(ckpt2)
    cb_mcp = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt2 + '/weights.{epoch:02d}-t{loss:.2f}-v{val_loss:.2f}.hdf5',
        save_weights_only=True, monitor='val_loss', save_best_only=False)
    cb_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
    cb_csv = tf.keras.callbacks.CSVLogger(f"{OUT}/history_part2_2bit_optimized.csv", append=False)
    stamp(f"checkpoints -> {ckpt2}")
    stamp(f"fingerprint = {fp2}")
    h2 = m2.fit(x=tg2, validation_data=vg2,
                callbacks=[cb_mcp, cb_early, cb_csv],
                epochs=EPOCHS, shuffle=False, verbose=1)
    save_hist(h2, "history_part2_2bit_optimized")
    stamp(f"Part 2 done at epoch {len(h2.history['loss'])} of {EPOCHS}.")

    stamp("=== EVALUATE: performance parquet on 3sr test split ===")
    save_performance_parquet(checkpoints=ckpt2, output_directory=PERF, test_generator=vg2,
                             model_type=MODEL_TYPE, train_type="2bit_optimized", fingerprint=fp2,
                             timeslices=TIMESLICES, soft_quantize_layer=False)

    # Combined summary: Part 1 info from the partial run + Part 2 info from now.
    PART1_CKPT_DIR = "/work/users/das214/SmartPixels/smart-pixels-ml/runs/weights/weights-2t-ViT_Max-soft_quantize_layer-1a62cd8b-checkpoints"
    summary = {
        "model_type": MODEL_TYPE,
        "epochs_each_part": EPOCHS,
        "part1_note": "Part 1 was run 1000 epochs but stopped at epoch 522 (user decision); thresholds and best Part-1 checkpoint extracted from that point.",
        "part1_best_val_loss": -43762.58,
        "part1_best_checkpoint": thr_json.get("source_checkpoint"),
        "part1_checkpoints": PART1_CKPT_DIR,
        "part2_final_val_loss": float(h2.history["val_loss"][-1]),
        "part2_best_val_loss": float(min(h2.history["val_loss"])),
        "optimized_thresholds": thr_json["thresholds"],
        "part2_checkpoints": ckpt2,
        "performance_dir": PERF,
        "total_seconds_part2": round(time.time() - t0),
    }
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
    stamp("TRAINING COMPLETE")
    print(json.dumps(summary, indent=1), flush=True)

except Exception:
    with open(f"{OUT}/FAILED.txt", "w") as f:
        f.write(traceback.format_exc())
    traceback.print_exc()
    sys.exit(1)
