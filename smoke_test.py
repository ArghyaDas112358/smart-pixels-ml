"""
Smoke test for the smart-pixels-ml 2bit_optimization pipeline.
Exercises Part 1 (soft-quantize threshold opt) + Part 2 (2-bit training) end-to-end
on a tiny symlinked dataset for 2 epochs each. Proves env + helpers + data path work
before committing to the full 1000+1000 epoch run.
"""
import os, sys
sys.path.insert(0, "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers")
import numpy as np
import json

from prepare_tfrecords import generate_tfrecords, load_tfrecords
from train import create_model, train, get_best_thresholds, cleanup_models_and_generators

SMK = "/work/users/das214/SmartPixels/smart-pixels-ml/runs/smoke_dataset"
WEIGHTS = "/work/users/das214/SmartPixels/smart-pixels-ml/runs/smoke_weights"
os.makedirs(WEIGHTS, exist_ok=True)

model_type = "Conv2D_Max"
timeslices = 2
seed = 42
initial_thresholds = [247.8, 668.4, 1662.9]
threshold_offset = 80.0
initial_levels = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

print("=== PART 1: generate TFRecords + soft-quantize training (2 epochs) ===", flush=True)
dtrain, dval, tfr_train, tfr_val = generate_tfrecords(
    dataset_dir=SMK,
    model_type=model_type,
    train_batch_size=500,
    val_batch_size=500,
    select_contained=False,
    timeslices=timeslices,
    tfrecords_exist=False,
    seed=seed,
)
tg1, vg1 = load_tfrecords(tfr_train, tfr_val, noise=[0, 80], seed=seed)
m1 = create_model(model_type, timeslices=timeslices, soft_quantize_layer=True,
                  initial_thresholds=initial_thresholds, threshold_offset=threshold_offset,
                  initial_levels=initial_levels)
ckpt1, fp1, h1 = train(m1, model_type, WEIGHTS, tg1, vg1, timeslices=timeslices,
                       train_type="soft_quantize_layer", epochs=2, seed=seed, verbose=2)
thresholds, levels = get_best_thresholds(ckpt1, model_type, timeslices=timeslices,
                                         initial_thresholds=initial_thresholds,
                                         threshold_offset=threshold_offset, initial_levels=initial_levels)
print(f"Optimized thresholds: {thresholds}", flush=True)
cleanup_models_and_generators([m1, tg1, vg1])

print("=== PART 2: 2-bit digitized training (2 epochs) ===", flush=True)
tg2, vg2 = load_tfrecords(tfr_train, tfr_val, noise=-1, digitize=True,
                          digitize_levels=initial_levels, digitize_thresholds=thresholds, seed=seed)
m2 = create_model(model_type, timeslices=timeslices, soft_quantize_layer=False)
ckpt2, fp2, h2 = train(m2, model_type, WEIGHTS, tg2, vg2, timeslices=timeslices,
                       train_type="2bit_optimized", epochs=2, seed=seed, verbose=2)
print("\nSMOKE TEST PASSED: Part 1 + Part 2 ran end-to-end.", flush=True)
print(f"  Part1 val_loss history: {h1.history.get('val_loss')}", flush=True)
print(f"  Part2 val_loss history: {h2.history.get('val_loss')}", flush=True)
