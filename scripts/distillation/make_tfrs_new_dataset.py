"""
Generate TFRecords for the NEW dataset (3srb, 101 time samples at 10ps,
convolved to 200ps, ~300K events). Uses time samples [0, 100] = first + last,
the direct analog of the old 20-slice [0,19] '2t' convention.

Writes into <dataset>/TFR_files/2t/ (TFR_train + TFR_test), same layout the
training drivers load.
"""
import os, sys, time, json, traceback

HELPERS = "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers"
sys.path.insert(0, HELPERS)

NEW_DATASET = '/depot/cms/users/das214/datasets/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d'
OUT_LOG = '/work/users/das214/SmartPixels/smart-pixels-ml/runs/new_dataset_tfr/gen.log'
os.makedirs(os.path.dirname(OUT_LOG), exist_ok=True)
log = open(OUT_LOG, 'a', buffering=1)
stamp = lambda m: log.write(f"[{time.strftime('%H:%M:%S')}] {m}\n") or log.flush()

try:
    from prepare_tfrecords import generate_tfrecords
    stamp(f"start TFR generation: {NEW_DATASET}")
    stamp("time_stamps_override=[0,100] (first+last of 101 samples; analog of old [0,19])")
    t0 = time.time()
    BATCH = int(os.environ.get('TFR_BATCH', '5000'))   # user-specified: keep 5000
    TIME_STAMPS = [11, 26]                              # user-specified: samples 11 & 26 (110ps, 260ps)
    stamp(f"batch size = {BATCH}  time_stamps = {TIME_STAMPS}")
    out = generate_tfrecords(
        dataset_dir=NEW_DATASET,
        model_type='ViT_Max',
        train_batch_size=BATCH,
        val_batch_size=BATCH,
        select_contained=False,
        timeslices=2,
        tfrecords_exist=False,        # CREATE
        seed=42,
        max_workers=4,
        time_stamps_override=TIME_STAMPS,
    )
    stamp(f"DONE in {time.time()-t0:.0f}s")
    # count what was written
    tfrdir = os.path.join(NEW_DATASET, 'TFR_files', '2t')
    for sub in os.listdir(tfrdir):
        p = os.path.join(tfrdir, sub)
        stamp(f"  {sub}: {len(os.listdir(p))} files")
    with open(os.path.join(os.path.dirname(OUT_LOG), 'SUCCESS'), 'w') as f:
        f.write('ok\n')
except Exception:
    stamp("FAILED")
    stamp(traceback.format_exc())
    with open(os.path.join(os.path.dirname(OUT_LOG), 'FAILED.txt'), 'w') as f:
        f.write(traceback.format_exc())
    sys.exit(1)
