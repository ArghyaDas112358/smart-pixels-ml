"""
Generate CLEAN (NO-noise), CONTAINED-cluster TFRecords for the new mV 3srb dataset.
Control for the noise-ceiling question: identical to the noise build except sigma=0.

- Time-sample indices [11, 26]  ("2 and 5 ns" timeslices), 2-timeslice / (2,16,16) input.
- Gaussian noise N(mu=0, sigma=4.64 mV) BAKED into the stored X (80e -> 4.64 mV).
- Contained selection: rawElectronChargeOriginal_atEdge < 50  (faithful port of the
  stock generator's `chargeOriginal_atEdge < 50`; the column was renamed in the
  with_contained_var dataset).  ~48.9% of rows survive.
- Batch size 5000, 80/20 split already on disk (contained/train parts 0-79,
  contained/test parts 80-99).
- Output mirrors the colleague's layout: TFR_files_2_5_noise_corr_contained/{TFR_train,TFR_test}

Approach: pre-filter each parquet to contained rows keeping only the [11,26] pixel
columns + labels (slim), write to a scratch dir, then run a noise-baking subclass of
OptimizedDataGenerator (select_contained=False, since rows are already filtered).

Env:
  SMOKE=1   -> only 2 train files + 1 test file, into *_SMOKE output dir.
"""
import os, sys, time, glob, shutil, traceback
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HELPERS = "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers"
sys.path.insert(0, HELPERS)

BASE = "/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d"
SRC_TRAIN = os.path.join(BASE, "contained", "train")
SRC_TEST  = os.path.join(BASE, "contained", "test")

SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT_DIR = os.path.join(BASE, "TFR_files_2_5_clean_contained" + ("_SMOKE" if SMOKE else ""))
SCRATCH = "/tmp/claude-978920/-work-users-das214-SmartPixels/7a0a041e-b87d-47e5-ae22-8b5c100f4193/scratchpad/filtered_2_5" + ("_SMOKE" if SMOKE else "")

TIME_STAMPS = [11, 26]
BATCH = 5000
NOISE_MU, NOISE_SIGMA = 0.0, 0.0   # CLEAN control: no baked noise
# Contained = original_atEdge == False (creator's definition; == zero charge at edge,
# bit-for-bit identical to rawElectronChargeOriginal_atEdge == 0). ~47.1% of rows.
CONTAIN_COL = "original_atEdge"
LABELS = ['x-midplane', 'y-midplane', 'cotAlpha', 'cotBeta']
SEED = 42
PIX_PER_T = 256  # 16x16

LOGDIR = "/work/users/das214/SmartPixels/smart-pixels-ml/runs/tfr_2_5_noise_contained"
os.makedirs(LOGDIR, exist_ok=True)
logf = open(os.path.join(LOGDIR, "gen_SMOKE.log" if SMOKE else "gen.log"), "a", buffering=1)
def stamp(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    logf.write(line + "\n"); logf.flush(); print(line, flush=True)


def time_cols(stamps):
    cols = []
    for t in stamps:
        cols += [str(i) for i in range(t * PIX_PER_T, (t + 1) * PIX_PER_T)]
    return cols

RECON = time_cols(TIME_STAMPS)  # 512 cols, in [11]-block then [26]-block order


def prefilter(src_dir, dst_dir, n_limit=None):
    """Filter to contained rows; keep RECON + LABELS; preserve part.N ordering/names."""
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_dir, "part.*.parquet")),
                   key=lambda p: int(p.split("part.")[1].split(".parquet")[0]))
    if n_limit:
        files = files[:n_limit]
    kept_total = 0; raw_total = 0
    for f in files:
        name = os.path.basename(f)
        df = pd.read_parquet(f, columns=RECON + LABELS + [CONTAIN_COL])
        raw_total += len(df)
        df = df.loc[~df[CONTAIN_COL].astype(bool)].reset_index(drop=True)  # contained = not at edge
        df = df[RECON + LABELS]  # drop containment col, fix column order
        kept_total += len(df)
        df.to_parquet(os.path.join(dst_dir, name), index=False)
    stamp(f"  prefilter {src_dir} -> {dst_dir}: {len(files)} files, "
          f"{kept_total}/{raw_total} rows kept ({100*kept_total/max(raw_total,1):.1f}%)")
    return kept_total


def main():
    t0 = time.time()
    stamp("=" * 70)
    stamp(f"SMOKE={SMOKE}  TIME_STAMPS={TIME_STAMPS}  BATCH={BATCH}  "
          f"noise=N({NOISE_MU},{NOISE_SIGMA})  contained={CONTAIN_COL}==False")
    stamp(f"OUT_DIR={OUT_DIR}")

    # 1) pre-filter to contained, slim columns, into scratch
    tmp_train = os.path.join(SCRATCH, "train")
    tmp_test  = os.path.join(SCRATCH, "test")
    nlim = 2 if SMOKE else None
    nlim_t = 1 if SMOKE else None
    stamp("pre-filtering...")
    prefilter(SRC_TRAIN, tmp_train, nlim)
    prefilter(SRC_TEST, tmp_test, nlim_t)

    # 2) noise-baking generator via monkeypatched generate_tfrecords
    import prepare_tfrecords
    from OptimizedDataGenerator_v3 import OptimizedDataGenerator

    class NoisyGen(OptimizedDataGenerator):
        def prepare_batch_data(self, batch_index):
            X, y = super().prepare_batch_data(batch_index)
            rng = np.random.default_rng(SEED * 100003 + int(batch_index))
            X = X + rng.normal(NOISE_MU, NOISE_SIGMA, size=X.shape).astype(X.dtype)
            return X, y

    prepare_tfrecords.OptimizedDataGenerator = NoisyGen  # inject

    stamp("generating TFRecords (noise baked in)...")
    out = prepare_tfrecords.generate_tfrecords(
        dataset_dir=SCRATCH,
        model_type='ViT_Max',          # -> 4 labels (x,y,cotA,cotB)
        train_batch_size=BATCH,
        val_batch_size=BATCH,
        select_contained=False,        # already filtered
        timeslices=2,
        tfrecords_exist=False,
        seed=SEED,
        max_workers=4,
        time_stamps_override=TIME_STAMPS,
    )
    _, _, tfr_train, tfr_val = out
    stamp(f"generated: train={tfr_train}  val={tfr_val}")

    # 3) relocate scratch/TFR_files/2t/{TFR_train,TFR_test} -> OUT_DIR
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    for sub, srcp in (("TFR_train", tfr_train), ("TFR_test", tfr_val)):
        dst = os.path.join(OUT_DIR, sub)
        shutil.move(srcp, dst)
        nrec = len(glob.glob(os.path.join(dst, "*.tfrecord")))
        stamp(f"  {sub}: {nrec} tfrecords -> {dst}")

    # cleanup scratch
    shutil.rmtree(SCRATCH, ignore_errors=True)
    stamp(f"DONE in {time.time()-t0:.0f}s -> {OUT_DIR}")
    with open(os.path.join(LOGDIR, "SUCCESS_SMOKE" if SMOKE else "SUCCESS"), "w") as fh:
        fh.write("ok\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        stamp("FAILED\n" + traceback.format_exc())
        sys.exit(1)
