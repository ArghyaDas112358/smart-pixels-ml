"""
Generate BAKED-NOISE, CONTAINED-cluster TFRecords for the "1 ns / 6 ns" slice pair.

Faithful clone of make_tfrs_contained_2_5_noise.py, changed only for the 1_6 study:
- Time-sample indices [6, 31]  ("1 and 6 ns"; index = 5*ns + 1, same map that gives
  2ns,5ns -> [11,26]).  2-timeslice / (2,16,16) input.
- Gaussian noise N(mu=0, sigma=4.64 mV) BAKED in, i.i.d. per sample (same as 2_5).
- Contained = original_atEdge == False (creator's definition).
- FULL dataset: all 80 train + 20 test contained parts (never a subset).
- Output: TFR_files_1_6_iid_contained/{TFR_train,TFR_test}  (das214-owned; deliberately
  NOT the colleague's TFR_files_1_6_noise_corr_contained, which is uid 2076095's).
- max_workers=1: the generator's stats pass collects row counts via as_completed(), so
  >1 worker mis-assigns counts and silently truncates tails (the 737-row loss in the 2_5
  build). Serial => row-exact. (Only deviation from the 2_5 script; thresholds unaffected.)

Env: SMOKE=1 -> 2 train + 1 test file into *_SMOKE dir.
"""
import os, sys, time, glob, shutil, traceback
import numpy as np
import pandas as pd

HELPERS = "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers"
sys.path.insert(0, HELPERS)

BASE = "/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d"
SRC_TRAIN = os.path.join(BASE, "contained", "train")
SRC_TEST  = os.path.join(BASE, "contained", "test")

SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT_DIR = os.path.join(BASE, "TFR_files_1_6_iid_contained" + ("_SMOKE" if SMOKE else ""))
SCRATCH = "/work/users/das214/SmartPixels/scratch_1_6" + ("_SMOKE" if SMOKE else "")

TIME_STAMPS = [6, 31]                 # 1 ns, 6 ns
BATCH = 5000
NOISE_MU, NOISE_SIGMA = 0.0, 4.64
CONTAIN_COL = "original_atEdge"       # contained = False
LABELS = ['x-midplane', 'y-midplane', 'cotAlpha', 'cotBeta']
SEED = 42
PIX_PER_T = 256

LOGDIR = "/work/users/das214/SmartPixels/smart-pixels-ml/runs/tfr_1_6_iid_contained"
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

RECON = time_cols(TIME_STAMPS)        # 512 cols: [6]-block then [31]-block


def prefilter(src_dir, dst_dir, n_limit=None):
    """Filter to contained rows; keep RECON + LABELS + dropna; preserve part.N order."""
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
        df = df.loc[~df[CONTAIN_COL].astype(bool)]
        df = df.dropna(subset=RECON)                 # exact batch plan downstream
        df = df[RECON + LABELS].reset_index(drop=True)
        kept_total += len(df)
        df.to_parquet(os.path.join(dst_dir, name), index=False)
    stamp(f"  prefilter {src_dir} -> {dst_dir}: {len(files)} files, "
          f"{kept_total}/{raw_total} rows kept ({100*kept_total/max(raw_total,1):.1f}%)")
    return kept_total


def main():
    t0 = time.time()
    stamp("=" * 70)
    stamp(f"1_6 SMOKE={SMOKE}  TIME_STAMPS={TIME_STAMPS}  BATCH={BATCH}  "
          f"noise=N({NOISE_MU},{NOISE_SIGMA}) iid  contained={CONTAIN_COL}==False")
    stamp(f"OUT_DIR={OUT_DIR}")

    tmp_train = os.path.join(SCRATCH, "train")
    tmp_test  = os.path.join(SCRATCH, "test")
    nlim = 2 if SMOKE else None
    nlim_t = 1 if SMOKE else None
    stamp("pre-filtering (contained + dropna)...")
    prefilter(SRC_TRAIN, tmp_train, nlim)
    prefilter(SRC_TEST, tmp_test, nlim_t)

    import prepare_tfrecords
    from OptimizedDataGenerator_v3 import OptimizedDataGenerator

    class NoisyGen(OptimizedDataGenerator):
        def prepare_batch_data(self, batch_index):
            X, y = super().prepare_batch_data(batch_index)
            rng = np.random.default_rng(SEED * 100003 + int(batch_index))
            X = X + rng.normal(NOISE_MU, NOISE_SIGMA, size=X.shape).astype(X.dtype)
            return X, y

    prepare_tfrecords.OptimizedDataGenerator = NoisyGen

    stamp("generating TFRecords (noise baked in)...")
    out = prepare_tfrecords.generate_tfrecords(
        dataset_dir=SCRATCH,
        model_type='ViT_Max',
        train_batch_size=BATCH,
        val_batch_size=BATCH,
        select_contained=False,
        timeslices=2,
        tfrecords_exist=False,
        seed=SEED,
        max_workers=1,                 # row-exact (see header)
        time_stamps_override=TIME_STAMPS,
    )
    _, _, tfr_train, tfr_val = out
    stamp(f"generated: train={tfr_train}  val={tfr_val}")

    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    for sub, srcp in (("TFR_train", tfr_train), ("TFR_test", tfr_val)):
        dst = os.path.join(OUT_DIR, sub)
        shutil.move(srcp, dst)
        nrec = len(glob.glob(os.path.join(dst, "*.tfrecord")))
        stamp(f"  {sub}: {nrec} tfrecords -> {dst}")

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
