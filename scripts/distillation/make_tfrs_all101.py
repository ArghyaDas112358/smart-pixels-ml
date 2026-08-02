"""
Generate ALL-101-SLICE TFRecords for SoftRouter slice discovery (plan step 1-2).

- FULL DATASET: all 80 train + 20 test parts of the contained split (user rule:
  never subset a dataset without explicit permission — full statistics always).
- Prefilter (per the code-verified data rules in docs/soft_router_plan.md §5):
    contained = original_atEdge == False   AND   dropna over ALL pixel columns
  (the generator plans batches BEFORE its own dropna -> silent row loss otherwise).
- Noise N(0, 4.64 mV) BAKED into all 101 slices, i.i.d. per sample (same model as
  the production 2_5 TFRs).
  *** OPEN QUESTION (Shiqi): real noise is shaped by the ~200 ps preamp response,
  so adjacent 10 ps samples should be CORRELATED. i.i.d. flatters near-adjacent
  slice pairs during discovery. If the answer is "correlated", set NOISE_MODE
  = "corr200ps" below and regenerate (white noise convolved with the same
  response kernel, rescaled to sigma=4.64). ***
- labels_scale PINNED to the production 2_5 TFR train metadata, so discovery /
  validation / baseline all live in identical label units.
- Batch 5000 (standing convention). Output mirrors the production layout:
    <BASE>/TFR_files_all101_noise_contained_discovery/{TFR_train,TFR_test}
  plus MANIFEST.json recording subset, noise model, pinned scale.

Env: SMOKE=1 -> 2 train + 1 test files into *_SMOKE dir.
"""
import os, sys, json, time, glob, shutil, traceback
import numpy as np
import pandas as pd

HELPERS = "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers"
sys.path.insert(0, HELPERS)

BASE = "/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d"
SRC_TRAIN = os.path.join(BASE, "contained", "train")
SRC_TEST = os.path.join(BASE, "contained", "test")
PROD_META = os.path.join(BASE, "TFR_files_2_5_noise_corr_contained", "TFR_train", "metadata.json")

SMOKE = os.environ.get("SMOKE", "0") == "1"
N_TRAIN, N_TEST = (2, 1) if SMOKE else (80, 20)   # FULL dataset (smoke = machinery check only)
OUT_DIR = os.path.join(BASE, "TFR_files_all101_noise_contained_discovery" + ("_SMOKE" if SMOKE else ""))
SCRATCH = "/work/users/das214/SmartPixels/scratch_all101" + ("_SMOKE" if SMOKE else "")

N_SLICES = 101
PIX_PER_T = 256                       # 16x16
RECON = [str(i) for i in range(N_SLICES * PIX_PER_T)]   # ALL 25,856 pixel columns
LABELS = ['x-midplane', 'y-midplane', 'cotAlpha', 'cotBeta']
CONTAIN_COL = "original_atEdge"       # contained = False (creator's definition)
BATCH = 5000
NOISE_MU, NOISE_SIGMA = 0.0, 4.64
NOISE_MODE = "iid"                    # "iid" | "corr200ps" (pending Shiqi's answer)
SEED = 42
FALLBACK_LABELS_SCALE = [123.7301, 30.9423, 6.5879, 1.8481]  # prod 2_5 train metadata

LOGDIR = "/work/users/das214/SmartPixels/smart-pixels-ml/runs/tfr_all101_discovery"
os.makedirs(LOGDIR, exist_ok=True)
logf = open(os.path.join(LOGDIR, "gen_SMOKE.log" if SMOKE else "gen.log"), "a", buffering=1)
def stamp(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    logf.write(line + "\n"); logf.flush(); print(line, flush=True)


def pinned_labels_scale():
    """One label scale for discovery + validation + baseline (plan §5)."""
    try:
        ls = json.load(open(PROD_META))["labels_scale"]
        stamp(f"labels_scale pinned from production metadata: {[round(v,4) for v in ls]}")
        return np.asarray(ls, dtype=np.float32)
    except Exception as e:
        stamp(f"WARNING: could not read {PROD_META} ({e}); using fallback constants")
        return np.asarray(FALLBACK_LABELS_SCALE, dtype=np.float32)


def numeric_first(src_dir, n):
    files = sorted(glob.glob(os.path.join(src_dir, "part.*.parquet")),
                   key=lambda p: int(p.split("part.")[1].split(".parquet")[0]))
    return files[:n]


def prefilter(files, dst_dir):
    """contained + dropna(ALL pixel cols) -> slim parquet (RECON+LABELS)."""
    os.makedirs(dst_dir, exist_ok=True)
    kept = raw = 0
    for f in files:
        name = os.path.basename(f)
        t0 = time.time()
        df = pd.read_parquet(f, columns=RECON + LABELS + [CONTAIN_COL])
        raw += len(df)
        df = df.loc[~df[CONTAIN_COL].astype(bool)]
        df = df.dropna(subset=RECON)              # exact batch plan downstream
        df = df[RECON + LABELS].reset_index(drop=True)
        kept += len(df)
        df.to_parquet(os.path.join(dst_dir, name), index=False)
        stamp(f"  {name}: {len(df)} rows kept ({time.time()-t0:.0f}s)")
        del df
    stamp(f"  -> {dst_dir}: {kept}/{raw} rows kept ({100*kept/max(raw,1):.1f}%)")
    return kept


def main():
    t0 = time.time()
    stamp("=" * 70)
    stamp(f"ALL-101 discovery TFRs  SMOKE={SMOKE}  subset={N_TRAIN}+{N_TEST} files  "
          f"batch={BATCH}  noise={NOISE_MODE} N({NOISE_MU},{NOISE_SIGMA})")
    stamp(f"OUT_DIR={OUT_DIR}")
    if NOISE_MODE != "iid":
        raise NotImplementedError("corr200ps noise pending Shiqi's answer on the noise model")

    ls_pin = pinned_labels_scale()

    tr_files = numeric_first(SRC_TRAIN, N_TRAIN)
    te_files = numeric_first(SRC_TEST, N_TEST)
    stamp(f"train subset: {os.path.basename(tr_files[0])}..{os.path.basename(tr_files[-1])}  "
          f"test subset: {os.path.basename(te_files[0])}..{os.path.basename(te_files[-1])}")

    if os.path.isdir(SCRATCH):
        shutil.rmtree(SCRATCH)
    stamp("pre-filtering (contained + dropna over all 25,856 pixel cols)...")
    n_tr = prefilter(tr_files, os.path.join(SCRATCH, "train"))
    n_te = prefilter(te_files, os.path.join(SCRATCH, "test"))

    import prepare_tfrecords
    from OptimizedDataGenerator_v3 import OptimizedDataGenerator

    class NoisyGen(OptimizedDataGenerator):
        """Bake seeded N(mu, sigma) into every slice at TFR-write time."""
        def prepare_batch_data(self, batch_index):
            X, y = super().prepare_batch_data(batch_index)
            rng = np.random.default_rng(SEED * 100003 + int(batch_index))
            X = X + rng.normal(NOISE_MU, NOISE_SIGMA, size=X.shape).astype(X.dtype)
            return X, y

    prepare_tfrecords.OptimizedDataGenerator = NoisyGen

    stamp("generating all-101 TFRecords (noise baked, labels_scale pinned)...")
    out = prepare_tfrecords.generate_tfrecords(
        dataset_dir=SCRATCH,
        model_type='ViT_Max',
        train_batch_size=BATCH,
        val_batch_size=BATCH,
        select_contained=False,          # already filtered upstream
        timeslices=N_SLICES,
        tfrecords_exist=False,
        seed=SEED,
        # max_workers MUST be 1: the generator's stats pass collects per-file row
        # counts via as_completed() (v3.py:331) -> with >1 workers the counts land
        # on the WRONG files (completion order != file order) and tail rows are
        # silently truncated (this, not NaNs, caused the 737-row loss in the 2_5
        # set). A single worker completes FIFO, so the offsets stay aligned.
        max_workers=1,
        time_stamps_override=list(range(N_SLICES)),
        labels_scale=ls_pin,             # PINNED
    )
    _, _, tfr_train, tfr_val = out

    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    counts = {}
    for sub, srcp in (("TFR_train", tfr_train), ("TFR_test", tfr_val)):
        dst = os.path.join(OUT_DIR, sub)
        shutil.move(srcp, dst)
        counts[sub] = len(glob.glob(os.path.join(dst, "*.tfrecord")))
        stamp(f"  {sub}: {counts[sub]} tfrecords -> {dst}")

    json.dump({
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "SoftRouter slice discovery (all 101 slices, joint w/ SoftQuantize)",
        "subset": {"train_files": [os.path.basename(f) for f in tr_files],
                   "test_files": [os.path.basename(f) for f in te_files],
                   "train_rows": n_tr, "test_rows": n_te},
        "prefilter": "original_atEdge==False AND dropna(all pixel cols)",
        "noise": {"mode": NOISE_MODE, "mu": NOISE_MU, "sigma": NOISE_SIGMA,
                  "open_question": "correlated (~200ps) noise pending Shiqi"},
        "labels_scale_pinned": [float(v) for v in ls_pin],
        "batch": BATCH, "time_stamps": "0..100", "seed": SEED,
    }, open(os.path.join(OUT_DIR, "MANIFEST.json"), "w"), indent=1)

    shutil.rmtree(SCRATCH, ignore_errors=True)
    stamp(f"DONE in {time.time()-t0:.0f}s -> {OUT_DIR}  ({counts})")
    with open(os.path.join(LOGDIR, "SUCCESS_SMOKE" if SMOKE else "SUCCESS"), "w") as fh:
        fh.write("ok\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        stamp("FAILED\n" + traceback.format_exc())
        sys.exit(1)
