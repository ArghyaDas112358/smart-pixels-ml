"""
Verify the OptimizedDataGenerator_v3 leak fix (cached per-file iterators).

Two things must hold, and the first matters more than the second: a data-loading
"optimisation" that silently changes what the model sees would corrupt every
result in the project.

  1. EQUIVALENCE -- batches are bit-identical to the old build-per-call code
     path, across a re-read (same index twice) and across an on_epoch_end file
     reshuffle.
  2. NO GROWTH -- repeatedly pulling batches no longer creates a new
     TFRecordDataset/map/iterator each time (checked by object count + host RSS).

Run on CPU: CUDA_VISIBLE_DEVICES='' python smoke_dg_leakfix.py
"""
import os, sys, gc, resource
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HELPERS = "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers"
sys.path.insert(0, HELPERS)

import numpy as np
import tensorflow as tf
from OptimizedDataGenerator_v3 import OptimizedDataGenerator

BASE = os.environ.get(
    "SMARTPIX_DATA_BASE",
    "/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d")
TFR = os.path.join(os.environ.get("SMARTPIX_TFR", os.path.join(BASE, "TFR_files_all101_noise_contained_discovery")),
                   "TFR_test")

FAILED = []
def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok: FAILED.append(name)

def rss():
    """CURRENT resident set size in MB.

    NOT resource.getrusage().ru_maxrss -- that is a high-water MARK which never
    decreases, so it cannot tell a leak from a one-time plateau (it reported
    +2.5 GB here purely from touching 8 x 517 MB files for the first time).
    /proc/self/statm field 2 is resident pages right now.
    """
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024.0 ** 2


def old_getitem(gen, batch_index):
    """The pre-fix code path, reproduced verbatim for comparison."""
    path = gen.tfrecord_filenames[batch_index]
    raw = tf.data.TFRecordDataset(path)
    parsed = raw.map(gen._parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    X, y = next(iter(parsed))
    X = tf.reshape(X, [-1, *X.shape[1:]])
    y = tf.reshape(y, [-1, *y.shape[1:]])
    return X.numpy(), y.numpy()


# noise=-1 and shuffle=False so the comparison is deterministic (the generator
# otherwise adds fresh Gaussian noise and permutes rows per call by design).
gen = OptimizedDataGenerator(load_from_tfrecords_dir=TFR, shuffle=False, seed=42, noise=-1)
print(f"generator: {len(gen)} batches from {TFR}")

# ---- 1. equivalence on first read -------------------------------------------
worst = 0.0
for i in (0, 1, len(gen) - 1):
    Xo, yo = old_getitem(gen, i)
    Xn, yn = gen[i]
    Xn = np.asarray(Xn); yn = np.asarray(yn)
    worst = max(worst, float(np.abs(Xo - Xn).max()), float(np.abs(yo - yn).max()))
check("batches identical to the pre-fix code path", worst == 0.0,
      f"max abs diff {worst:.3e} over 3 batches")

# ---- 2. re-reading the SAME index returns the SAME batch --------------------
# (the persistent iterator must not advance past the file's single batch)
X1, _ = gen[0]; X2, _ = gen[0]; X3, _ = gen[0]
same = np.array_equal(np.asarray(X1), np.asarray(X2)) and np.array_equal(np.asarray(X2), np.asarray(X3))
check("repeated reads of one index are stable (.take(1).repeat())", same)

# ---- 3. survives an on_epoch_end file reshuffle ------------------------------
# The cache is keyed by PATH, so a reshuffled index must still deliver that
# file's data -- this is what a batch_index-keyed cache would get wrong.
before = {}
for i in range(len(gen)):
    before[str(gen.tfrecord_filenames[i])] = np.asarray(gen[i][0]).copy()
gen.on_epoch_end()
ok_after = True
for i in range(len(gen)):
    p = str(gen.tfrecord_filenames[i])
    if not np.array_equal(np.asarray(gen[i][0]), before[p]):
        ok_after = False; break
check("correct data after on_epoch_end reshuffles the file order", ok_after,
      f"{len(gen)} files re-checked by path")

# ---- 4. no per-call pipeline construction ------------------------------------
gc.collect()
n0 = len(gc.get_objects())
trace = []
for p in range(20):
    for i in range(len(gen)):
        _ = gen[i]
    gc.collect()
    trace.append(rss())
n1 = len(gc.get_objects())
check("live object count stable over 20 passes", abs(n1 - n0) < 5000,
      f"{n0} -> {n1} ({n1 - n0:+d})")

# A leak is LINEAR growth that persists; a warm-up is growth that plateaus. Fit
# the slope over the SECOND HALF only, after first-touch allocation has settled.
half = trace[len(trace) // 2:]
slope = float(np.polyfit(np.arange(len(half), dtype=float), np.array(half), 1)[0])
per_read = slope / len(gen)
print(f"    RSS trace (MB): first {trace[0]:.0f}  mid {trace[len(trace)//2]:.0f}  last {trace[-1]:.0f}")
check("host RSS flat once warm (2nd-half slope)", abs(slope) < 20.0,
      f"{slope:+.1f} MB/pass = {per_read:+.2f} MB per batch read "
      f"(pre-fix was ~4.4 MB/read)")
check("iterator cache bounded by file count", len(getattr(gen, '_tfr_iters', {})) == len(gen),
      f"{len(getattr(gen, '_tfr_iters', {}))} cached iterators for {len(gen)} files")

print()
if FAILED:
    print(f"SMOKE: {len(FAILED)} FAILED -> {FAILED}"); sys.exit(1)
print("SMOKE: all checks PASS")
