# tests/test_smoke_pipeline.py
"""
Smoke test for the SmartPixels ML pipeline.

# CHANGE: Converted the original interactive script into a pytest test using tmp_path.
# REASON: CI-friendly, non-interactive, auto-cleaned temporary workspace.

# CHANGE: Removed colorized logging and input(), replaced with plain prints and assertions.
# REASON: Deterministic output in CI; no interactive prompts.

# CHANGE: Switched to package-safe imports by augmenting sys.path for 'smart_pixels_ml/src'.
# REASON: Ensure imports work without installing the package. Keeps repo structure intact.

# CHANGE: Minimized dataset size and epochs.
# REASON: Keep test fast and reliable under CI time limits.

# CHANGE: Separated TFRecord generation and loading phases with the same parameters used in codebase.
# REASON: Validate both code paths without heavy runtime.

# CHANGE: Skips the test gracefully if required heavy deps are missing (tensorflow/pyarrow).
# REASON: Clear failure mode; prevents cryptic ImportErrors in CI if requirements are incomplete.
"""

from __future__ import annotations
import os
import sys
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# --- Optional dependency checks (skip test if missing heavy deps) ---
pytest.importorskip("tensorflow", reason="TensorFlow not installed")
try:
    import pyarrow  # noqa: F401
except Exception:
    pytest.skip("pyarrow/fastparquet not available for Parquet IO", allow_module_level=True)

import tensorflow as tf  # after importorskip

# --- Make 'smart_pixels_ml/src' importable without packaging install ---
REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = REPO_ROOT / "smart_pixels_ml"
SRC_DIR = PKG_DIR / "src"

# Insert at beginning so local sources win over site-packages
sys.path.insert(0, str(SRC_DIR))

# Now imports align with your tree:
from DG.OptimizedDataGenerator_v2p5 import OptimizedDataGenerator  # type: ignore
from losses.loss import custom_loss  # type: ignore
from models.models import CreateModel  # type: ignore


# ----------------------------
# Small, fast test parameters
# ----------------------------
DATA_SET_SIZE = 24   # keep > BATCH_SIZE to exercise batching
BATCH_SIZE = 4
TRAIN_FILE_COUNT = 2
VAL_FILE_COUNT = 2
NUM_EPOCHS = 1

INPUT_SHAPE_HW_C = (13, 21, 2)   # Model expects (H,W,C) = (13,21,2)
TIME_STAMPS = [0, 19]            # Two timestamps -> 2 channels
TIME_LEN = 20                    # Matches stamps above
SCALINGS = [75.0, 18.75, 10.0, 1.22]
LABELS = ["x-midplane", "y-midplane", "cotAlpha", "cotBeta"]


def _generate_dummy_parquet(data_dir: Path, labels_dir: Path, n_files: int) -> None:
    """
    Generate tiny dummy data/label parquet files to feed DG.
    Data shape matches the code's expectations: (N, 13, 21, 20).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_files):
        rng = np.random.default_rng(1234 + i)
        # [N, H, W, T]
        sample = rng.random((DATA_SET_SIZE, INPUT_SHAPE_HW_C[0], INPUT_SHAPE_HW_C[1], TIME_LEN), dtype=np.float32)
        flat = sample.reshape(DATA_SET_SIZE, -1)  # flatten to columns for parquet
        cols = [str(j) for j in range(flat.shape[1])]
        df_data = pd.DataFrame(flat, columns=cols)
        df_data["event_id"] = np.arange(DATA_SET_SIZE)

        labels = rng.random((DATA_SET_SIZE, len(LABELS)), dtype=np.float32)
        df_lbl = pd.DataFrame(labels, columns=LABELS)
        df_lbl["event_id"] = np.arange(DATA_SET_SIZE)

        df_data.to_parquet(data_dir / f"recon3D_data_{i}.parquet", index=False)
        df_lbl.to_parquet(labels_dir / f"labels_data_{i}.parquet", index=False)


def _assert_nonempty_glob(pattern: str) -> list[str]:
    files = glob.glob(pattern)
    assert files, f"No files matched pattern: {pattern}"
    return files


@pytest.mark.timeout(300)
def test_end_to_end_smoke(tmp_path: Path) -> None:
    """
    End-to-end smoke:
      1) Generate small Parquet datasets
      2) Instantiate DGs to write TFRecords (train/val)
      3) Reload from TFRecords
      4) Build a tiny model and run 1 epoch
      5) Evaluate on validation generator
    """

    # --- Layout under tmp_path (auto-cleaned by pytest) ---
    test_root = tmp_path
    data_dir = test_root / "data"
    labels_dir = test_root / "labels"
    tfrecords_train = test_root / "tfrecords" / "train"
    tfrecords_val = test_root / "tfrecords" / "validation"
    base_model_dir = test_root / "base_model"

    # Ensure directories exist
    for d in (data_dir, labels_dir, tfrecords_train, tfrecords_val, base_model_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1) Generate tiny Parquet datasets
    _generate_dummy_parquet(data_dir, labels_dir, n_files=max(TRAIN_FILE_COUNT, VAL_FILE_COUNT))

    # Quick sanity
    _assert_nonempty_glob(str(data_dir / "*.parquet"))
    _assert_nonempty_glob(str(labels_dir / "*.parquet"))

    # 2) Instantiate DG to WRITE TFRecords
    train_writer = OptimizedDataGenerator(
        data_directory_path=str(data_dir),
        labels_directory_path=str(labels_dir),
        is_directory_recursive=False,
        file_type="parquet",
        data_format="3D",
        batch_size=BATCH_SIZE,
        file_count=TRAIN_FILE_COUNT,
        to_standardize=True,
        include_y_local=False,
        labels_list=LABELS,
        scaling_list=SCALINGS,
        input_shape=(2, 13, 21),     # (C, H, W) as per your generator API
        transpose=(0, 2, 3, 1),      # -> (C,H,W) -> (H,W,C) for Keras
        files_from_end=False,
        shuffle=True,
        tfrecords_dir=str(tfrecords_train),
        use_time_stamps=TIME_STAMPS,  # 2 time slices -> 2 channels
        max_workers=1,
        seed=42,
        quantize=True,
    )

    val_writer = OptimizedDataGenerator(
        data_directory_path=str(data_dir),
        labels_directory_path=str(labels_dir),
        is_directory_recursive=False,
        file_type="parquet",
        data_format="3D",
        batch_size=BATCH_SIZE,
        file_count=VAL_FILE_COUNT,
        to_standardize=True,
        include_y_local=False,
        labels_list=LABELS,
        scaling_list=SCALINGS,
        input_shape=(2, 13, 21),
        transpose=(0, 2, 3, 1),
        files_from_end=True,
        shuffle=True,
        tfrecords_dir=str(tfrecords_val),
        use_time_stamps=TIME_STAMPS,
        max_workers=1,
        seed=43,
        quantize=True,
    )

    # Ensure TFRecords were written
    _assert_nonempty_glob(str(tfrecords_train / "*.tfrecord*"))
    _assert_nonempty_glob(str(tfrecords_val / "*.tfrecord*"))

    # Garbage collect writer instances (ensure no handles are open)
    del train_writer, val_writer

    # 3) Reload DGs from TFRecords
    train_gen = OptimizedDataGenerator(
        load_from_tfrecords_dir=str(tfrecords_train),
        max_workers=1,
        seed=44,
        quantize=True,
    )
    val_gen = OptimizedDataGenerator(
        load_from_tfrecords_dir=str(tfrecords_val),
        max_workers=1,
        seed=45,
        quantize=True,
    )

    # 4) Build a tiny model
    model = CreateModel(INPUT_SHAPE_HW_C, n_filters=4, pool_size=3)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=custom_loss)

    # Lightweight callbacks to exercise code paths
    ckpt_path = base_model_dir / "weights.{epoch:02d}-t{loss:.2f}-v{val_loss:.2f}.hdf5"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=1, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            save_weights_only=True,
            monitor="val_loss",
            save_best_only=False,
        ),
    ]

    # 5) Run one quick epoch
    history = model.fit(
        x=train_gen,
        validation_data=val_gen,
        epochs=NUM_EPOCHS,
        shuffle=False,
        verbose=0,
        callbacks=callbacks,
    )
    # Basic sanity assertions
    assert history is not None
    assert (base_model_dir.exists() and any(base_model_dir.iterdir())), "No checkpoint files created"

    # Evaluate — just check it returns a finite float
    val_loss = model.evaluate(val_gen, verbose=0)
    assert isinstance(val_loss, (float, np.floating)) and math.isfinite(float(val_loss))
