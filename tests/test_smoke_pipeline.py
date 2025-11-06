"""
Smoke test for the SmartPixels ML pipeline.
(Refactored with pytest fixtures for clarity and modularity)
"""
from __future__ import annotations
import os
import sys
import glob
import math
from pathlib import Path
from typing import NamedTuple

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
DATA_SET_SIZE = 24  # keep > BATCH_SIZE to exercise batching
BATCH_SIZE = 4
TRAIN_FILE_COUNT = 2
VAL_FILE_COUNT = 2
NUM_EPOCHS = 1

INPUT_SHAPE_HW_C = (13, 21, 2)  # Model expects (H,W,C) = (13,21,2)
TIME_STAMPS = [0, 19]  # Two timestamps -> 2 channels
TIME_LEN = 20  # Matches stamps above
SCALINGS = [75.0, 18.75, 10.0, 1.22]
LABELS = ["x-midplane", "y-midplane", "cotAlpha", "cotBeta"]


# --- Helper Classes for Fixture Payloads ---
class PipelinePaths(NamedTuple):
    """A simple struct to hold all temporary paths for the test."""
    data_dir: Path
    labels_dir: Path
    tfrecords_train: Path
    tfrecords_val: Path
    base_model_dir: Path

class TFRecordPaths(NamedTuple):
    """A simple struct to hold the output of the TFRecord writing step."""
    train_dir: Path
    val_dir: Path


# --- Fixture Definitions ---

@pytest.fixture(scope="module")
def pipeline_paths(tmp_path_factory: pytest.TempPathFactory) -> PipelinePaths:
    """
    Creates all necessary temporary directories for a test run.
    'scope="module"' means this runs only ONCE for all tests in this file.
    """
    test_root = tmp_path_factory.mktemp("smoke_test_root")
    paths = PipelinePaths(
        data_dir=test_root / "data",
        labels_dir=test_root / "labels",
        tfrecords_train=test_root / "tfrecords" / "train",
        tfrecords_val=test_root / "tfrecords" / "validation",
        base_model_dir=test_root / "base_model",
    )
    # Ensure directories exist
    for d in paths:
        d.mkdir(parents=True, exist_ok=True)
    return paths


@pytest.fixture(scope="module")
def generated_parquet_data(pipeline_paths: PipelinePaths) -> Path:
    """
    Depends on 'pipeline_paths'. Generates dummy Parquet files.
    Yields the data directory path.
    """
    _generate_dummy_parquet(
        pipeline_paths.data_dir,
        pipeline_paths.labels_dir,
        n_files=max(TRAIN_FILE_COUNT, VAL_FILE_COUNT)
    )
    _assert_nonempty_glob(str(pipeline_paths.data_dir / "part.*.parquet"))
    return pipeline_paths.data_dir


@pytest.fixture(scope="module")
def written_tfrecords(pipeline_paths: PipelinePaths, generated_parquet_data: Path) -> TFRecordPaths:
    """
    Depends on 'pipeline_paths' and 'generated_parquet_data'.
    Runs the DataGenerator in "writer" mode to create TFRecords.
    """
    # We use 'generated_parquet_data' to ensure data exists, but get paths from 'pipeline_paths'
    data_dir = pipeline_paths.data_dir
    tfrecords_train = pipeline_paths.tfrecords_train
    tfrecords_val = pipeline_paths.tfrecords_val

    # 2) Instantiate DG to WRITE TFRecords
    train_writer = OptimizedDataGenerator(
        dataset_base_dir=str(data_dir),
        batch_size=BATCH_SIZE,
        file_count=TRAIN_FILE_COUNT,
        to_standardize=True,
        labels_list=LABELS,
        input_shape=(2, 13, 21),  # (C, H, W) as per your generator API
        transpose=(0, 2, 3, 1),  # -> (C,H,W) -> (H,W,C) for Keras
        files_from_end=False,
        shuffle=True,
        tfrecords_dir=str(tfrecords_train),
        use_time_stamps=TIME_STAMPS,
        max_workers=1,
        seed=42,
    )

    val_writer = OptimizedDataGenerator(
        dataset_base_dir=str(data_dir),
        batch_size=BATCH_SIZE,
        file_count=VAL_FILE_COUNT,
        to_standardize=True,
        labels_list=LABELS,
        input_shape=(2, 13, 21),
        transpose=(0, 2, 3, 1),
        files_from_end=True,
        shuffle=True,
        tfrecords_dir=str(tfrecords_val),
        use_time_stamps=TIME_STAMPS,
        max_workers=1,
        seed=43,
    )

    # Ensure TFRecords were written
    _assert_nonempty_glob(str(tfrecords_train / "*.tfrecord*"))
    _assert_nonempty_glob(str(tfrecords_val / "*.tfrecord*"))
    
    # Garbage collect writer instances
    del train_writer, val_writer

    return TFRecordPaths(train_dir=tfrecords_train, val_dir=tfrecords_val)


# --- Test Functions (Now much smaller!) ---

def test_data_generation(generated_parquet_data: Path):
    """Tests that the parquet generation fixture ran successfully."""
    assert generated_parquet_data.exists()
    assert any(generated_parquet_data.glob("part.*.parquet"))
    print("Dummy Parquet data generated.") # This won't show unless test fails or -rA is used

def test_tfrecord_writing(written_tfrecords: TFRecordPaths):
    """Tests that the TFRecord writing fixture ran successfully."""
    assert written_tfrecords.train_dir.exists()
    assert written_tfrecords.val_dir.exists()
    assert any(written_tfrecords.train_dir.glob("*.tfrecord*"))
    assert any(written_tfrecords.val_dir.glob("*.tfrecord*"))
    print("TFRecords written successfully.")

@pytest.mark.timeout(300)
def test_model_training_and_evaluation(written_tfrecords: TFRecordPaths, pipeline_paths: PipelinePaths):
    """
    The final step:
    1) Reloads DGs from the TFRecords created by the fixture.
    2) Builds a tiny model.
    3) Runs 1 epoch and evaluates.
    """
    # 3) Reload DGs from TFRecords
    train_gen = OptimizedDataGenerator(
        load_from_tfrecords_dir=str(written_tfrecords.train_dir),
        max_workers=1,
        seed=44,
        quantize=True,
    )
    val_gen = OptimizedDataGenerator(
        load_from_tfrecords_dir=str(written_tfrecords.val_dir),
        max_workers=1,
        seed=45,
        quantize=True,
    )

    # 4) Build a tiny model
    model = CreateModel(INPUT_SHAPE_HW_C, n_filters=4, pool_size=3)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=custom_loss)

    # Lightweight callbacks to exercise code paths
    ckpt_path = pipeline_paths.base_model_dir / "weights.{epoch:02d}.hdf5"
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
        shuffle=False,  # Shuffling is done in on_epoch_end / __getitem__
        verbose=0,
        callbacks=callbacks,
    )
    
    # Basic sanity assertions
    assert history is not None
    model_dir = pipeline_paths.base_model_dir
    assert (model_dir.exists() and any(model_dir.glob("*.hdf5"))), "No checkpoint files created"

    # Evaluate — just check it returns a finite float
    val_loss = model.evaluate(val_gen, verbose=0)
    assert isinstance(val_loss, (float, np.floating)) and math.isfinite(float(val_loss))
    print(f"Model trained and evaluated with final val_loss: {val_loss:.4f}")


# --- Helper Functions (Unchanged) ---

def _generate_dummy_parquet(data_dir: Path, labels_dir: Path, n_files: int) -> None:
    """
    Generate tiny dummy Parquet files the DG will actually find:
    - Filenames: part.00000.parquet, part.00001.parquet, ...
    - Columns: recon columns ("0".."H*W*T-1") + LABELS + event_id
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_files):
        rng = np.random.default_rng(1234 + i)

        # Recon volume [N, H, W, T] -> flatten to columns "0".. for parquet
        sample = rng.random(
            (DATA_SET_SIZE, INPUT_SHAPE_HW_C[0], INPUT_SHAPE_HW_C[1], TIME_LEN),
            dtype=np.float32,
        )
        flat = sample.reshape(DATA_SET_SIZE, -1)
        cols = [str(j) for j in range(flat.shape[1])]
        df_data = pd.DataFrame(flat, columns=cols)
        df_data["event_id"] = np.arange(DATA_SET_SIZE)

        # Add labels into the SAME parquet file
        labels = rng.random((DATA_SET_SIZE, len(LABELS)), dtype=np.float32)
        for j, name in enumerate(LABELS):
            df_data[name] = labels[:, j]

        # File pattern the generator globs for
        df_data.to_parquet(data_dir / f"part.{i:05d}.parquet", index=False)

        # Optional: keep a separate labels file for debugging
        df_lbl = pd.DataFrame(labels, columns=LABELS)
        df_lbl["event_id"] = np.arange(DATA_SET_SIZE)
        df_lbl.to_parquet(labels_dir / f"labels_data_{i}.parquet", index=False)


def _assert_nonempty_glob(pattern: str) -> list[str]:
    files = glob.glob(pattern)
    assert files, f"No files matched pattern: {pattern}"
    return files