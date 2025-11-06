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
from typing import NamedTuple, Any

import numpy as np
import pandas as pd
import pytest
from rich.console import Console 
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

# --- Test Parameters (Unchanged) ---
DATA_SET_SIZE = 24
BATCH_SIZE = 4
TRAIN_FILE_COUNT = 2
VAL_FILE_COUNT = 2
NUM_EPOCHS = 1
INPUT_SHAPE_HW_C = (13, 21, 2)
TIME_STAMPS = [0, 19]
TIME_LEN = 20
SCALINGS = [75.0, 18.75, 10.0, 1.22]
LABELS = ["x-midplane", "y-midplane", "cotAlpha", "cotBeta"]
console = Console() # Global console for printing


# --- Helper Classes for Fixture Payloads ---
class PipelinePaths(NamedTuple):
    """Holds all temporary paths for the test."""
    data_dir: Path
    labels_dir: Path
    tfrecords_train: Path
    tfrecords_val: Path
    base_model_dir: Path

class TFRecordPaths(NamedTuple):
    """Holds the output of the TFRecord writing step."""
    train_dir: Path
    val_dir: Path

class LoadedGenerators(NamedTuple):
    """Holds the train and validation data generators."""
    train_gen: OptimizedDataGenerator
    val_gen: OptimizedDataGenerator

class TrainedModel(NamedTuple):
    """Holds the trained model and its history."""
    model: tf.keras.Model
    history: tf.keras.callbacks.History


# --- Fixture Definitions ---

@pytest.fixture(scope="module")
def pipeline_paths(tmp_path_factory: pytest.TempPathFactory) -> PipelinePaths:
    """[Stage 1] Creates all necessary temporary directories."""
    console.print("\n[bold yellow]🔧 1. Setting up pipeline directories...[/bold yellow]")
    test_root = tmp_path_factory.mktemp("smoke_test_root")
    paths = PipelinePaths(
        data_dir=test_root / "data",
        labels_dir=test_root / "labels",
        tfrecords_train=test_root / "tfrecords" / "train",
        tfrecords_val=test_root / "tfrecords" / "validation",
        base_model_dir=test_root / "base_model",
    )
    for d in paths:
        d.mkdir(parents=True, exist_ok=True)
    return paths


@pytest.fixture(scope="module")
def generated_parquet_data(pipeline_paths: PipelinePaths) -> Path:
    """[Stage 2] Generates dummy Parquet files."""
    console.print("[bold yellow]🏭 2. Generating dummy Parquet data...[/bold yellow]")
    _generate_dummy_parquet(
        pipeline_paths.data_dir,
        pipeline_paths.labels_dir,
        n_files=max(TRAIN_FILE_COUNT, VAL_FILE_COUNT)
    )
    _assert_nonempty_glob(str(pipeline_paths.data_dir / "part.*.parquet"))
    console.print("[green]   ... Parquet generation complete.[/green]")
    return pipeline_paths.data_dir


@pytest.fixture(scope="module")
def written_tfrecords(pipeline_paths: PipelinePaths, generated_parquet_data: Path) -> TFRecordPaths:
    """[Stage 3] Runs the DataGenerator in 'writer' mode."""
    # We use 'generated_parquet_data' to ensure data exists
    console.print("[bold yellow]✍️  3. Writing TFRecords (this will show progress)...[/bold yellow]")
    data_dir = pipeline_paths.data_dir
    tfrecords_train = pipeline_paths.tfrecords_train
    tfrecords_val = pipeline_paths.tfrecords_val

    # Run writers
    train_writer = OptimizedDataGenerator(
        dataset_base_dir=str(data_dir), batch_size=BATCH_SIZE, file_count=TRAIN_FILE_COUNT,
        to_standardize=True, labels_list=LABELS, input_shape=(2, 13, 21),
        transpose=(0, 2, 3, 1), files_from_end=False, shuffle=True,
        tfrecords_dir=str(tfrecords_train), use_time_stamps=TIME_STAMPS, max_workers=1, seed=42,
    )
    val_writer = OptimizedDataGenerator(
        dataset_base_dir=str(data_dir), batch_size=BATCH_SIZE, file_count=VAL_FILE_COUNT,
        to_standardize=True, labels_list=LABELS, input_shape=(2, 13, 21),
        transpose=(0, 2, 3, 1), files_from_end=True, shuffle=True,
        tfrecords_dir=str(tfrecords_val), use_time_stamps=TIME_STAMPS, max_workers=1, seed=43,
    )
    
    del train_writer, val_writer # Clean up
    console.print("[green]   ... TFRecord writing complete.[/green]")
    return TFRecordPaths(train_dir=tfrecords_train, val_dir=tfrecords_val)


@pytest.fixture(scope="module")
def loaded_generators(written_tfrecords: TFRecordPaths) -> LoadedGenerators:
    """[Stage 4] Reloads DGs from the written TFRecords."""
    console.print("[bold yellow]🧠 4. Loading data from TFRecords...[/bold yellow]")
    train_gen = OptimizedDataGenerator(
        load_from_tfrecords_dir=str(written_tfrecords.train_dir),
        max_workers=1, seed=44, quantize=True,
    )
    val_gen = OptimizedDataGenerator(
        load_from_tfrecords_dir=str(written_tfrecords.val_dir),
        max_workers=1, seed=45, quantize=True,
    )
    console.print("[green]   ... Data generators ready.[/green]")
    return LoadedGenerators(train_gen=train_gen, val_gen=val_gen)


@pytest.fixture(scope="module")
def compiled_model() -> tf.keras.Model:
    """[Stage 5] Builds and compiles the model."""
    console.print("[bold yellow]🛠️  5. Building and compiling model...[/bold yellow]")
    model = CreateModel(INPUT_SHAPE_HW_C, n_filters=4, pool_size=3)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=custom_loss)
    console.print("[green]   ... Model compiled.[/green]")
    return model


@pytest.fixture(scope="module")
def trained_model(
    compiled_model: tf.keras.Model,
    loaded_generators: LoadedGenerators,
    pipeline_paths: PipelinePaths
) -> TrainedModel:
    """[Stage 6] Runs model.fit()."""
    console.print("[bold yellow]🏃 6. Starting model training (1 epoch)...[/bold yellow]")
    model = compiled_model # Use the model from the previous stage
    
    ckpt_path = pipeline_paths.base_model_dir / "weights.{epoch:02d}.hdf5"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=1, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path), save_weights_only=True,
            monitor="val_loss", save_best_only=False,
        ),
    ]

    history = model.fit(
        x=loaded_generators.train_gen,
        validation_data=loaded_generators.val_gen,
        epochs=NUM_EPOCHS,
        shuffle=False,  # Shuffling is done in DG
        verbose=1,
        callbacks=callbacks,
    )
    console.print("[green]   ... Training complete.[/green]")
    return TrainedModel(model=model, history=history)


@pytest.fixture(scope="module")
def evaluation_result(
    trained_model: TrainedModel,
    loaded_generators: LoadedGenerators
) -> float:
    """[Stage 7] Runs model.evaluate()."""
    console.print("[bold yellow]📊 7. Evaluating model...[/bold yellow]")
    val_loss = trained_model.model.evaluate(loaded_generators.val_gen, verbose=1)
    console.print(f"[green]   ... Evaluation complete. Final Val Loss: {val_loss:.4f}[/green]")
    return val_loss


# --- Test Functions (Now granular and clean) ---

def test_gpu_availability():
    console.print("[bold yellow]🔍 0. Checking for GPU...[/bold yellow]")
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        pytest.skip("No CUDA-enabled GPU found by TensorFlow.")
    console.print(f"[green]   ... Found {len(gpus)} GPU(s): {gpus[0].name}[/green]")
    assert True

def test_data_generation(generated_parquet_data: Path):
    """Verifies [Stage 2] ran and produced files."""
    assert generated_parquet_data.exists()
    assert any(generated_parquet_data.glob("part.*.parquet"))

def test_tfrecord_writing(written_tfrecords: TFRecordPaths):
    """Verifies [Stage 3] ran and produced files."""
    assert written_tfrecords.train_dir.exists()
    assert any(written_tfrecords.train_dir.glob("*.tfrecord*"))

def test_data_loading(loaded_generators: LoadedGenerators):
    """Verifies [Stage 4] ran and generators are configured."""
    assert loaded_generators.train_gen is not None
    assert loaded_generators.val_gen is not None
    assert len(loaded_generators.train_gen) > 0
    assert len(loaded_generators.val_gen) > 0

def test_model_compilation(compiled_model: tf.keras.Model):
    """Verifies [Stage 5] ran and the model is compiled."""
    assert compiled_model.optimizer is not None
    assert compiled_model.loss is not None

def test_model_training(trained_model: TrainedModel, pipeline_paths: PipelinePaths):
    """Verifies [Stage 6] ran, history was created, and checkpoints were saved."""
    assert trained_model.history is not None
    assert "val_loss" in trained_model.history.history
    
    model_dir = pipeline_paths.base_model_dir
    assert (model_dir.exists() and any(model_dir.glob("*.hdf5"))), "No checkpoint files created"

def test_model_evaluation(evaluation_result: float):
    """Verifies [Stage 7] ran and the loss is a valid number."""
    assert isinstance(evaluation_result, (float, np.floating))
    assert math.isfinite(float(evaluation_result))


# --- Helper Functions (Unchanged) ---
# ... (all helper code from _generate_dummy_parquet down) ...
def _generate_dummy_parquet(data_dir: Path, labels_dir: Path, n_files: int) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        rng = np.random.default_rng(1234 + i)
        sample = rng.random(
            (DATA_SET_SIZE, INPUT_SHAPE_HW_C[0], INPUT_SHAPE_HW_C[1], TIME_LEN),
            dtype=np.float32,
        )
        flat = sample.reshape(DATA_SET_SIZE, -1)
        cols = [str(j) for j in range(flat.shape[1])]
        df_data = pd.DataFrame(flat, columns=cols)
        df_data["event_id"] = np.arange(DATA_SET_SIZE)
        labels = rng.random((DATA_SET_SIZE, len(LABELS)), dtype=np.float32)
        for j, name in enumerate(LABELS):
            df_data[name] = labels[:, j]
        df_data.to_parquet(data_dir / f"part.{i:05d}.parquet", index=False)
        df_lbl = pd.DataFrame(labels, columns=LABELS)
        df_lbl["event_id"] = np.arange(DATA_SET_SIZE)
        df_lbl.to_parquet(labels_dir / f"labels_data_{i}.parquet", index=False)

def _assert_nonempty_glob(pattern: str) -> list[str]:
    files = glob.glob(pattern)
    assert files, f"No files matched pattern: {pattern}"
    return files