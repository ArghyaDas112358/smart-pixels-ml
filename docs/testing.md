# Testing Module 

This module contains unit tests for the Smart Pixels ML project.

## Scripts

## `test_run().py`
- **Description**: 
    - This tests the entire pipeline of the Smart Pixels ML project. It loads the dataset, preprocesses it, trains the model, saves the model, and evaluates the model. 

- **Usage**:
    ```python
    python test_run()
    ```

- **Example**:
    ![test_run](Images/test_run.gif)

### Functions
1. `check_directories()`
    - **Description**:
        Checks if the essential directories (`TEST_ROOT`, `DATA_DIR`, `LABELS_DIR`, `TFRECORDS_DIR`, etc.) exist. Logs an error if any directory is missing.
    - **Returns**:
    `True` if all required directories exist; otherwise, logs an error and returns False.

2. `generate_dummy_data(num_files=NUM_DUMMY_FILES)`
    - **Description**:
    Generates dummy Parquet files for testing, including random input data and labels.
    - **Parameters**:
    `num_files` (int): The number of dummy data files to generate.
    - **Outputs**:
    Creates Parquet files in `DATA_DIR` and `LABELS_DIR`.

3. `generate_tfrecords()`
    - **Description**:
        Initializes data generators and generates TFRecords for training and validation datasets.
    - **Outputs**:
        TFRecord files saved in `TFRECORDS_DIR_TRAIN` and `TFRECORDS_DIR_VALIDATION`.

4. `load_tfrecords()`:
    - **Description**:
        Loads pre-generated TFRecords for training and validation datasets.
    - **Returns**:
        `training_generator` and `validation_generator`.

5. `test_model_generation()`
    - **Description**:
        Builds and compiles a test model using the CreateModel function.
    - **Returns**: 
        A compiled Keras model.
    - **Raises**:
        Logs an error and exits if the model cannot be built.

6. `test_train_model()`
    - **Description**:
        Tests model training by generating dummy data, creating TFRecords, and training the model.
    - **Outputs**:
        Logs training progress, final validation loss, and saves model weights during training.

7. `run_smoke_test()`:
    - **Description**:
        Runs the entire smoke test, including data generation, model training, and evaluation.
    - **Outputs**:
        Logs the final evaluation metrics and saves the model.
---
## `some_test.py`
### Functions
...

---

## TODO 📝
 - 🔄 Add tests for:
    - Loading models from various formats (.hdf5, .h5, .pb, etc.).
    - Plotting training/evaluation results.
    - Evaluating metrics like accuracy, loss, etc.
 - 🚀 Implement benchmarks tests for:
    - Model training and evaluation speed.
    - Memory usage.
    - GPU utilization.
