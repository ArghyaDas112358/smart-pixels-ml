import tensorflow as tf
import glob
import numpy as np
import pandas as pd
import os
import math
import logging
import gc

# Custom quantizer
def qkeras_data_prep_quantizer(data, bits=4, int_bits=0, alpha=1):
    from qkeras import quantized_bits
    quantizer = quantized_bits(bits, int_bits, alpha=alpha)
    return quantizer(data)

class TFDataGenerator:
    def __init__(
        self,
        data_directory_path="./",
        labels_directory_path="./",
        is_directory_recursive=False,
        file_type="csv",
        data_format="2D",
        batch_size=32,
        file_count=None,
        labels_list="cotAlpha",
        to_standardize=False,
        input_shape=(13, 21),
        transpose=None,
        include_y_local=False,
        files_from_end=False,
        shuffle=False,
        current=False,
        sample_delta_t=200,
        load_from_tfrecords_dir=None,
        tfrecords_dir=None,
        use_time_stamps=-1,
        seed=None,
        quantize=False,
        max_workers=1,
        **kwargs,
    ):
        """
        Initializes the data generator.
        """
        self.batch_size = batch_size
        self.labels_list = labels_list
        self.input_shape = input_shape
        self.transpose = transpose
        self.to_standardize = to_standardize
        self.include_y_local = include_y_local
        self.quantize = quantize
        self.shuffle = shuffle
        self.seed = seed if seed is not None else 42
        self.rng = np.random.default_rng(self.seed)

        # Decide which time stamps to load
        len_xy, ntime = 13 * 21, 20
        if use_time_stamps == -1:
            self.use_time_stamps = np.arange(ntime)
        else:
            self.use_time_stamps = np.array(use_time_stamps)
        idx = [
            [i * len_xy, (i + 1) * len_xy] for i in range(ntime)
        ]  # 20 time stamps of length 13*21
        self.use_columns = []
        for i in self.use_time_stamps:
            self.use_columns.extend(
                [str(j) for j in range(idx[i][0], idx[i][1])]
            )
        if file_type not in ["csv", "parquet"]:
            raise ValueError('file_type can only be "csv" or "parquet"!')
        self.file_type = file_type

        # Collect file paths
        self.recon_files = glob.glob(
            os.path.join(
                data_directory_path,
                f"recon{data_format}*.{file_type}",
            ),
            recursive=is_directory_recursive,
        )
        self.recon_files.sort()
        if file_count is not None:
            if not files_from_end:
                self.recon_files = self.recon_files[:file_count]
            else:
                self.recon_files = self.recon_files[-file_count:]

        self.label_files = [
            os.path.join(
                labels_directory_path,
                os.path.basename(f).replace(
                    f"recon{data_format}", "labels"
                ),
            )
            for f in self.recon_files
        ]

        if load_from_tfrecords_dir is not None:
            if not os.path.isdir(load_from_tfrecords_dir):
                raise ValueError(
                    f"Directory {load_from_tfrecords_dir} does not exist."
                )
            self.tfrecords_dir = load_from_tfrecords_dir
        else:
            self.tfrecords_dir = (
                tfrecords_dir
                if tfrecords_dir is not None
                else "./tfrecords"
            )
            os.makedirs(self.tfrecords_dir, exist_ok=True)
            self.create_tfrecords()

        # Build the dataset
        self.dataset = self.build_dataset()

    def create_tfrecords(self):
        """
        Converts data into TFRecords for efficient loading.
        """
        for recon_file, label_file in zip(
            self.recon_files, self.label_files
        ):
            tfrecord_file = os.path.join(
                self.tfrecords_dir,
                os.path.basename(recon_file).replace(
                    f".{self.file_type}", ".tfrecord"
                ),
            )
            if os.path.exists(tfrecord_file):
                continue  # Skip if already processed
            self.process_and_save_tfrecord(
                recon_file, label_file, tfrecord_file
            )

    def process_and_save_tfrecord(
        self, recon_file, label_file, tfrecord_file
    ):
        """
        Processes data files and saves them as TFRecords.
        """
        if self.file_type == "csv":
            recon_df = pd.read_csv(recon_file, usecols=self.use_columns)
            labels_df = pd.read_csv(label_file, usecols=[self.labels_list])
        elif self.file_type == "parquet":
            recon_df = pd.read_parquet(
                recon_file, columns=self.use_columns
            )
            labels_df = pd.read_parquet(
                label_file, columns=[self.labels_list]
            )
        else:
            raise ValueError("Unsupported file type.")

        # Drop NaNs
        has_nans = recon_df.isna().any(axis=1)
        recon_df = recon_df[~has_nans]
        labels_df = labels_df[~has_nans]

        # Convert to numpy
        X = recon_df.values
        y = labels_df.values.astype(np.float32)

        # Apply log transformation
        nonzeros = np.abs(X) > 0
        X[nonzeros] = (
            np.sign(X[nonzeros])
            * np.log1p(np.abs(X[nonzeros]))
            / math.log(2)
        )

        # Standardization
        if self.to_standardize:
            mean = X[nonzeros].mean()
            std = X[nonzeros].std() + 1e-10
            X[nonzeros] = (X[nonzeros] - mean) / std

        # Reshape
        X = X.reshape((-1, *self.input_shape))
        if self.transpose is not None:
            X = X.transpose(self.transpose)

        # Normalize labels
        y = y / np.array([75.0, 18.75, 8.0, 0.5])

        # Save to TFRecord
        with tf.io.TFRecordWriter(tfrecord_file) as writer:
            for xi, yi in zip(X, y):
                example = self.serialize_example(xi, yi)
                writer.write(example)

        # Clean up
        del recon_df, labels_df, X, y
        gc.collect()

    @staticmethod
    def serialize_example(X, y):
        """
        Creates a serialized tf.train.Example from input data.
        """
        feature = {
            "X": tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[tf.io.serialize_tensor(X).numpy()]
                )
            ),
            "y": tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[tf.io.serialize_tensor(y).numpy()]
                )
            ),
        }
        example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature)
        )
        return example_proto.SerializeToString()

    def build_dataset(self):
        """
        Builds a tf.data.Dataset for efficient data loading.
        """
        tfrecord_files = tf.io.gfile.glob(
            os.path.join(self.tfrecords_dir, "*.tfrecord")
        )
        if self.shuffle:
            self.rng.shuffle(tfrecord_files)

        dataset = tf.data.TFRecordDataset(
            tfrecord_files,
            num_parallel_reads=tf.data.AUTOTUNE,
        )

        # Parse the serialized data
        dataset = dataset.map(
            self._parse_function, num_parallel_calls=tf.data.AUTOTUNE
        )

        # Shuffle, batch, and prefetch
        if self.shuffle:
            dataset = dataset.shuffle(
                buffer_size=10000, seed=self.seed, reshuffle_each_iteration=True
            )
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def _parse_function(self, example_proto):
        """
        Parses a serialized tf.train.Example into tensors.
        """
        feature_description = {
            "X": tf.io.FixedLenFeature([], tf.string),
            "y": tf.io.FixedLenFeature([], tf.string),
        }
        parsed_example = tf.io.parse_single_example(
            example_proto, feature_description
        )
        X = tf.io.parse_tensor(parsed_example["X"], out_type=tf.float32)
        y = tf.io.parse_tensor(parsed_example["y"], out_type=tf.float32)

        if self.quantize:
            X = qkeras_data_prep_quantizer(X, bits=4, int_bits=0, alpha=1)

        return X, y

    def get_dataset(self):
        """
        Returns the prepared dataset.
        """
        return self.dataset

