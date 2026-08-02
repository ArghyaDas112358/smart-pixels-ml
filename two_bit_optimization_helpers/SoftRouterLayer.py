# -*- coding: utf-8 -*-
# SoftRouterLayer.py
#
# Learnable time-slice selector ("soft router") — the sibling of SoftQuantizeLayer.
# SoftQuantize learns the ADC thresholds on the VALUE axis; this layer learns WHICH
# time slices to read out on the TIME axis. Both are offline bootstrapping tools:
# only their learned numbers (thresholds / slice indices) cross to the ASIC.
#
# Design (docs/soft_router_plan.md, Fig B):
#   - weights: slot_logits W (num_slots, num_slices) + log_k (annealed by the same
#     cosine AnnealingScheduler as SoftQuantize; k = exp(log_k), 1 -> 67).
#   - each slot s: a_s = softmax(k * (W_s + mask_s)) over the slice axis, where
#     mask_s = sum_{j<s} log(1 - a_j + eps)  (successive removal -> later slots
#     cannot pick earlier slots' slices; duplicate-free BY CONSTRUCTION, no penalty).
#   - soft path:  y_soft[..., s] = sum_t a_s[t] * x[..., t]     (gradients)
#   - hard path:  y_hard[..., s] = x[..., argmax a_s]           (forward value)
#   - STE:        y = stop_gradient(y_hard - y_soft) + y_soft   (= SoftQuantizeLayer.py:186)
#   The forward pass is ALWAYS the hard num_slots-slice pick, so the training loss
#   is an honest 2-slice loss; the all-slice soft blend only steers gradients.
#   Cardinality is STRUCTURAL: there are exactly num_slots output channels.

import math

import numpy as np
import tensorflow as tf


class SoftRouterLayer(tf.keras.layers.Layer):
    """Selects `num_slots` time slices out of the input's last axis, learnably.

    Input  (B, H, W, T)  ->  Output (B, H, W, num_slots)

    After training, `selected_indices()` returns the committed slice indices
    (the ASIC readout configuration); the layer is then discarded, exactly like
    SoftQuantizeLayer after it donates its thresholds to the ADC.
    """

    def __init__(self,
                 num_slots: int = 2,
                 initial_k: float = 1.0,
                 trainable_k: bool = False,
                 logits_init_stddev: float = 0.5,
                 mask_eps: float = 1e-6,
                 seed=None,
                 **kwargs):
        super(SoftRouterLayer, self).__init__(**kwargs)
        assert isinstance(num_slots, int) and num_slots >= 1, "'num_slots' must be a positive integer."
        self.num_slots = num_slots
        self.initial_k = initial_k
        self.trainable_k = trainable_k
        self.logits_init_stddev = logits_init_stddev
        self.mask_eps = mask_eps
        self.seed = seed

    def build(self, input_shape):
        self.num_slices = int(input_shape[-1])
        assert self.num_slices >= self.num_slots, (
            f"input has {self.num_slices} slices but num_slots={self.num_slots}")

        self.slot_logits = self.add_weight(
            name='slot_logits',
            shape=(self.num_slots, self.num_slices),
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=self.logits_init_stddev, seed=self.seed),
            trainable=True,
        )
        # same knob/name as SoftQuantizeLayer so one AnnealingScheduler class drives both
        self.log_k = self.add_weight(
            name='log_k',
            shape=(1,),
            initializer=tf.constant_initializer(math.log(self.initial_k)),
            trainable=self.trainable_k,
        )
        super(SoftRouterLayer, self).build(input_shape)

    @property
    def k(self):
        return tf.exp(self.log_k)

    def slot_weights(self):
        """Masked softmax weights [a_1, ..., a_S], each (num_slices,).

        Successive removal: slot s sees earlier slots' picks suppressed via
        log(1 - a_j + eps), so as k anneals hard the slots commit to DISTINCT slices.
        """
        k = self.k
        weights = []
        mask = tf.zeros((self.num_slices,), dtype=tf.float32)
        for s in range(self.num_slots):
            logits = self.slot_logits[s] + mask
            a = tf.nn.softmax(k * logits)
            weights.append(a)
            mask = mask + tf.math.log(1.0 - a + self.mask_eps)
        return weights

    def hard_indices(self, weights=None):
        """Distinct argmax picks (list of scalar int32 tensors), one per slot.

        Earlier picks are explicitly excluded before each argmax, so the hard
        indices are distinct at EVERY k (the soft mask alone only guarantees
        distinctness once the softmaxes have sharpened).
        """
        if weights is None:
            weights = self.slot_weights()
        picked = []
        for a in weights:
            aa = a
            for ip in picked:
                aa = tf.tensor_scatter_nd_update(
                    aa,
                    tf.reshape(tf.cast(ip, tf.int32), (1, 1)),
                    tf.constant([-1.0], dtype=aa.dtype),  # a in [0,1] -> -1 never wins
                )
            picked.append(tf.argmax(aa, output_type=tf.int32))
        return picked

    def call(self, inputs, training=None):
        weights = self.slot_weights()

        # soft: convex blend over all slices (gradient path)
        y_soft = tf.stack(
            [tf.tensordot(inputs, a, axes=[[-1], [0]]) for a in weights], axis=-1)

        # hard: the actual slice pick (forward value)
        idx = self.hard_indices(weights)
        y_hard = tf.stack([tf.gather(inputs, i, axis=-1) for i in idx], axis=-1)

        if training:
            return tf.stop_gradient(y_hard - y_soft) + y_soft
        return tf.stop_gradient(y_hard)

    # ---- extraction / logging helpers ------------------------------------
    def selected_indices(self):
        """The committed slice indices as plain ints — the ASIC readout config."""
        return [int(i.numpy()) for i in self.hard_indices()]

    def slot_weights_numpy(self):
        """Current masked slot weights as a (num_slots, num_slices) float array."""
        return np.stack([a.numpy() for a in self.slot_weights()])

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.num_slots,)

    def get_config(self):
        config = super(SoftRouterLayer, self).get_config()
        config.update({
            'num_slots': self.num_slots,
            'initial_k': self.initial_k,
            'trainable_k': self.trainable_k,
            'logits_init_stddev': self.logits_init_stddev,
            'mask_eps': self.mask_eps,
            'seed': self.seed,
        })
        return config
