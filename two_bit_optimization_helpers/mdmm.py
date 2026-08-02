# -*- coding: utf-8 -*-
# mdmm.py
#
# MDMM (Modified Differential Method of Multipliers, Platt & Barr 1988) for
# output constraints -- Keras 2.15 / TF 2.15 port of Harshul's Keras 3
# implementation (github.com/guptaharshul24/smart-pixels-ml, models/mdmm.py),
# itself adapted from das214's original TF code (mdmm_tf_mnist). The penalty
# math, the second deterministic forward pass, and the sign-flip multiplier
# ascent are kept byte-for-byte in spirit; only the Keras-3-isms are removed:
#   - Model.compute_loss(x=, y=, y_pred=) exists in TF 2.15 -- kept as is.
#   - id()-based lambda identification works on both Keras generations -- kept.
#   - add_weight(name=, shape=, ...) keyword form works in Keras 2 -- kept.
#
# What it does: wraps any model; training loss becomes
#     loss = task_loss + sum_c  scale_c * ( max(lambda_c, 0) * inf_c
#                                           + damping_c * inf_c^2 / 2 )
# with inf_c = max(min_value - metric_c, 0) (hinge inequality). The gradient
# sign is FLIPPED for the lambda variables (ascent), so each multiplier grows
# while its constraint is violated and stops moving once satisfied. val_loss
# stays the plain compiled loss (test_step untouched) -> directly comparable
# to non-MDMM runs.
#
# Two constraint-evaluation modes (constraint_pass):
#   'deterministic' (Harshul's original): a SECOND training=False forward pass.
#     Right for SPREAD constraints (MinStd/MinMad) -- dropout noise inflates
#     output spread by ~15-35% and would silently satisfy them.
#     WARNING for models whose selection layers sit above SoftQuantizeLayer:
#     the quantizer's eval branch is `tf.stop_gradient(hard_q)` -- a hard
#     gradient wall -- so in this mode the constraint gradient NEVER reaches
#     anything upstream of the quantizer (observed 2026-07-28: SimpleRouter
#     theta exactly frozen for 250+ epochs). It also measures the constraint
#     on the deterministic top-2 view while weights train on sampled pairs,
#     and the dropout-free train-batch corr is satisfiable by memorization
#     (train corr >= 0.5, val corr 0.17 -> lambdas freeze, total stall).
#   'primary': penalties computed directly on the training=True y_pred of the
#     main pass. Right for TRUTH-AWARE constraints (MinCorrConstraint): dropout
#     DEcorrelates, so it can only make the floor harder, never gameable; the
#     gradient flows through the quantizer's STE and the router's sampled pair
#     (the exact SIMPLE gradient of E[penalty]); the measured view is the
#     trained view; and the second forward pass (memory + time) is saved.
#
# Constraint-metric history (upstream campaigns, do not re-derive):
#   MinStdConstraint  -- GAMED: 99.7% collapsed bulk + 0.3% extreme outliers
#                        inflate batch std past target (run 438bcf1c).
#   MinMadConstraint  -- GAMED: dispersion satisfied with ~ZERO correlation to
#                        truth (runs eabfe9f3, 4c28f1e8).
#   MinCorrConstraint -- truth-aware Pearson floor; forbids every observed
#                        cheat (constant, outlier-salted, spread-uncorrelated).
#                        This is the one to use.
# Scale must be sized against the loss magnitude (~1e4 for our NLL ~ -3e4):
# at scale 1 the Nadam-capped lambda ascent (~lr/step) never reaches useful
# pressure within 5000 epochs.
#
# The lambdas are NOT saved in checkpoints (save/load delegate to the inner
# model); a mid-run resume restarts them at 0 and they re-grow wherever the
# constraint is still violated.

import tensorflow as tf

keras = tf.keras
layers = tf.keras.layers


class OutputConstraint(layers.Layer):
    """Base class: a constraint on the model output with its own multiplier."""
    def __init__(self, scale=1.0, damping=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.damping = damping
        self.lmbda = self.add_weight(
            name=self.name + '_lmbda',
            shape=(),
            initializer='zeros',
            trainable=True,
        )

    def fn(self, outputs):
        raise NotImplementedError

    def infeasibility(self, fn_value):
        raise NotImplementedError

    def call(self, outputs):
        inf = self.infeasibility(self.fn(outputs))
        l_term = tf.math.maximum(self.lmbda, 0.0) * inf
        damp_term = self.damping * tf.square(inf) / 2
        return self.scale * (l_term + damp_term)


class MinStdConstraint(OutputConstraint):
    """std(outputs[:, column]) >= min_value. DEPRECATED for anti-collapse use:
    std is quadratically outlier-sensitive and was gamed upstream (constant
    bulk + 0.3% extreme outliers). Kept for record/comparison only."""
    def __init__(self, column, min_value, scale=1.0, damping=1.0, **kwargs):
        super().__init__(scale=scale, damping=damping, **kwargs)
        self.column = column
        self.min_value = min_value

    def fn(self, outputs):
        return tf.math.reduce_std(outputs[:, self.column])

    def infeasibility(self, fn_value):
        return tf.math.maximum(self.min_value - fn_value, 0.0)


class MinMadConstraint(OutputConstraint):
    """mean(|outputs[:, column] - mean|) >= min_value. DEPRECATED for
    anti-collapse use: forces dispersion but was satisfied upstream with zero
    truth correlation. Kept for record/comparison only."""
    def __init__(self, column, min_value, scale=1.0, damping=1.0, **kwargs):
        super().__init__(scale=scale, damping=damping, **kwargs)
        self.column = column
        self.min_value = min_value

    def fn(self, outputs):
        col = outputs[:, self.column]
        return tf.reduce_mean(tf.abs(col - tf.reduce_mean(col)))

    def infeasibility(self, fn_value):
        return tf.math.maximum(self.min_value - fn_value, 0.0)


class MinCorrConstraint(OutputConstraint):
    """Pearson corr(outputs[:, column], y_true[:, label_column]) >= min_value.

    Truth-aware: forbids ALL lazy strategies at once (constant, outlier-salted,
    and spread-but-uncorrelated predictions), since only genuine dependence on
    the true value raises the correlation. Requires the MDMM wrapper to pass
    y_true (needs_truth=True).
    """
    needs_truth = True

    def __init__(self, column, label_column, min_value, scale=1.0, damping=1.0, **kwargs):
        super().__init__(scale=scale, damping=damping, **kwargs)
        self.column = column
        self.label_column = label_column
        self.min_value = min_value

    def fn(self, outputs, y_true=None):
        p = outputs[:, self.column]
        t = tf.cast(y_true[:, self.label_column], p.dtype)
        p_c = p - tf.reduce_mean(p)
        t_c = t - tf.reduce_mean(t)
        cov = tf.reduce_mean(p_c * t_c)
        denom = tf.math.reduce_std(p) * tf.math.reduce_std(t) + 1e-6
        return cov / denom

    def infeasibility(self, fn_value):
        return tf.math.maximum(self.min_value - fn_value, 0.0)

    def call(self, outputs, y_true=None):
        inf = self.infeasibility(self.fn(outputs, y_true=y_true))
        l_term = tf.math.maximum(self.lmbda, 0.0) * inf
        damp_term = self.damping * tf.square(inf) / 2
        return self.scale * (l_term + damp_term)


class MDMM(keras.Model):
    """Wraps a model; adds constraint penalties to the training loss.

    train_step: loss = compute_loss + sum(constraint penalties); the gradient
    sign is flipped for the lambda variables (ascent) so each multiplier grows
    while its constraint is violated and stops moving once satisfied.
    val_loss stays the plain compiled loss (test_step is untouched), so
    checkpoint filenames and best_val_loss remain comparable to non-MDMM runs.
    """
    def __init__(self, model, constraints, constraint_samples=None,
                 constraint_pass='deterministic', name='MDMM', **kwargs):
        super().__init__(name=name, **kwargs)
        assert constraint_pass in ('deterministic', 'primary'), constraint_pass
        self.model = model
        self.constraints_list = list(constraints)
        self._lmbda_ids = {id(c.lmbda) for c in self.constraints_list}
        # Evaluate constraints on only the first N samples of each batch if
        # the second forward pass doesn't fit in GPU memory alongside the
        # primary training pass (a spread/correlation estimate only needs
        # ~3-6% accuracy at N~128-512). None = full batch (exact estimate).
        self.constraint_samples = constraint_samples
        # See module docstring: 'deterministic' for spread constraints,
        # 'primary' for truth-aware constraints / models with selection
        # layers above SoftQuantizeLayer (eval-mode gradient wall).
        self.constraint_pass = constraint_pass

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss_obj = self.compute_loss(x=x, y=y, y_pred=y_pred)
            y_c = y if self.constraint_samples is None else y[:self.constraint_samples]
            if self.constraint_pass == 'primary':
                y_det = y_pred if self.constraint_samples is None \
                    else y_pred[:self.constraint_samples]
            else:
                x_c = x if self.constraint_samples is None else x[:self.constraint_samples]
                y_det = self.model(x_c, training=False)
            penalties = {}
            for c in self.constraints_list:
                if getattr(c, "needs_truth", False):
                    penalties["pen_" + c.name] = c(y_det, y_true=y_c)
                else:
                    penalties["pen_" + c.name] = c(y_det)
            loss = loss_obj + tf.add_n(list(penalties.values()))

        grads = tape.gradient(loss, self.trainable_variables)
        grads_and_vars = []
        for grad, var in zip(grads, self.trainable_variables):
            if grad is None:
                continue
            if id(var) in self._lmbda_ids:
                grads_and_vars.append((-grad, var))
            else:
                grads_and_vars.append((grad, var))
        self.optimizer.apply_gradients(grads_and_vars)

        out = {"loss": loss, "loss_obj": loss_obj}
        out.update(penalties)
        return out

    # --- delegation so existing callbacks/checkpoints work on the inner model ---
    def get_layer(self, name=None, index=None):
        return self.model.get_layer(name=name, index=index)

    def save_weights(self, filepath, *args, **kwargs):
        self.model.save_weights(filepath, *args, **kwargs)

    def load_weights(self, filepath, *args, **kwargs):
        self.model.load_weights(filepath, *args, **kwargs)

    def summary(self, *args, **kwargs):
        return self.model.summary(*args, **kwargs)
