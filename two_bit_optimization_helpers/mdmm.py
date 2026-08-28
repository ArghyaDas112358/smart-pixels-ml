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


class ZeroBiasConstraint(OutputConstraint):
    """mean( true - pred ) == 0 inside EVERY soft bin of one target.

    Fix 2 of the residual-bias plan, and the reason the plan needs MDMM at all.
    lgray's "L1 regression with the vertex of the function at y = 0 for each of
    the bins" is exactly an infeasibility measure of |mean residual|: the V has
    constant gradient right down to zero, so unlike a squared penalty it keeps
    pushing instead of going soft near the end and leaving a residual bias.

    One multiplier PER BIN, not one for the constraint. A single shared
    multiplier would let a large bias in one bin be paid for by small biases
    elsewhere -- which is the trade we are trying to forbid.

    transform:
      "identity" -- residual in the model's own label space (x, y)
      "cot2deg"  -- residual in DEGREES (alpha, beta). The plot that started
                    this is in degrees, and cot-space and degree-space bias are
                    not the same thing once a bin is wide, so the angles are
                    constrained where they are actually read.
    """
    needs_truth = True

    def __init__(self, column, label_column, centers, sigma,
                 transform="identity", label_scale=1.0,
                 scale=1.0, damping=1.0, tol=0.0, max_lambda=None,
                 inf_cap=None, **kwargs):
        # base class makes a scalar lmbda; replace it with one per bin
        super().__init__(scale=scale, damping=damping, **kwargs)
        self.column = column
        self.label_column = label_column
        self.centers = tf.constant(centers, dtype=tf.float32)
        self.sigma = float(sigma)
        self.transform = transform
        self.label_scale = float(label_scale)
        # TOLERANCE BAND. "mean bias exactly zero in all 60 bins" is an equality
        # constraint that is probably not reachable, and an unreachable equality
        # means lambda climbs forever BY DESIGN -- which is exactly what happened:
        # the multipliers hit 361k, outvoted the NLL entirely, and the model took
        # the degenerate escape of widening its predictions (sigma +80..200%),
        # because a blurrier prediction has less conditional bias almost by
        # construction. A band makes the constraint SATISFIABLE, so a bin that is
        # good enough costs nothing and its multiplier relaxes.
        # tol is in units of the target's own residual spread. Set it against
        # the PER-BATCH noise, not the eval-set noise: the constraint sees one
        # batch (5,000 events / 15 soft bins ~ 356 per bin -> a bin mean
        # fluctuates by 0.056 sigma), while the residual PLOT is drawn on 37,919
        # events (~2,712 per bin -> 0.021 sigma). Measured, not assumed. A
        # tolerance derived from the plot's noise sits at ~1 sigma of the
        # batch noise, so a third of bins breach it by chance every step and
        # lambda climbs on nothing. 3x the batch noise is ~0.17.
        self.tol = float(tol)
        self.max_lambda = max_lambda
        self.inf_cap = inf_cap
        self.n_bins = int(len(centers))
        self.lmbda = self.add_weight(
            name=self.name + '_lmbda_bins',
            shape=(self.n_bins,),
            initializer='zeros',
            trainable=True,
        )

    def _to_space(self, v):
        if self.transform == "cot2deg":
            from conditional_nll import cot_to_deg
            return cot_to_deg(v, self.label_scale)
        return tf.cast(v, tf.float32)

    def fn(self, outputs, y_true=None):
        """Kernel-weighted mean residual in each bin -> (n_bins,)."""
        pred = self._to_space(outputs[:, self.column])
        true = self._to_space(tf.cast(y_true[:, self.label_column], outputs.dtype))
        res = true - pred
        d = (tf.expand_dims(true, -1) - tf.reshape(self.centers, (1, -1))) / self.sigma
        K = tf.nn.softmax(-0.5 * tf.square(d), axis=-1)          # (N, n_bins)
        occ = tf.reduce_sum(K, axis=0)                            # (n_bins,)
        return tf.reduce_sum(K * tf.expand_dims(res, -1), axis=0) / (occ + 1e-6)

    def _fn_and_spread(self, outputs, y_true):
        """Per-bin mean residual, and the overall residual spread it is
        normalised by. stop_gradient on the spread: it is a unit conversion, not
        something the model should be able to game by inflating its errors."""
        pred = self._to_space(outputs[:, self.column])
        true = self._to_space(tf.cast(y_true[:, self.label_column], outputs.dtype))
        res = true - pred
        d = (tf.expand_dims(true, -1) - tf.reshape(self.centers, (1, -1))) / self.sigma
        K = tf.nn.softmax(-0.5 * tf.square(d), axis=-1)
        occ = tf.reduce_sum(K, axis=0)
        bias = tf.reduce_sum(K * tf.expand_dims(res, -1), axis=0) / (occ + 1e-6)
        spread = tf.stop_gradient(tf.sqrt(tf.math.reduce_variance(res) + 1e-12))
        return bias, spread

    def infeasibility(self, fn_value):
        return tf.abs(fn_value)

    def call(self, outputs, y_true=None):
        bias, spread = self._fn_and_spread(outputs, y_true)
        # DIMENSIONLESS: bias measured in units of that target's own residual
        # spread. Without this, x is in microns (bias ~0.5) and beta is in
        # degrees (bias ~1.5) and one `scale` cannot serve both -- and the
        # degree targets swamped the task loss by four orders of magnitude.
        inf = self.infeasibility(bias) / (spread + 1e-6)
        inf = tf.maximum(inf - self.tol, 0.0)          # inside the band -> free
        if self.inf_cap is not None:
            inf = tf.minimum(inf, self.inf_cap)         # one wild bin cannot dominate
        lam = tf.math.maximum(self.lmbda, 0.0)
        if self.max_lambda is not None:
            lam = tf.minimum(lam, self.max_lambda)      # backstop, not the fix
        l_term = lam * inf
        damp_term = self.damping * tf.square(inf) / 2
        return self.scale * tf.reduce_sum(l_term + damp_term)


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
        # sqrt(var + eps), NOT reduce_std: d(std)/dp is 0/0 = NaN at zero
        # variance, and zero variance is reachable -- on the clip plateau the
        # corr penalty is the ONLY gradient, and Pearson's scale invariance
        # leaves sd(pred) free to drift to 0 (NaN'd O18 seed 18042 at epoch 4).
        # The 1e-6 on the denominator only guards the forward pass.
        sd_p = tf.sqrt(tf.math.reduce_variance(p) + 1e-12)
        sd_t = tf.sqrt(tf.math.reduce_variance(t) + 1e-12)
        return cov / (sd_p * sd_t + 1e-6)

    def infeasibility(self, fn_value):
        return tf.math.maximum(self.min_value - fn_value, 0.0)

    def call(self, outputs, y_true=None):
        inf = self.infeasibility(self.fn(outputs, y_true=y_true))
        l_term = tf.math.maximum(self.lmbda, 0.0) * inf
        damp_term = self.damping * tf.square(inf) / 2
        return self.scale * (l_term + damp_term)


class MaxBlockNLLConstraint(OutputConstraint):
    """mean(-log p(block | earlier targets)) <= max_value, on the split NLL.

    Constrains the PHYSICS quantity rather than a proxy. Two properties the
    correlation floor does not have:

      * it cannot be met by faking confidence -- spreading the predictions
        wrongly enlarges z^2, which raises the NLL, and
      * it cannot be met by hedging -- inflating L_kk pays the log L_kk term.

    Correlation is invariant to affine rescaling, so it happily passes a model
    whose angle predictions span a few percent of the true range (measured:
    seed 4042 clears corr >= 0.5 at sd(pred)/sd(true) = 0.06). The conditional
    NLL has no such blind spot.

    `block` is 'angle' (terms 2+3 = -log p(cotA,cotB | x,y)), 'position' (0+1),
    or a single target: 'x', 'y', 'cotA', 'cotB' (or its index).

    MEASURED 2026-08-03 on the full 37,919-event validation set, the ANGLE BLOCK
    barely discriminates -- seed 2042 beats 4042 by 0.047 nats, because the seeds
    trade the two angles against each other (2042 is 0.374 better on cotA and
    0.326 worse on cotB). 'cotA' separates them cleanly instead: -0.044 for 2042
    against +0.319..+0.397 for the rest. Prefer block='cotA'.

    'cotA' is also unambiguous: term 2 is conditioned ONLY on x,y with nothing
    downstream, so unlike cotB it does not depend on the chain order.

    If the target is below what two 2-bit slices can support the constraint is
    infeasible and lambda climbs without bound -- which is informative: sweep
    max_value and the divergence point locates the information ceiling. Cap
    lambda when running such a sweep.
    """
    needs_truth = True

    def __init__(self, max_value, block='angle', scale=1.0, damping=1.0,
                 minval=1e-9, maxval=1e9, max_lambda=None, inf_cap=None, **kwargs):
        super().__init__(scale=scale, damping=damping, **kwargs)
        self.max_value = max_value
        self.block = block
        self.minval = minval
        self.maxval = maxval
        # Ceiling on the EFFECTIVE multiplier. Without one, an unreachable target
        # makes lambda climb forever: measured 2026-08-04 on O13 seeds 13042/13142
        # at lambda ~30, the penalty (scale * lambda * gap) reached ~80,000 against
        # a task loss of ~-13,000, and both seeds destroyed their own likelihood --
        # cot alpha reversed -0.02 -> +0.38 and the router collapsed onto the
        # adjacent pair [7,8]. Seed 13342 escaped only by reaching the target,
        # which stops lambda growing. The cap keeps the pressure without letting
        # the constraint swamp the objective.
        self.max_lambda = max_lambda
        # Ceiling on the INFEASIBILITY used in the penalty. The damping term is
        # scale * inf^2 / 2 and carries NO lambda, so a lambda cap cannot bound it.
        # Measured 2026-08-04 (O13 seed 13442): a seed that starts on the clipped
        # plateau has both angle terms pinned at the 20.72 ceiling, so inf ~ 21 and
        # the damping term alone is 5000 * 21^2 / 2 ~ 1.1e6 per constraint -- 23x
        # the task loss. The gradients NaN'd the model on epoch 2. Capping inf
        # keeps early pressure gentle and lets the NLL drive the escape, exactly as
        # it does in the corr-constrained runs where inf is bounded by 1.5 anyway.
        self.inf_cap = inf_cap

    def fn(self, outputs, y_true=None):
        from conditional_nll import nll_terms
        # float64 THROUGHOUT. At initialisation L's diagonal sits on its 1e-9
        # floor, so z = L^-1 (y-mu) divides by 1e-9 and the terms reach ~1e73;
        # in float32 that is +inf for ~23% of events.
        t = nll_terms(tf.cast(y_true, tf.float64), tf.cast(outputs, tf.float64))
        NAMED = {'x': 0, 'y': 1, 'cotA': 2, 'cotB': 3}
        if self.block == 'angle':
            v = t[:, 2] + t[:, 3]
        elif self.block == 'position':
            v = t[:, 0] + t[:, 1]
        else:
            v = t[:, NAMED[self.block] if self.block in NAMED else int(self.block)]
        # SAME per-event bound loss.custom_loss imposes by clipping the density to
        # [minval, maxval] before the log. Without it the infeasibility at init is
        # ~1e70 and the damping term scale*inf^2/2 overflows float32 -> NaN on the
        # very first batch (observed 2026-08-03). With it, the block at init reads
        # 20.55 against custom_loss's own 20.59 -- the same regime, bounded.
        lo = -tf.math.log(tf.constant(self.maxval, tf.float64))
        hi = -tf.math.log(tf.constant(self.minval, tf.float64))
        v = tf.clip_by_value(v, lo, hi)
        return tf.cast(tf.reduce_mean(v), outputs.dtype)

    def infeasibility(self, fn_value):
        return tf.math.maximum(fn_value - self.max_value, 0.0)

    def call(self, outputs, y_true=None):
        inf = self.infeasibility(self.fn(outputs, y_true=y_true))
        if self.inf_cap is not None:
            inf = tf.math.minimum(inf, tf.cast(self.inf_cap, inf.dtype))
        lam = tf.math.maximum(self.lmbda, 0.0)
        if self.max_lambda is not None:
            lam = tf.math.minimum(lam, tf.cast(self.max_lambda, lam.dtype))
        l_term = lam * inf
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

        # Bandit-style routers (O21c) learn from the realized batch loss rather
        # than a gradient; they expose bandit_update() and get the loss AFTER
        # the step so the reward reflects the pair that was actually read out.
        # hasattr keeps every gradient-router model on the exact old path.
        for _l in self.model.layers:
            if hasattr(_l, "bandit_update"):
                _l.bandit_update(tf.stop_gradient(loss))

        out = {"loss": loss, "loss_obj": loss_obj}
        out.update(penalties)
        # Compiled metrics are NOT surfaced automatically -- this train_step
        # builds its own return dict, so anything passed to compile(metrics=...)
        # silently vanishes. O22 needs plain_nll here: the training loss may be a
        # weighted composite, and plain_nll is the one number that stays
        # comparable with the ledger.
        for _m in (self.compiled_metrics._metrics if self.compiled_metrics is not None
                   and getattr(self.compiled_metrics, "_metrics", None) else []):
            try:
                out[_m.__name__ if callable(_m) else str(_m)] = _m(y, y_pred)
            except Exception:
                pass
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
