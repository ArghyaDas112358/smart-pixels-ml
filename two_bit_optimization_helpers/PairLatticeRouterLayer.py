# -*- coding: utf-8 -*-
# PairLatticeRouterLayer.py
#
# PAIR-LATTICE subset-sampling time-slice selector (option O21a) — the sibling
# of SimpleRouterLayer with the factorised per-slice theta replaced by a FREE
# logit psi_s for every one of the C(T,2) slice pairs:
#
#     p(pair s) = softmax_s( smooth(psi) )
#
# Why: SimpleRouterLayer's p({a,b}) ~ exp(theta_a + theta_b) forces a rank-1
# structure on the pair distribution — the layer cannot prefer (11,26) without
# also inflating (11,x) and (x,26) for every x, so a WIDE pair competes with
# 198 parasitic pairs it never asked for. Here every pair carries its own
# logit, and the coarse-to-fine smoothing lives in PAIR SPACE: psi is scattered
# into the symmetric (T,T) pair matrix and blurred with a 2D Gaussian, so the
# neighbours of (11,26) are (10,27), (12,25), (10,26), ... — wide pairs are
# neighbours of wide pairs. A slice-axis kernel on theta (option O15/O18) can
# only slide the two marginals independently; this one slides the JOINT
# hypothesis, which is the whole point of the arm.
#
# Mechanics otherwise mirror SimpleRouterLayer exactly:
#   - training forward: sample ONE pair per step (shared across the batch),
#     output x[..., [a, b]] in ASCENDING slice order (a < b by triu
#     construction, matching the production 2-slice semantics).
#   - inference forward: argmax over the smoothed psi — deterministic, and the
#     same readout selected_indices() reports.
#   - backward (tf.custom_gradient): per-slice pseudo-gradient dz built from
#     the downstream channel gradients (identical construction), lifted to
#     per-pair pseudo-values v_s = dz[ia_s] + dz[ib_s], then the exact
#     categorical score-function/covariance gradient
#         d(psi_smoothed)_s = p_s * (v_s - sum_t p_t v_t)
#     in float64. Autodiff carries it back through the smoothing to psi.
#   The layer is the FIRST layer of the model, so no gradient is returned for
#   the input tensor.
#
# After training, `selected_indices()` returns the committed slice indices
# (the ASIC readout configuration), exactly like SimpleRouterLayer.

import numpy as np
import tensorflow as tf


class PairLatticeRouterLayer(tf.keras.layers.Layer):
    """Samples/selects 2 time slices out of the input's last axis, learnably,
    from a free categorical distribution over all C(T,2) slice pairs.

    Input  (B, H, W, T)  ->  Output (B, H, W, 2)

    training=True : one pair per call, sampled from softmax(smooth(psi));
    training falsy: argmax pair of the smoothed psi (deterministic).
    """

    def __init__(self,
                 num_slots: int = 2,
                 logits_init_stddev: float = 0.0,
                 seed=None,
                 smooth_logits: bool = False,
                 **kwargs):
        super(PairLatticeRouterLayer, self).__init__(**kwargs)
        assert num_slots == 2, "PairLatticeRouterLayer supports num_slots=2 only (the pair lattice IS C(T,2))."
        self.num_slots = num_slots
        self.logits_init_stddev = logits_init_stddev
        self.seed = seed
        # smooth_logits: expose a NON-TRAINABLE `smooth_sigma` weight so a
        # scheduler (SmoothSigmaScheduler, duck-typed on `smooth_sigma`) can
        # blur psi with a 2D Gaussian IN PAIR SPACE before the softmax. Same
        # coarse-to-fine contract as SimpleRouterLayer's O15 kernel: sigma is
        # ANNEALED TO EXACTLY 0, never held, and at sigma=0 the layer is
        # bit-identical to the un-smoothed one. OPT-IN because the extra
        # weight breaks load_weights() for checkpoints written without it.
        self.smooth_logits = smooth_logits

    def build(self, input_shape):
        self.num_slices = int(input_shape[-1])
        assert self.num_slices >= self.num_slots, (
            f"input has {self.num_slices} slices but num_slots={self.num_slots}")

        # all C(T,2) pairs (a<b), precomputed once. ia[s] < ib[s] for every s,
        # so a sampled/argmaxed pair is ascending by construction.
        ia, ib = np.triu_indices(self.num_slices, k=1)
        self.num_pairs = len(ia)
        self.ia = tf.constant(ia, dtype=tf.int32)
        self.ib = tf.constant(ib, dtype=tf.int32)
        # scatter/gather index sets for the (T,T) pair matrix: the upper
        # triangle alone for the gather-back, BOTH triangles for the scatter so
        # the matrix is symmetric and the 2D blur treats (i,j) and (j,i) as the
        # same hypothesis.
        triu = np.stack([ia, ib], axis=1)
        self._triu_idx = tf.constant(triu, dtype=tf.int32)                       # (P, 2)
        self._both_idx = tf.constant(
            np.concatenate([triu, triu[:, ::-1]], axis=0), dtype=tf.int32)       # (2P, 2)

        if self.logits_init_stddev and self.logits_init_stddev > 0.0:
            init = tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=self.logits_init_stddev, seed=self.seed)
        else:
            init = tf.keras.initializers.Zeros()   # uniform over all pairs at start
        self.psi = self.add_weight(
            name='psi',
            shape=(self.num_pairs,),
            initializer=init,
            trainable=True,
        )
        # visit counter: how often each slice was in the sampled pair (training only)
        self.visits = self.add_weight(
            name='visits',
            shape=(self.num_slices,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=False,
            dtype=tf.float32,
        )
        if self.smooth_logits:
            # Gaussian blur width IN SLICES along each axis of the pair matrix,
            # driven by SmoothSigmaScheduler. sigma = 0 -> identity.
            self.smooth_sigma = self.add_weight(
                name='smooth_sigma',
                shape=(1,),
                initializer=tf.keras.initializers.Zeros(),
                trainable=False,
            )
            d = np.arange(self.num_slices, dtype=np.float32)
            self._dist2 = tf.constant((d[:, None] - d[None, :]) ** 2)            # (T,T)
            # valid-entry mask of the pair matrix: every off-diagonal cell holds
            # a real pair, the diagonal (i,i) is NOT a pair and must neither
            # receive nor contribute smoothing mass.
            self._offmask = tf.constant(
                1.0 - np.eye(self.num_slices, dtype=np.float32))
        super(PairLatticeRouterLayer, self).build(input_shape)

    # ------------------------------------------------------------------ sigma
    def _sigma_t(self):
        """Smoothing width as a graph-safe scalar tensor (internal use)."""
        if not self.smooth_logits:
            return tf.constant(0.0, dtype=tf.float32)
        return tf.maximum(tf.reshape(self.smooth_sigma, []), 0.0)

    def sigma(self):
        """Current smoothing width in slices as a python float (0.0 when
        smoothing is disabled). Eager-only by design — the logger calls it
        between epochs; graph code uses _sigma_t()."""
        return float(self._sigma_t().numpy())

    def reset_visits(self):
        self.visits.assign(tf.zeros_like(self.visits))

    # -------------------------------------------------------------- smoothing
    def smooth(self, psi):
        """psi -> 2D-Gaussian blur of psi over the symmetric pair matrix.

        Separable: with K the UNNORMALISED 1D kernel matrix
        (K[i,k] = exp(-(i-k)^2 / 2 sigma^2), 3-sigma cutoff), the blur of the
        scattered matrix M is K M K^T and the normaliser is K mask K^T — i.e.
        each pair cell is renormalised by the kernel mass that lands on VALID
        (off-diagonal, in-range) cells. That makes the EDGE HANDLING exact in
        both senses at once: pairs near slice 0/100 (matrix border) and pairs
        near the diagonal (narrow |a-b|) are not dragged toward zero the way a
        zero-padded convolution would drag them. Same reason SimpleRouterLayer
        row-normalises its 1D kernel; here the excluded diagonal is a second
        edge the 1D version doesn't have.
        """
        if not self.smooth_logits:
            return psi
        sg = self._sigma_t()

        def _smoothed():
            K = tf.exp(-self._dist2 / (2.0 * tf.square(sg)))
            K = tf.where(self._dist2 <= tf.square(3.0 * sg), K, tf.zeros_like(K))
            M = tf.scatter_nd(self._both_idx,
                              tf.concat([psi, psi], axis=0),
                              (self.num_slices, self.num_slices))
            num = tf.matmul(tf.matmul(K, M), K)            # K symmetric: K M K^T
            den = tf.matmul(tf.matmul(K, self._offmask), K)
            # den >= 1 on every valid cell (the cell's own K[i,i]K[j,j] = 1 term
            # always lands in range), but on the DIAGONAL den -> 0 as sigma
            # shrinks below 1/3 and num/den would be 0/0 there. The diagonal is
            # never gathered, but a NaN in the forward would poison the
            # gather's scatter-gradient (0/0 again), so pad the diagonal
            # denominator to 1 — exact on all gathered cells.
            den = den + (1.0 - self._offmask)
            return tf.gather_nd(num / den, self._triu_idx)

        # identity below the scheduler's floor, bit-identical to no smoothing
        return tf.cond(sg > 1e-6, _smoothed, lambda: psi)

    # ----------------------------------------------------------- distribution
    @staticmethod
    def _pair_probs(phi):
        """float64 softmax over the pair lattice, max-subtracted.

        Unlike SimpleRouterLayer._pair_stats there is no factorised shortcut to
        be tempted by — psi is already the full pair logit vector — so the
        distribution is a plain softmax: a sum of positives after the max
        subtraction, exact all the way to full commitment. float64 because this
        is where the distribution commits: at psi spreads ~40 the runner-up
        weights are ~1e-18 and float32 would round the tail (and with it every
        gradient p_s*(v_s - E v) on the tail) to zero.
        """
        phi = tf.cast(phi, tf.float64)
        w = tf.exp(phi - tf.reduce_max(phi))
        return w / tf.reduce_sum(w)

    def mu(self):
        """Exact per-slice marginals mu_i = P(i in sampled pair); sum(mu) == 2.

        Row sums of the pair-probability matrix: scatter p symmetrically and
        sum each row — implemented as a segment-sum over (ia, ib), which is the
        same contraction without materialising the matrix. Each pair
        contributes its whole probability to BOTH of its slices, and p sums to
        1, so sum(mu) == 2 identically. Computed on the SMOOTHED psi, so mu
        always describes the distribution the layer actually samples from.
        """
        phi = self.smooth(tf.convert_to_tensor(self.psi))
        p = self._pair_probs(phi)
        mu = tf.math.unsorted_segment_sum(
            tf.concat([p, p], axis=0),
            tf.concat([self.ia, self.ib], axis=0),
            num_segments=self.num_slices)
        return tf.cast(mu, self.psi.dtype)

    def mu_numpy(self):
        """Marginals as a (num_slices,) float array — THE slice-importance map."""
        return self.mu().numpy()

    @property
    def theta(self):
        """Per-slice log-marginal-quality score: logsumexp over each slice's
        partners of the smoothed psi. LOGGER DISPLAY ONLY — nothing samples
        from it. It is the pair-lattice analogue of SimpleRouterLayer's theta
        (there, theta_i + logsumexp_j(theta_j) plays this role); softmax(theta)
        would give a marginal-like ranking but NOT mu, which is exact and
        computed separately."""
        phi = self.smooth(tf.convert_to_tensor(self.psi))
        neg = tf.fill((self.num_slices, self.num_slices),
                      tf.constant(-np.inf, dtype=phi.dtype))
        # -inf diagonal drops (i,i) from the logsumexp; every row keeps its
        # T-1 real partners, so the reduction is always finite.
        M = tf.tensor_scatter_nd_update(
            neg, self._both_idx, tf.concat([phi, phi], axis=0))
        return tf.reduce_logsumexp(M, axis=1)

    def selected_indices(self):
        """Ascending argmax pair of the SMOOTHED psi — the ASIC readout config.

        Must use the smoothed phi, not raw psi: the sampler, mu() and the eval
        forward all work on phi, and reporting the raw-psi argmax would
        disagree with the pair the layer actually reads out whenever sigma > 0.
        """
        phi = self.smooth(tf.convert_to_tensor(self.psi)).numpy()
        s = int(np.argmax(phi))
        return sorted([int(self.ia.numpy()[s]), int(self.ib.numpy()[s])])

    # ------------------------------------------------------------------ call
    def call(self, inputs, training=None):
        phi = self.smooth(tf.convert_to_tensor(self.psi))

        if training:
            # sample ONE pair per training step (shared across the batch).
            # max-subtracted float64 logits: tf.random.categorical exponentiates
            # internally, and the committed regime (psi spreads ~40+) overflows
            # float32 exp without the shift.
            logits = tf.cast(phi, tf.float64)
            logits = logits - tf.reduce_max(logits)
            s = tf.random.categorical(logits[tf.newaxis, :], 1, seed=self.seed)
            s = tf.cast(s[0, 0], tf.int32)
            a = tf.gather(self.ia, s)     # a < b by construction of triu pairs
            b = tf.gather(self.ib, s)
            self.visits.assign_add(
                tf.one_hot(a, self.num_slices) + tf.one_hot(b, self.num_slices))
        else:
            s = tf.cast(tf.argmax(phi), tf.int32)   # same readout as selected_indices()
            a = tf.gather(self.ia, s)
            b = tf.gather(self.ib, s)

        @tf.custom_gradient
        def select(x, phin):
            # forward: hard 2-slice pick, ascending slice order
            y = tf.stack([tf.gather(x, a, axis=-1),
                          tf.gather(x, b, axis=-1)], axis=-1)   # (B, H, W, 2)

            def grad(dy):
                # per-slice pseudo-gradient dz: what the loss gradient WOULD be
                # if slice t occupied a channel; the sampled slices keep their
                # true channel gradient, everything else gets the channel mean.
                # IDENTICAL to SimpleRouterLayer — the lift to pair space
                # happens after, not here.
                g0 = tf.einsum('bhw,bhwt->t', dy[..., 0], x)
                g1 = tf.einsum('bhw,bhwt->t', dy[..., 1], x)
                dz = 0.5 * (g0 + g1)
                dz = tf.tensor_scatter_nd_update(
                    dz,
                    tf.reshape(tf.stack([a, b]), (2, 1)),
                    tf.stack([tf.gather(g0, a), tf.gather(g1, b)]))

                # per-pair pseudo-value: pair s's linearised loss change is the
                # sum of its two slices' (slot 0 for ia, slot 1 for ib — the
                # ascending forward semantics make that assignment exact).
                dz64 = tf.cast(dz, tf.float64)
                v = tf.gather(dz64, self.ia) + tf.gather(dz64, self.ib)
                # exact categorical covariance gradient of E_p[v] wrt the
                # smoothed logits: dphi_s = p_s (v_s - E_p[v]). This is the
                # plain-softmax specialisation of the reference's
                # mu*dz + P@dz - mu*sum(mu*dz); it sums to 0 over s by
                # construction. Autodiff carries dphi back through smooth()
                # (a linear map, adjoint applied exactly) to psi.
                p = PairLatticeRouterLayer._pair_probs(phin)
                dphi = p * (v - tf.reduce_sum(p * v))
                dphi = tf.cast(dphi, dz.dtype)
                # no dx — this router is the model's first layer, nothing
                # upstream needs it.
                return None, dphi

            return y, grad

        return select(inputs, phi)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.num_slots,)

    def get_config(self):
        config = super(PairLatticeRouterLayer, self).get_config()
        config.update({
            'num_slots': self.num_slots,
            'logits_init_stddev': self.logits_init_stddev,
            'seed': self.seed,
            'smooth_logits': self.smooth_logits,
        })
        return config
