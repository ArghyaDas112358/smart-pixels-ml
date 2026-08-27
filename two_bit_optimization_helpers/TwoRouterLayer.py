# -*- coding: utf-8 -*-
# TwoRouterLayer.py
#
# TWO-ROUTER time-slice selector (option O21b) — the structural sibling of
# SimpleRouterLayer. Same job (learn WHICH k=2 of the T time slices to read
# out), different parameterization: instead of ONE theta whose pair softmax
# p({a,b}) ~ exp(th_a+th_b) must grow TWO bumps to express "one early slice +
# one later slice", this layer owns two INDEPENDENT logit vectors:
#
#     p(a)   = softmax(phiA)                      slot-A router
#     p(b|a) = softmax(phiB with index a masked)  slot-B router, no collision
#
# Each router can commit to its own REGION, so the early+late readout the
# physics wants is the DEFAULT expressive mode, not a contortion of a single
# softmax that has to hold two modes against the winner-take-all pull of its
# own normalization.
#
# Design:
#   - training forward: sample a ~ Cat(phiA), then b ~ Cat(phiB masked at a,
#     renormalized) — ONE pair per step, shared across the batch. Output is
#     x[..., sorted([a,b])] in ASCENDING slice order (early slice -> channel 0,
#     the production 2-slice semantics), so router A's slice may land in slot 0
#     OR slot 1 depending on the draw; the backward pass routes each router's
#     pseudo-gradient to the slot its slice actually occupied.
#   - inference forward: a* = argmax(phiA), b* = argmax(phiB masked at a*),
#     ascending — identical to selected_indices().
#   - backward (tf.custom_gradient): per-router EXACT single-categorical
#     covariance, dphi_i = p_i * (dz_i - sum(p*dz)), with dz built exactly like
#     SimpleRouterLayer's (true slot gradient for the sampled slice, channel
#     mean for the rest). See the estimator comment in call() for the one
#     deliberate approximation (the collision mask is ignored in the estimator).
#   - anti-overlap: a differentiable penalty overlap_scale * sum_i pA_i*pB_i is
#     registered via add_loss() from the smoothed distributions — no estimator
#     needed, autodiff handles it, and it decays to ~0 once the routers commit
#     to different slices.
#   - per-router annealed Gaussian smoothing of the logits along the slice
#     axis, both routers driven by ONE shared non-trainable `smooth_sigma`
#     weight (same scheduler hook and same kernel semantics as
#     SimpleRouterLayer's O15 option: identity at sigma<=1e-6, 3-sigma cutoff,
#     exact edge normalization). Always present — this layer has no checkpoint
#     lineage to stay compatible with, so there is no opt-in flag.
#
# After training, `selected_indices()` returns the committed slice indices
# (the ASIC readout configuration), exactly like SimpleRouterLayer.

import numpy as np
import tensorflow as tf


class TwoRouterLayer(tf.keras.layers.Layer):
    """Selects 2 time slices out of the input's last axis via TWO routers.

    Input  (B, H, W, T)  ->  Output (B, H, W, 2)

    training=True : one pair per call — a ~ softmax(phiA), b ~ softmax(phiB
                    masked at a); output in ascending slice order.
    training falsy: deterministic — argmax(phiA), masked argmax(phiB).
    """

    def __init__(self,
                 num_slots: int = 2,
                 logits_init_stddev: float = 0.0,
                 seed=None,
                 overlap_scale: float = 100.0,
                 **kwargs):
        super(TwoRouterLayer, self).__init__(**kwargs)
        assert num_slots == 2, "TwoRouterLayer is exactly two routers (k=2)."
        self.num_slots = num_slots
        self.logits_init_stddev = logits_init_stddev
        self.seed = seed
        # overlap_scale: weight on the anti-overlap penalty sum_i pA_i * pB_i.
        # The penalty is bounded by overlap_scale (the sum is <= 1), so the
        # default 100 caps it at ~0.3% of the task loss — the task NLL is a
        # batch SUM of magnitude ~3e4 (loss.py uses K.sum), so the penalty can
        # never outvote the task on WHICH regions matter. What it can do is
        # break the tie the task is indifferent to: while both routers sit on
        # overlapping mass at moderate concentration it injects logit gradients
        # of order 1-10 per step, and once the argmaxes separate the product
        # pA_i*pB_i — and with it the gradient — decays toward 0. A standing
        # nudge, not a constraint.
        self.overlap_scale = overlap_scale

    def build(self, input_shape):
        self.num_slices = int(input_shape[-1])
        assert self.num_slices >= self.num_slots, (
            f"input has {self.num_slices} slices but num_slots={self.num_slots}")

        if self.logits_init_stddev and self.logits_init_stddev > 0.0:
            init = tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=self.logits_init_stddev, seed=self.seed)
        else:
            init = tf.keras.initializers.Zeros()   # both routers uniform at start
        self.thetaA = self.add_weight(
            name='thetaA',
            shape=(self.num_slices,),
            initializer=init,
            trainable=True,
        )
        self.thetaB = self.add_weight(
            name='thetaB',
            shape=(self.num_slices,),
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
        # Gaussian smoothing width over the slice axis, driven by a scheduler
        # (duck-typed on `smooth_sigma`, the SmoothSigmaScheduler hook).
        # ONE weight shared by both routers on purpose: the kernel is a
        # coarse-to-fine SEARCH aid (theta has no metric between slices — a
        # gradient can only re-vote slice by slice unless neighbours are
        # coupled), and there is no reason for the two searches to be at
        # different resolutions. sigma = 0 -> identity -> the layer is
        # bit-identical to an un-smoothed one, so the scheduler MUST anneal to
        # exactly 0 (held sigma makes one-slice-wide solutions inexpressible,
        # measured on the O13 checkpoints).
        self.smooth_sigma = self.add_weight(
            name='smooth_sigma',
            shape=(1,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=False,
        )
        d = np.arange(self.num_slices, dtype=np.float32)
        self._dist2 = tf.constant((d[:, None] - d[None, :]) ** 2)   # (T,T)
        super(TwoRouterLayer, self).build(input_shape)

    def sigma(self):
        """Current smoothing width in slices (shared by both routers)."""
        return tf.maximum(tf.reshape(self.smooth_sigma, []), 0.0)

    def smooth(self, phi):
        """phi -> (phi * k_sigma) / (1 * k_sigma) along the slice axis.

        Implemented as a (T,T) row-normalised weight matrix rather than a conv so
        the EDGE HANDLING is exact: each row is renormalised by the kernel mass
        that actually lands in range, so slices near 0 and 100 are not dragged
        toward zero the way plain zero-padded convolution would drag them. The
        early slices this arm exists to make reachable sit in that boundary
        region.
        """
        sg = self.sigma()

        def _smoothed():
            w = tf.exp(-self._dist2 / (2.0 * tf.square(sg)))
            w = tf.where(self._dist2 <= tf.square(3.0 * sg), w, tf.zeros_like(w))
            w = w / tf.reduce_sum(w, axis=1, keepdims=True)
            return tf.linalg.matvec(w, phi)

        return tf.cond(sg > 1e-6, _smoothed, lambda: phi)

    def _phis(self):
        """Both routers' EFFECTIVE (smoothed) logits — everything downstream
        (sampling, readout, marginals, penalty, estimator) works on these, never
        on raw theta, so all views of the layer agree whenever sigma > 0."""
        return (self.smooth(tf.convert_to_tensor(self.thetaA)),
                self.smooth(tf.convert_to_tensor(self.thetaB)))

    @staticmethod
    def _softmax64(phi):
        """Stable softmax where the distribution commits.

        float64 + max-shift, matching SimpleRouterLayer's _pair_stats
        discipline: only exponentials of non-positive numbers are summed —
        positives only, no difference of near-equal quantities — so the
        distribution stays exact all the way to full commitment instead of
        NaN-ing once one logit dominates.
        """
        phi = tf.cast(phi, tf.float64)
        w = tf.exp(phi - tf.reduce_max(phi))
        return w / tf.reduce_sum(w)

    @property
    def theta(self):
        """(T,) per-slice score for LOGGER DISPLAY ONLY: the elementwise max of
        the two smoothed logit tracks. Neither router's distribution is
        recoverable from it — use mu() for anything quantitative. It exists so
        SimpleRouterLogger's `r.theta.numpy()` column keeps showing where the
        strongest per-slice preference sits."""
        phiA, phiB = self._phis()
        return tf.maximum(phiA, phiB)

    def reset_visits(self):
        self.visits.assign(tf.zeros_like(self.visits))

    def mu(self):
        """Per-slice marginals mu_i = pA_i + pB_i; sum(mu) == 2 identically.

        pB is the UNMASKED softmax(phiB): the true marginal of the sampled pair
        would fold the collision mask into pB's contribution, but that term is
        O(pB(a)) — the mass router B puts on router A's slice — which the
        anti-overlap penalty actively drives to zero. Using the unmasked
        marginals keeps mu consistent with the gradient estimator (same
        approximation, see call()) and keeps sum(mu) == 2 exact for the
        logger's entropy."""
        phiA, phiB = self._phis()
        mu = self._softmax64(phiA) + self._softmax64(phiB)
        return tf.cast(mu, self.thetaA.dtype)

    def mu_numpy(self):
        """Marginals as a (num_slices,) float array — THE slice-importance map."""
        return self.mu().numpy()

    def selected_indices(self):
        """Ascending [a*, b*]: a* = argmax(phiA), b* = argmax(phiB masked at
        a*) — the ASIC readout config, identical to the training=False forward.
        Must use the smoothed phis, not raw theta: the sampler works on phi,
        and a pair read off theta would disagree whenever sigma > 0."""
        phiA, phiB = self._phis()
        a = int(np.argmax(phiA.numpy()))
        pb = phiB.numpy().astype(np.float64).copy()
        pb[a] = -np.inf
        b = int(np.argmax(pb))
        return sorted((a, b))

    def call(self, inputs, training=None):
        phiA, phiB = self._phis()

        # Anti-overlap penalty, straight through autodiff — no estimator needed
        # because it depends only on the logits, not on the sampled pair.
        # Computed in float64 like every other place a distribution commits,
        # cast back for add_loss.
        pA_pen = self._softmax64(phiA)
        pB_pen = self._softmax64(phiB)
        overlap = tf.reduce_sum(pA_pen * pB_pen)
        self.add_loss(tf.cast(self.overlap_scale * overlap, phiA.dtype))

        # -1e9, not -inf: exp(-1e9 - max) underflows to exactly 0 in both the
        # categorical sampler and any softmax, without ever forming inf - inf.
        neg = tf.constant(-1e9, dtype=phiB.dtype)
        idx = tf.range(self.num_slices)

        if training:
            # sample ONE pair per training step (shared across the batch):
            # a from router A, then b from router B with a masked out — the
            # renormalization is implicit in categorical sampling, and b == a
            # is impossible by construction.
            s = tf.random.categorical(phiA[tf.newaxis, :], 1, seed=self.seed)
            a = tf.cast(s[0, 0], tf.int32)
            phiB_masked = tf.where(tf.equal(idx, a), neg, phiB)
            s = tf.random.categorical(phiB_masked[tf.newaxis, :], 1, seed=self.seed)
            b = tf.cast(s[0, 0], tf.int32)
            self.visits.assign_add(
                tf.one_hot(a, self.num_slices) + tf.one_hot(b, self.num_slices))
        else:
            a = tf.cast(tf.argmax(phiA), tf.int32)
            phiB_masked = tf.where(tf.equal(idx, a), neg, phiB)
            b = tf.cast(tf.argmax(phiB_masked), tf.int32)

        # ascending slice order: early slice -> channel 0 (production
        # semantics). Router A's slice occupies slot 0 iff a < b; the backward
        # pass must honour that mapping, so remember it here.
        lo = tf.minimum(a, b)
        hi = tf.maximum(a, b)
        a_is_lo = a < b

        @tf.custom_gradient
        def select(x, phiA_in, phiB_in):
            # forward: hard 2-slice pick, ascending slice order
            y = tf.stack([tf.gather(x, lo, axis=-1),
                          tf.gather(x, hi, axis=-1)], axis=-1)   # (B, H, W, 2)

            def grad(dy):
                # Per-slice pseudo-gradient dz per ROUTER, built exactly like
                # SimpleRouterLayer's: what the loss gradient WOULD be if slice
                # t occupied a channel — channel mean for unsampled slices, the
                # TRUE slot gradient for the slice each router actually placed.
                # SLOT MAPPING: the ascending sort means router A's slice sits
                # in slot 0 only when a < b, so A's true gradient is g0 or g1
                # accordingly (and B's is the other one).
                g0 = tf.einsum('bhw,bhwt->t', dy[..., 0], x)
                g1 = tf.einsum('bhw,bhwt->t', dy[..., 1], x)
                mean = 0.5 * (g0 + g1)
                gA = tf.where(a_is_lo, tf.gather(g0, a), tf.gather(g1, a))
                gB = tf.where(a_is_lo, tf.gather(g1, b), tf.gather(g0, b))
                dzA = tf.tensor_scatter_nd_update(mean, tf.reshape(a, (1, 1)), gA[tf.newaxis])
                dzB = tf.tensor_scatter_nd_update(mean, tf.reshape(b, (1, 1)), gB[tf.newaxis])

                # Exact single-categorical covariance per router (float64):
                #     dphi_i = p_i * (dz_i - sum_j p_j dz_j)
                # ESTIMATOR APPROXIMATION, on purpose: b was really drawn from
                # the a-masked, renormalized softmax(phiB), whose score
                # function carries a collision-mask correction (terms in
                # pB(a) / (1 - pB(a)), plus a cross term into dphiA from the
                # mask's dependence on a). Both are O(pB(a)) — the mass router
                # B leaves on router A's slice — which the anti-overlap penalty
                # drives to zero, so we use the clean UNMASKED covariance for
                # both routers. Negligible once the routers separate; near
                # total overlap the penalty gradient dominates anyway.
                pA = TwoRouterLayer._softmax64(phiA_in)
                pB = TwoRouterLayer._softmax64(phiB_in)
                dzA64 = tf.cast(dzA, tf.float64)
                dzB64 = tf.cast(dzB, tf.float64)
                dphiA = pA * (dzA64 - tf.reduce_sum(pA * dzA64))
                dphiB = pB * (dzB64 - tf.reduce_sum(pB * dzB64))
                # Gradient wrt the SMOOTHED logits phi = smooth(theta), which
                # is what `select` was handed; the kernel's self-adjoint
                # chain-rule factor (dtheta = k_sigma @ dphi) is applied by
                # autodiff itself since smooth() is ordinary traced ops.
                # no dx — this router is the model's first layer, nothing
                # upstream needs it.
                return (None,
                        tf.cast(dphiA, dy.dtype),
                        tf.cast(dphiB, dy.dtype))

            return y, grad

        return select(inputs, phiA, phiB)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.num_slots,)

    def get_config(self):
        config = super(TwoRouterLayer, self).get_config()
        config.update({
            'num_slots': self.num_slots,
            'logits_init_stddev': self.logits_init_stddev,
            'seed': self.seed,
            'overlap_scale': self.overlap_scale,
        })
        return config
