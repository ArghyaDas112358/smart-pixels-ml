# -*- coding: utf-8 -*-
# SimpleRouterLayer.py
#
# SIMPLE subset-sampling time-slice selector (Ahmed et al., ICLR 2023) — the
# Option-D sibling of SoftRouterLayer. Same job (learn WHICH k=2 time slices to
# read out on the TIME axis), different mechanism: instead of an annealed
# masked-softmax blend, this layer SAMPLES one slice pair per training step from
# an exponential-family distribution over all C(T,2) pairs and backpropagates
# the EXACT gradient of the expected loss through a closed-form covariance
# expression. No annealer, no straight-through estimator, no temperature.
#
# Design:
#   - ONE trainable weight theta (T,). Pair distribution:
#       p({a,b}) = exp(theta_a + theta_b) / Z,   Z = sum_{a<b} exp(theta_a + theta_b)
#   - marginals (exact, stable):  w = exp(theta - max(theta));
#       S1 = sum(w); S2 = sum(w^2); Z = 0.5*(S1^2 - S2); mu = w*(S1 - w)/Z
#     sum(mu) == 2 identically (two slots).
#   - training forward: sample ONE pair (a,b), a<b, shared across the batch;
#     output x[..., [a, b]] in ASCENDING slice order (early slice -> channel 0,
#     matching the production 2-slice semantics).
#   - inference forward: deterministic ascending top-2 of theta.
#   - backward (tf.custom_gradient): per-slice pseudo-gradient dz built from the
#     downstream channel gradients, then dtheta = Cov(z) @ dz where Cov has
#     diag mu_i(1-mu_i) and off-diag p_ij - mu_i*mu_j with p_ij = w_i*w_j/Z —
#     evaluated in closed form without materializing the (T,T) matrix.
#   The layer is the FIRST layer of the model, so no gradient is returned for
#   the input tensor. Downstream SoftQuantizeLayer keeps its own annealer; this
#   layer needs NONE.
#
# After training, `selected_indices()` returns the committed slice indices
# (the ASIC readout configuration), exactly like SoftRouterLayer.

import numpy as np
import tensorflow as tf


class SimpleRouterLayer(tf.keras.layers.Layer):
    """Samples/selects 2 time slices out of the input's last axis, learnably.

    Input  (B, H, W, T)  ->  Output (B, H, W, 2)

    training=True : one pair per call, sampled from p({a,b}) ~ exp(th_a + th_b);
    training falsy: ascending top-2 of theta (deterministic).
    """

    def __init__(self,
                 num_slots: int = 2,
                 logits_init_stddev: float = 0.0,
                 seed=None,
                 anneal_beta: bool = False,
                 **kwargs):
        super(SimpleRouterLayer, self).__init__(**kwargs)
        assert num_slots == 2, "SimpleRouterLayer currently supports num_slots=2 only (k=2 pairs)."
        self.num_slots = num_slots
        self.logits_init_stddev = logits_init_stddev
        self.seed = seed
        # anneal_beta: expose an inverse-temperature `log_k` weight so an
        # AnnealingScheduler can SHARPEN the pair distribution over training
        # (option O4 / the Concrete-Autoencoder recipe):
        #     p({a,b}) ~ exp(beta * (theta_a + theta_b)),  beta = exp(log_k)
        # SIMPLE's exactness is untouched -- the marginals and the covariance
        # gradient are computed for the distribution at beta*theta, and dtheta
        # picks up the chain-rule factor beta. Without this the pair
        # distribution has NOTHING pushing it to concentrate: measured mu
        # spanned ~60 effective slices after 2500 epochs, so one weight set and
        # one threshold triple had to serve amplitudes from 74 to 335 mV.
        # OPT-IN ON PURPOSE: enabling it adds a weight, which would break
        # load_weights() for every checkpoint written by the beta-less runs.
        self.anneal_beta = anneal_beta

    def build(self, input_shape):
        self.num_slices = int(input_shape[-1])
        assert self.num_slices >= self.num_slots, (
            f"input has {self.num_slices} slices but num_slots={self.num_slots}")

        if self.logits_init_stddev and self.logits_init_stddev > 0.0:
            init = tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=self.logits_init_stddev, seed=self.seed)
        else:
            init = tf.keras.initializers.Zeros()   # uniform over all pairs at start
        self.theta = self.add_weight(
            name='theta',
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
        # inverse temperature, driven by an AnnealingScheduler (duck-typed on
        # `log_k`, same hook SoftQuantizeLayer uses). beta = exp(log_k), so
        # log_k = 0 -> beta = 1 -> identical to the un-annealed layer.
        if self.anneal_beta:
            self.log_k = self.add_weight(
                name='log_k',
                shape=(1,),
                initializer=tf.keras.initializers.Zeros(),
                trainable=False,
            )
        # all C(T,2) pairs (a<b), precomputed once
        ia, ib = np.triu_indices(self.num_slices, k=1)
        self.ia = tf.constant(ia, dtype=tf.int32)
        self.ib = tf.constant(ib, dtype=tf.int32)
        super(SimpleRouterLayer, self).build(input_shape)

    def beta(self):
        """Inverse temperature (1.0 when beta annealing is disabled)."""
        if not self.anneal_beta:
            return tf.constant(1.0, dtype=tf.float32)
        return tf.exp(tf.reshape(self.log_k, []))

    def reset_visits(self):
        self.visits.assign(tf.zeros_like(self.visits))

    @staticmethod
    def _pair_stats(phi, num_slices):
        """Exact pair distribution for logits `phi`, computed WITHOUT any
        difference of near-equal quantities.

            P_ij = p({i,j}) = w_i w_j / Z   (i != j, zero diagonal)
            Z    = sum_{i<j} w_i w_j = 0.5 * sum_{i != j} w_i w_j
            mu_i = sum_j P_ij,   sum(mu) == 2

        Returns (mu, P) in float64.

        WHY NOT the algebraic short-cut: the closed forms
        Z = 0.5*(S1^2 - S2) and mu = w*(S1-w)/Z are mathematically identical but
        catastrophically unstable once one weight dominates -- exactly the
        committed regime beta annealing drives toward. With w = [1, 1.5e-8, ...]
        both S1^2 and S2 round to 1.0 in float32, so Z evaluates to 0 and mu
        becomes NaN (measured: NaN at beta=60, 5e-2 error by beta=10). Summing
        the off-diagonal outer product instead only ever adds positives, so it
        stays exact all the way to full commitment. O(T^2) = 10201 terms at
        T=101, i.e. free next to the ViT.
        """
        phi = tf.cast(phi, tf.float64)
        w = tf.exp(phi - tf.reduce_max(phi))
        off = 1.0 - tf.eye(num_slices, dtype=tf.float64)
        num = (w[:, None] * w[None, :]) * off          # w_i w_j, zero diagonal
        Z = 0.5 * tf.reduce_sum(num)
        P = num / Z                                    # p({i,j}), sums to 1 over i<j
        return tf.reduce_sum(P, axis=1), P

    def mu(self):
        """Exact per-slice marginals mu_i = P(i in sampled pair); sum(mu) == 2.

        Computed for the SHARPENED logits phi = beta * theta (beta = 1 unless
        beta annealing is on), so mu always describes the distribution the layer
        actually samples from.
        """
        phi = self.beta() * tf.convert_to_tensor(self.theta)
        mu, _ = self._pair_stats(phi, self.num_slices)
        return tf.cast(mu, self.theta.dtype)

    def mu_numpy(self):
        """Marginals as a (num_slices,) float array — THE slice-importance map."""
        return self.mu().numpy()

    def selected_indices(self):
        """Ascending top-2 of theta as plain ints — the ASIC readout config."""
        th = self.theta.numpy()
        top2 = np.argsort(th)[-2:]
        return sorted(int(i) for i in top2)

    def call(self, inputs, training=None):
        beta = self.beta()
        th = beta * tf.convert_to_tensor(self.theta)     # sharpened logits phi

        if training:
            # sample ONE pair per training step (shared across the batch)
            pair_logits = tf.gather(th, self.ia) + tf.gather(th, self.ib)
            s = tf.random.categorical(pair_logits[tf.newaxis, :], 1, seed=self.seed)
            s = tf.cast(s[0, 0], tf.int32)
            a = tf.gather(self.ia, s)     # a < b by construction of triu pairs
            b = tf.gather(self.ib, s)
            self.visits.assign_add(
                tf.one_hot(a, self.num_slices) + tf.one_hot(b, self.num_slices))
        else:
            top2 = tf.sort(tf.math.top_k(th, k=2).indices)   # ascending
            a, b = top2[0], top2[1]

        @tf.custom_gradient
        def select(x, thin):
            # forward: hard 2-slice pick, ascending slice order
            y = tf.stack([tf.gather(x, a, axis=-1),
                          tf.gather(x, b, axis=-1)], axis=-1)   # (B, H, W, 2)

            def grad(dy):
                # per-slice pseudo-gradient dz: what the loss gradient WOULD be
                # if slice t occupied a channel; the sampled slices keep their
                # true channel gradient, everything else gets the channel mean.
                g0 = tf.einsum('bhw,bhwt->t', dy[..., 0], x)
                g1 = tf.einsum('bhw,bhwt->t', dy[..., 1], x)
                dz = 0.5 * (g0 + g1)
                dz = tf.tensor_scatter_nd_update(
                    dz,
                    tf.reshape(tf.stack([a, b]), (2, 1)),
                    tf.stack([tf.gather(g0, a), tf.gather(g1, b)]))

                # exact dphi = Cov(z) @ dz, where Cov has diagonal mu_i(1-mu_i)
                # and off-diagonal p_ij - mu_i mu_j. Expanding and cancelling the
                # mu_i^2 dz_i terms leaves
                #     dphi = mu*dz + P@dz - mu*sum(mu*dz)
                # with P the zero-diagonal pair-probability matrix. Uses the
                # stable _pair_stats (see its docstring: the algebraic
                # S1^2 - S2 form NaNs once the distribution commits).
                mu64, P = SimpleRouterLayer._pair_stats(thin, self.num_slices)
                dz64 = tf.cast(dz, tf.float64)
                smd = tf.reduce_sum(mu64 * dz64)
                # Gradient wrt the SHARPENED logits phi = beta*theta, which is
                # what `select` was handed. The beta chain-rule factor
                # (dtheta = beta * dphi) is applied by autodiff itself, since
                # phi = beta * theta is an ordinary traced op in call() --
                # multiplying by beta here would double-count it.
                dphi = mu64 * dz64 + tf.linalg.matvec(P, dz64) - mu64 * smd
                dphi = tf.cast(dphi, dz.dtype)
                # no dx — this router is the model's first layer, nothing
                # upstream needs it.
                return None, dphi

            return y, grad

        return select(inputs, th)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.num_slots,)

    def get_config(self):
        config = super(SimpleRouterLayer, self).get_config()
        config.update({
            'num_slots': self.num_slots,
            'logits_init_stddev': self.logits_init_stddev,
            'seed': self.seed,
            'anneal_beta': self.anneal_beta,
        })
        return config
