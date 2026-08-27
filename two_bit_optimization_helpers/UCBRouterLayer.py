# -*- coding: utf-8 -*-
# UCBRouterLayer.py
#
# Discounted-UCB bandit time-slice selector (option O21c) — the gradient-free
# sibling of SimpleRouterLayer. Same job (pick WHICH k=2 of the T=101 time
# slices to read out), completely different mechanism: the 5050 slice pairs are
# the arms of a discounted-UCB multi-armed bandit, and the choice each training
# step is an argmax over q + exploration bonus — no theta, no sampling
# distribution, NO gradient search at all. The head's gradient flows through a
# plain differentiable gather of the chosen slices; the bandit state receives
# no gradient and is driven only by `bandit_update(loss)` after each step.
#
# Rationale: every gradient router we ran (SIMPLE, beta-annealed, smoothed)
# dies on exploration — mu stays near-uniform or commits to whatever basin the
# first few hundred samples happened to warm. UCB provably keeps sampling
# under-explored arms (the bonus of an unpulled arm grows without bound), so
# it cannot permanently ignore a distant basin; discounting makes it track the
# nonstationary reward created by the head improving on whatever slices the
# bandit currently feeds it.
#
# Division of labour with the training wrapper:
#   call(training=True)  : choose ONE arm (shared across the batch), remember
#                          it in `last_arm`, output x[..., [a, b]] ascending.
#   bandit_update(loss)  : credit that arm with the standardized negative loss.
#                          The wrapper (mdmm.py train_step, patched by the
#                          orchestrator) calls this AFTER the loss is computed;
#                          this file only guarantees the method is pure graph-
#                          safe tf ops. Every call(training=True) MUST be
#                          followed by exactly one bandit_update before the
#                          next choose — the untried-first sweep relies on it.
#
# After training, `selected_indices()` returns the committed slice pair (the
# ASIC readout configuration) = the pair of argmax q, exactly what
# call(training=False) reads out.

import numpy as np
import tensorflow as tf


class UCBRouterLayer(tf.keras.layers.Layer):
    """Hard-selects 2 time slices out of the input's last axis via a
    discounted-UCB bandit over all C(T,2) slice pairs.

    Input  (B, H, W, T)  ->  Output (B, H, W, 2)

    training=True : one pair per call = argmax of q + c*sqrt(log t / n),
                    untried pairs first; shared across the batch.
    training falsy: pair of argmax q (deterministic readout).
    """

    def __init__(self,
                 num_slots: int = 2,
                 c: float = 1.0,
                 gamma: float = 0.9995,
                 reward_alpha: float = 0.01,
                 **kwargs):
        super(UCBRouterLayer, self).__init__(**kwargs)
        assert num_slots == 2, "UCBRouterLayer currently supports num_slots=2 only (pair arms)."
        self.num_slots = num_slots
        # c = 1.0 by default BECAUSE rewards are standardized: bandit_update
        # divides by an EMA std, so q lives on a ~unit-variance scale and the
        # classic UCB1 constant (sqrt(2) for [0,1] rewards) has nothing to
        # compensate for. An arm idle for k steps has its n discounted by
        # gamma^k, so its bonus grows ~exp(k(1-gamma)/2) and is guaranteed to
        # eventually overtake any finite q gap — that is the exploration
        # story, and c sets how soon "eventually" is. KNOW THE REGIME this
        # puts the layer in: with 5050 arms and a 2000-step discount horizon,
        # an arm idle for one full cycle (>= 5050 steps) already carries a
        # bonus of ~9.8*c, above any standardized q gap seen in practice, so
        # post-sweep the bandit runs a CONTINUAL ROUND-ROBIN SURVEY (~0.2%
        # of chooses on the best arm at c=1, measured on the synthetic
        # bandit). q still ranks arms cleanly and tracks reward flips within
        # ~one cycle; what c=1 does NOT give is exploitation-heavy training
        # late in a run — shrinking c buys that only logarithmically, so an
        # orchestrator wanting commitment should anneal c toward ~0 instead.
        self.c = float(c)
        # gamma = 0.9995 -> effective memory 1/(1-gamma) = 2000 steps, ~65
        # epochs at the campaign's 31 steps/epoch. Long enough to average the
        # per-batch loss noise on an arm's q, short enough that the reward
        # shift from the head improving over a training chunk (a few hundred
        # epochs) is tracked rather than averaged away.
        self.gamma = float(gamma)
        # EMA horizon for the reward standardizer, ~100 steps. Only needs to
        # be monotone (it is: an affine map with positive scale preserves the
        # arms' ordering within any window where the stats move slowly).
        self.reward_alpha = float(reward_alpha)

    def build(self, input_shape):
        self.num_slices = int(input_shape[-1])
        assert self.num_slices >= self.num_slots, (
            f"input has {self.num_slices} slices but num_slots={self.num_slots}")

        # all C(T,2) pairs (a<b), precomputed once; arm s <-> pair (ia[s], ib[s])
        ia, ib = np.triu_indices(self.num_slices, k=1)
        self.num_arms = int(ia.size)                       # 5050 at T=101
        self.ia = tf.constant(ia, dtype=tf.int32)
        self.ib = tf.constant(ib, dtype=tf.int32)
        # (2*num_arms, 1) scatter indices: each arm contributes to both of its
        # slices — used by mu() to fold per-arm counts down to per-slice mass.
        self._slice_scatter = tf.constant(
            np.concatenate([ia, ib]).reshape(-1, 1), dtype=tf.int32)
        # (2*num_arms, 2) symmetric matrix indices for the theta display map.
        self._sym_scatter = tf.constant(
            np.stack([np.concatenate([ia, ib]), np.concatenate([ib, ia])], axis=1),
            dtype=tf.int32)

        # ALL bandit state is layer weights so save_weights/load_weights makes
        # checkpoint-resume exact (Gautschi chains die of a GPU leak and resume
        # every chunk; a bandit restarting with q=0 would replay the full
        # 5050-pull sweep each chunk). float64 where the value estimates
        # commit: q gaps between the best and second-best arm shrink as the
        # head converges, and the discounted counts are products of thousands
        # of gamma factors — both places float32 rounding would bite.
        f64_zero = tf.keras.initializers.Zeros()
        self.q = self.add_weight(
            name='q', shape=(self.num_arms,), dtype=tf.float64,
            initializer=f64_zero, trainable=False)
        self.n = self.add_weight(
            name='n', shape=(self.num_arms,), dtype=tf.float64,
            initializer=f64_zero, trainable=False)
        self.t = self.add_weight(
            name='t', shape=(), dtype=tf.float64,
            initializer=f64_zero, trainable=False)
        self.last_arm = self.add_weight(
            name='last_arm', shape=(), dtype=tf.int32,
            initializer=f64_zero, trainable=False)
        # untried-first bookkeeping. Deliberately NOT the discounted n: with
        # gamma=0.9995 an arm's n falls below 0.5 after ~1386 idle steps, which
        # is SHORTER than the 5050-arm sweep itself, so an "n < 0.5 => forced"
        # rule re-arms forced exploration forever and the bandit never exploits.
        # An undiscounted 0/1 flag keeps the intended semantics — every arm is
        # tried exactly once before any repeats — and leaves RE-exploration to
        # the UCB bonus, whose growth on idle arms is the principled version of
        # the same pressure.
        self.tried = self.add_weight(
            name='tried', shape=(self.num_arms,), dtype=tf.float64,
            initializer=f64_zero, trainable=False)
        # reward standardizer state (EMA mean/var of the RAW loss).
        self.loss_mean = self.add_weight(
            name='loss_mean', shape=(), dtype=tf.float64,
            initializer=f64_zero, trainable=False)
        self.loss_var = self.add_weight(
            name='loss_var', shape=(), dtype=tf.float64,
            initializer=tf.keras.initializers.Ones(), trainable=False)
        # visit counter: how often each slice was in the chosen pair (training
        # only) — same contract as SimpleRouterLayer.visits.
        self.visits = self.add_weight(
            name='visits', shape=(self.num_slices,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=False, dtype=tf.float32)
        super(UCBRouterLayer, self).build(input_shape)

    # ------------------------------------------------------------------ state

    def reset_visits(self):
        self.visits.assign(tf.zeros_like(self.visits))

    def sigma(self):
        """No smoothing kernel in this arm; logger contract wants a float."""
        return 0.0

    @property
    def theta(self):
        """(T,) per-slice score for DISPLAY ONLY: max of q over the slice's
        partners. Not a distribution, not what the bandit optimizes — it just
        gives the SimpleRouterLogger a theta-shaped map of which slices sit in
        high-value pairs. Base-filled with min(q) (not -inf) so an untouched
        diagonal can never win the row max or poison downstream arithmetic."""
        q = tf.convert_to_tensor(self.q)
        base = tf.fill((self.num_slices, self.num_slices), tf.reduce_min(q))
        m = tf.tensor_scatter_nd_update(
            base, self._sym_scatter, tf.concat([q, q], axis=0))
        return tf.cast(tf.reduce_max(m, axis=1), tf.float32)

    def mu(self):
        """(T,) per-slice sampling mass, sum == 2 identically (two slots).

        The bandit has no distribution over pairs — the analogue of the SIMPLE
        marginals is the DISCOUNTED pull counts folded to slices: mu_i =
        2 * sum_{arms containing i} n_arm / sum_slices(...). Discounting makes
        this a ~2000-step sliding window, so the logger's entropy reads as
        "how concentrated is the bandit's recent attention". Before the first
        update the counts are all zero; return the uniform 2/T then, matching
        an argmax over all-equal scores being maximally uncommitted.
        """
        per_slice = tf.tensor_scatter_nd_add(
            tf.zeros((self.num_slices,), dtype=tf.float64),
            self._slice_scatter,
            tf.concat([self.n, self.n], axis=0))
        tot = tf.reduce_sum(per_slice)                    # == 2 * sum(n)
        return tf.cast(
            tf.cond(tot > 0,
                    lambda: 2.0 * per_slice / tot,
                    lambda: tf.fill((self.num_slices,),
                                    tf.constant(2.0, tf.float64) / self.num_slices)),
            tf.float32)

    def mu_numpy(self):
        """mu as a (num_slices,) float array — THE slice-importance map."""
        return self.mu().numpy()

    def selected_indices(self):
        """Ascending slice pair of the argmax-q arm — the ASIC readout config.
        Identical to what call(training=False) gathers, by construction."""
        s = int(np.argmax(self.q.numpy()))
        return [int(self.ia.numpy()[s]), int(self.ib.numpy()[s])]

    # ------------------------------------------------------------------ update

    def bandit_update(self, loss):
        """Credit `last_arm` with the outcome of the step it was chosen for.

        Pure tf ops, graph-safe: the training wrapper calls this inside its
        tf.function train_step, AFTER the loss is computed. The reward is the
        NEGATIVE standardized loss — standardized because the raw task loss is
        a batch SUM in the tens of thousands (loss.py uses K.sum) and drifts
        by orders of magnitude as the head trains; q must stay on a stable
        O(1) scale for the c*sqrt(...) bonus to mean anything.
        """
        loss = tf.cast(tf.reshape(loss, []), tf.float64)
        a = self.reward_alpha

        # Seed the EMA mean with the very first loss instead of the arbitrary
        # 0 init: the first reward becomes 0 (neutral) rather than -loss/1,
        # which at |loss|~3e4 would stamp a q value thousands of standard
        # deviations off scale onto whichever arm happened to go first and
        # take ~60 corrective pulls (lr floor 0.05) to walk back.
        first = tf.cast(tf.equal(self.t, 0.0), tf.float64)
        self.loss_mean.assign(first * loss + (1.0 - first) * self.loss_mean)

        delta = loss - self.loss_mean
        self.loss_mean.assign_add(a * delta)
        # exponentially-weighted variance (West 1979 form): only ever adds a
        # non-negative delta^2 term, so it cannot go negative by cancellation.
        self.loss_var.assign((1.0 - a) * (self.loss_var + a * delta * delta))
        r = -delta / tf.sqrt(tf.maximum(self.loss_var, 1e-12))

        arm = tf.reshape(tf.stack([self.last_arm]), (1, 1))
        # discount EVERY arm's count, then credit the pulled one. This is what
        # makes the bandit nonstationarity-aware: an idle arm's n decays, its
        # UCB bonus grows, and it WILL be re-checked no matter how bad its old
        # q was — the property gradient routers lack.
        self.n.assign(self.gamma * self.n)
        self.n.assign(tf.tensor_scatter_nd_add(
            self.n, arm, tf.ones((1,), tf.float64)))
        self.t.assign(self.gamma * self.t + 1.0)
        self.tried.assign(tf.tensor_scatter_nd_update(
            self.tried, arm, tf.ones((1,), tf.float64)))

        # incremental value update with a FLOORED step: 1/n is the exact
        # sample-average schedule while an arm is fresh, but it freezes q as n
        # grows — fatal under a drifting reward. The 0.05 floor keeps every
        # update worth at least a ~20-pull window, so a flipped reward
        # landscape shows up in q within tens of pulls of the stale arm.
        n_arm = tf.reshape(tf.gather_nd(self.n, arm), [])
        lr = tf.maximum(tf.constant(0.05, tf.float64), 1.0 / n_arm)
        q_arm = tf.reshape(tf.gather_nd(self.q, arm), [])
        self.q.assign(tf.tensor_scatter_nd_add(
            self.q, arm, tf.reshape(lr * (r - q_arm), (1,))))

    # ------------------------------------------------------------------ forward

    def call(self, inputs, training=None):
        if training:
            # UCB score in float64: near convergence the top-two q gap can be
            # far below the float32 ulp of the bonus term.
            t_eff = tf.maximum(self.t, 2.0)
            scores = self.q + self.c * tf.sqrt(
                tf.math.log(t_eff) / tf.maximum(self.n, 1e-8))
            # untried-first: a bonus far above any reachable score (|q| is
            # O(10) standardized, the n=1e-8 floor caps the bonus at ~3e4).
            # All untried arms tie exactly (q=0, n=0), so argmax sweeps them
            # in index order — 5050 pulls = ~163 epochs of forced exploration
            # at 31 steps/epoch before the first repeat. Acceptable: with a
            # warm-started head one batch per arm is already an informative
            # reward, so the sweep IS the survey no gradient router managed.
            scores = tf.where(self.tried < 0.5, scores + 1e9, scores)
            s = tf.cast(tf.argmax(scores), tf.int32)
            self.last_arm.assign(s)
            a = tf.gather(self.ia, s)     # a < b by construction of triu pairs
            b = tf.gather(self.ib, s)
            self.visits.assign_add(
                tf.one_hot(a, self.num_slices) + tf.one_hot(b, self.num_slices))
        else:
            # deterministic readout: the best-value arm, ignoring exploration.
            s = tf.cast(tf.argmax(self.q), tf.int32)
            a = tf.gather(self.ia, s)
            b = tf.gather(self.ib, s)

        # plain differentiable gather, ascending slice order (early slice ->
        # channel 0, matching the production 2-slice semantics). No custom
        # gradient: the head trains normally on the chosen slices, and none of
        # the bandit variables are trainable, so autodiff has nothing to reach.
        return tf.stack([tf.gather(inputs, a, axis=-1),
                         tf.gather(inputs, b, axis=-1)], axis=-1)   # (B, H, W, 2)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.num_slots,)

    def get_config(self):
        config = super(UCBRouterLayer, self).get_config()
        config.update({
            'num_slots': self.num_slots,
            'c': self.c,
            'gamma': self.gamma,
            'reward_alpha': self.reward_alpha,
        })
        return config
