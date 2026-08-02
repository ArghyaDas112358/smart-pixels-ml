"""
Confidence-weighted distillation: a frozen teacher provides a per-event
total predictive variance; the student's data-NLL loss is weighted by
teacher confidence (1 / total_variance), clipped and normalized to mean 1.

This is NOT mimicry: there is no KL term forcing the student to match the
teacher's outputs. The teacher only supplies a per-event *curriculum* --
which labels to trust. On events where the teacher is confident (low
variance), the student is pushed harder; on ambiguous events (high
variance, likely noisy labels) it backs off.

Eval metric (`val_loss_data`) is the standard UNWEIGHTED per-event NLL,
identical to every other run, so results compare apples-to-apples with
the standalone QConv2D_Max (-3.228/event).
"""
import tensorflow as tf
import tensorflow_probability as tfp

MINVAL = 1e-9
MAXVAL = 1e9


def _build_tril(p14):
    """(B,14) custom_loss layout -> (mu (B,4), scale_tril (B,4,4))."""
    mu = p14[:, 0:8:2]
    diag = MINVAL + tf.maximum(p14[:, 1:8:2], 0.0)
    off = p14[:, 8:]
    z = tf.zeros_like(diag[:, 0])
    row1 = tf.stack([diag[:, 0], z, z, z], axis=-1)
    row2 = tf.stack([off[:, 0], diag[:, 1], z, z], axis=-1)
    row3 = tf.stack([off[:, 1], off[:, 2], diag[:, 2], z], axis=-1)
    row4 = tf.stack([off[:, 3], off[:, 4], off[:, 5], diag[:, 3]], axis=-1)
    L = tf.stack([row1, row2, row3, row4], axis=-2)
    return mu, L


def nll_per_event(y, p14):
    """Per-event NLL (B,), same math/clip as loss.custom_loss but not summed."""
    mu, L = _build_tril(p14)
    dist = tfp.distributions.MultivariateNormalTriL(loc=mu, scale_tril=L)
    like = tf.clip_by_value(dist.prob(y), MINVAL, MAXVAL)
    return -tf.math.log(like)


def teacher_total_variance(t14):
    """Per-event total predictive variance (B,) from the teacher's 14-output:
    sum of the 4 marginal variances built from the Cholesky factor."""
    _, L = _build_tril(t14)
    # marginal variance of output j = sum_k L[j,k]^2  (rows of L)
    var = tf.reduce_sum(tf.square(L), axis=-1)   # (B, 4)
    return tf.reduce_sum(var, axis=-1)           # (B,)


class ConfWeightTrainer(tf.keras.Model):
    def __init__(self, student, teacher, clip_ratio=10.0, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self.teacher.trainable = False
        for w in self.teacher.weights:
            w._trainable = False
        self.clip_ratio = float(clip_ratio)
        self.loss_data_tracker = tf.keras.metrics.Mean(name='loss_data')      # unweighted NLL
        self.wloss_tracker = tf.keras.metrics.Mean(name='wloss')              # weighted NLL
        self.wspread_tracker = tf.keras.metrics.Mean(name='w_spread')         # max/min weight diag

    def call(self, x, training=False):
        return self.student(x, training=training)

    def _weights(self, x):
        t14 = self.teacher(x, training=False)
        tv = teacher_total_variance(t14)             # (B,)
        w = 1.0 / (tv + 1e-9)
        w = w / (tf.reduce_mean(w) + 1e-9)           # mean -> 1
        lo = 1.0 / self.clip_ratio
        w = tf.clip_by_value(w, lo, self.clip_ratio)
        w = w / (tf.reduce_mean(w) + 1e-9)           # renormalize mean -> 1
        return w

    def train_step(self, data):
        x, y = data
        w = self._weights(x)
        with tf.GradientTape() as tape:
            s14 = self.student(x, training=True)
            nll = nll_per_event(y, s14)              # (B,)
            wloss = tf.reduce_mean(w * nll)          # confidence-weighted
        grads = tape.gradient(wloss, self.student.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_weights))
        self.loss_data_tracker.update_state(tf.reduce_mean(nll))
        self.wloss_tracker.update_state(wloss)
        self.wspread_tracker.update_state(
            tf.reduce_max(w) / (tf.reduce_min(w) + 1e-9))
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        s14 = self.student(x, training=False)
        nll = nll_per_event(y, s14)
        self.loss_data_tracker.update_state(tf.reduce_mean(nll))
        return {'loss_data': self.loss_data_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_data_tracker, self.wloss_tracker, self.wspread_tracker]
