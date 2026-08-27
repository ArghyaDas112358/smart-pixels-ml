"""
E2: DeiT-style hard-target distillation with a SEPARATE distillation head, for
the QMlp_MoE_Max_Distill student (18 outputs = moe_14 ; distill_means_4).

DeiT (arXiv:2012.12877) found hard-target distillation through a SEPARATE
distillation token beats soft-KL for ViT students, and the averaged
class+distill prediction can beat the teacher. The single-head means-only run
we did earlier forced the ground-truth and teacher signals into the SAME mean
weights, which is exactly the conflict DeiT's separate head avoids.

Training:
    s18 = student(x);  s14 = s18[:, :14];  s_distill = s18[:, 14:18]
    data_loss = custom_loss(y, s14) / n              # main head fits GT (means + cov)
    mse       = mean( (s_distill - teacher_means)^2 ) # distill head fits teacher means
    total     = data_loss + w * mse                  # fixed w, no MDMM

Inference / eval (the deployed prediction, what EarlyStopping monitors):
    deployed means = 0.5 * (s14 means + s_distill)   # average the two heads (DeiT)
    deployed cov   = s14 covariance                  # only the main head predicts cov
    val_loss_data  = NLL(deployed_14, y) / n         # directly comparable to standalone

`val_main_only` (main head alone, no averaging) is also reported so we can pick
whichever is better at the end.
"""
import tensorflow as tf

from loss import custom_loss


def _means(v):
    return v[..., 0:8:2]          # (B,4) from a 14-vector


def _reassemble14(means, src14):
    """14-vector with `means` (B,4) in the mean slots and the diag/off-diag
    taken from src14 (B,14). Layout: [mu0,d0,mu1,d1,mu2,d2,mu3,d3, off0..off5]."""
    diag = src14[..., 1:8:2]                              # (B,4)
    off = src14[..., 8:14]                                # (B,6)
    interleaved = tf.reshape(tf.stack([means, diag], axis=-1),
                             tf.concat([tf.shape(means)[:-1], [8]], axis=0))  # (B,8)
    return tf.concat([interleaved, off], axis=-1)         # (B,14)


class Distiller14DeiT(tf.keras.Model):
    def __init__(self, student, teacher, w=0.5, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self.teacher.trainable = False
        for wt in self.teacher.weights:
            wt._trainable = False
        self.w = float(w)

        self.loss_data_tracker = tf.keras.metrics.Mean(name='loss_data')
        self.mse_tracker = tf.keras.metrics.Mean(name='mse_distill')

    def call(self, x, training=False):
        return self.student(x, training=training)

    def _split(self, s18):
        return s18[..., :14], s18[..., 14:18]

    def train_step(self, data):
        x, y = data
        t_means = tf.stop_gradient(_means(self.teacher(x, training=False)))
        with tf.GradientTape() as tape:
            s18 = self.student(x, training=True)
            s14, s_distill = self._split(s18)
            n = tf.cast(tf.shape(x)[0], tf.float32)
            data_loss = custom_loss(y, s14) / n
            mse = tf.reduce_mean(tf.square(s_distill - t_means))
            total = data_loss + self.w * mse
        grads = tape.gradient(total, self.student.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_weights))
        self.loss_data_tracker.update_state(data_loss)
        self.mse_tracker.update_state(mse)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        t_means = tf.stop_gradient(_means(self.teacher(x, training=False)))
        s18 = self.student(x, training=False)
        s14, s_distill = self._split(s18)
        n = tf.cast(tf.shape(x)[0], tf.float32)
        main_only = custom_loss(y, s14) / n
        avg_means = 0.5 * (_means(s14) + s_distill)
        deployed14 = _reassemble14(avg_means, s14)
        deployed = custom_loss(y, deployed14) / n
        mse = tf.reduce_mean(tf.square(s_distill - t_means))
        self.loss_data_tracker.update_state(deployed)
        self.mse_tracker.update_state(mse)
        return {'loss_data': self.loss_data_tracker.result(),     # deployed (monitored)
                'main_only': main_only,
                'mse_distill': self.mse_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_data_tracker, self.mse_tracker]
