"""
Distillation for students that DIRECTLY emit the 14-output Max layout
(QConv2D_Max, Conv2D_Max, etc.), instead of the two-headed
(unsigned_means, chol) layout the symbolic+tiny-NN students use.

No sign borrowing: the student is expected to predict signed means
itself. Same MDMM dual ascent on lambda as Distiller, same Gaussian KL.
"""
import tensorflow as tf
import tensorflow_probability as tfp

from loss import custom_loss
from distill import gaussian_kl  # reuse the existing KL helper


class Distiller14(tf.keras.Model):
    def __init__(self, student, teacher,
                 lambda_init=0.0, mdmm_eta=1e-3, kl_target=0.0,
                 warmup_steps=0, kl_reverse=False, fixed_beta=None, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        for w in self.teacher.weights:
            w._trainable = False
        self.teacher.trainable = False

        self.kl_reverse = bool(kl_reverse)
        # fixed_beta: if set, use a constant KD weight instead of MDMM's lambda
        # (cleaner for reverse-KL experiments; avoids lambda runaway).
        self.fixed_beta = None if fixed_beta is None else float(fixed_beta)
        self.lam = tf.Variable(lambda_init, dtype=tf.float32,
                               trainable=False, name='mdmm_lambda')
        self.mdmm_eta = float(mdmm_eta)
        self.kl_target = float(kl_target)
        self.warmup_steps = int(warmup_steps)
        self._step = tf.Variable(0, dtype=tf.int64, trainable=False, name='train_step')

        self.loss_data_tracker = tf.keras.metrics.Mean(name='loss_data')
        self.kl_tracker = tf.keras.metrics.Mean(name='kl')
        self.lam_tracker = tf.keras.metrics.Mean(name='lam')

    def call(self, x, training=False):
        return self.student(x, training=training)

    def _forward(self, x, training):
        t14 = self.teacher(x, training=False)
        s14 = self.student(x, training=training)
        return t14, s14

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            t14, s14 = self._forward(x, training=True)
            n = tf.cast(tf.shape(x)[0], tf.float32)
            data_loss = custom_loss(y, s14) / n
            kl = gaussian_kl(t14, s14, reverse=self.kl_reverse)
            warm = tf.cast(self._step >= self.warmup_steps, tf.float32)
            if self.fixed_beta is not None:
                total = data_loss + warm * self.fixed_beta * kl
            else:
                total = data_loss + warm * self.lam * kl
        grads = tape.gradient(total, self.student.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_weights))
        if self.fixed_beta is None:
            delta = tf.clip_by_value(self.mdmm_eta * (kl - self.kl_target), -1.0, 1.0)
            self.lam.assign(tf.maximum(0.0, self.lam + delta))
        self._step.assign_add(1)
        self.loss_data_tracker.update_state(data_loss)
        self.kl_tracker.update_state(kl)
        self.lam_tracker.update_state(self.lam)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        t14, s14 = self._forward(x, training=False)
        n = tf.cast(tf.shape(x)[0], tf.float32)
        data_loss = custom_loss(y, s14) / n
        kl = gaussian_kl(t14, s14, reverse=self.kl_reverse)
        self.loss_data_tracker.update_state(data_loss)
        self.kl_tracker.update_state(kl)
        return {'loss_data': self.loss_data_tracker.result(),
                'kl': self.kl_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_data_tracker, self.kl_tracker, self.lam_tracker]
