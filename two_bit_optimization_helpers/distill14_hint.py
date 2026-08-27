"""
Hint / FitNet-style feature distillation for direct-14-output students
(QConv2D_Max etc.). Extends the output-only Distiller14 with a
feature-matching term between the teacher's penultimate representation
and a learnable projection of the student's penultimate representation.

    total = data_loss + lambda * KL[T||S] + beta * MSE(Proj(h_S), h_T)

where
    h_T = teacher penultimate features  (ViT_Max: 64-dim Dense(relu))
    h_S = student penultimate features  (QConv2D_Max: 16-dim quantized_tanh)
    Proj = trainable Dense(dim(h_T)) mapping h_S -> teacher feature space
           (only needed because the architectures differ; the projection
            weights are NOT part of the deployed student).

The penultimate tensors are tapped via `model.layers[-1].input`, which for
both Functional models is the activation feeding the final output layer.
No change to the model factory is required.

This is the cross-architecture recipe from Liu et al. 2022
(arXiv 2207.05273): project the student into the teacher's feature space
and match there, rather than mimicking features directly.
"""
import tensorflow as tf

from loss import custom_loss
from distill import gaussian_kl


class Distiller14Hint(tf.keras.Model):
    def __init__(self, student, teacher,
                 beta_hint=1.0,
                 lambda_init=0.0, mdmm_eta=1e-3, kl_target=0.0,
                 warmup_steps=0, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self.teacher.trainable = False
        for w in self.teacher.weights:
            w._trainable = False

        # Penultimate-feature taps (share layers with the originals).
        self.teacher_tap = tf.keras.Model(
            teacher.input, [teacher.output, teacher.layers[-1].input],
            name='teacher_tap')
        self.teacher_tap.trainable = False
        self.student_tap = tf.keras.Model(
            student.input, [student.output, student.layers[-1].input],
            name='student_tap')

        # Dims of the tapped hidden tensors.
        t_dim = self.teacher_tap.output_shape[1][-1]
        s_dim = self.student_tap.output_shape[1][-1]
        # Learnable projection student-hidden -> teacher-hidden space.
        self.proj = tf.keras.layers.Dense(t_dim, name='hint_projection')
        self.proj.build((None, s_dim))
        self._t_dim, self._s_dim = t_dim, s_dim

        self.beta_hint = float(beta_hint)
        self.lam = tf.Variable(lambda_init, dtype=tf.float32,
                               trainable=False, name='mdmm_lambda')
        self.mdmm_eta = float(mdmm_eta)
        self.kl_target = float(kl_target)
        self.warmup_steps = int(warmup_steps)
        self._step = tf.Variable(0, dtype=tf.int64, trainable=False, name='train_step')

        self.loss_data_tracker = tf.keras.metrics.Mean(name='loss_data')
        self.kl_tracker = tf.keras.metrics.Mean(name='kl')
        self.hint_tracker = tf.keras.metrics.Mean(name='hint')
        self.lam_tracker = tf.keras.metrics.Mean(name='lam')

    def call(self, x, training=False):
        return self.student(x, training=training)

    @property
    def _student_trainables(self):
        # student weights + projection weights (proj does not deploy)
        return self.student.trainable_weights + self.proj.trainable_weights

    def train_step(self, data):
        x, y = data
        t14, h_t = self.teacher_tap(x, training=False)
        with tf.GradientTape() as tape:
            s14, h_s = self.student_tap(x, training=True)
            n = tf.cast(tf.shape(x)[0], tf.float32)
            data_loss = custom_loss(y, s14) / n
            kl = gaussian_kl(t14, s14)
            h_s_proj = self.proj(h_s)
            hint = tf.reduce_mean(tf.square(h_s_proj - tf.stop_gradient(h_t)))
            warm = tf.cast(self._step >= self.warmup_steps, tf.float32)
            total = data_loss + warm * self.lam * kl + self.beta_hint * hint
        grads = tape.gradient(total, self._student_trainables)
        self.optimizer.apply_gradients(zip(grads, self._student_trainables))
        delta = tf.clip_by_value(self.mdmm_eta * (kl - self.kl_target), -1.0, 1.0)
        self.lam.assign(tf.maximum(0.0, self.lam + delta))
        self._step.assign_add(1)
        self.loss_data_tracker.update_state(data_loss)
        self.kl_tracker.update_state(kl)
        self.hint_tracker.update_state(hint)
        self.lam_tracker.update_state(self.lam)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        t14, h_t = self.teacher_tap(x, training=False)
        s14, h_s = self.student_tap(x, training=False)
        n = tf.cast(tf.shape(x)[0], tf.float32)
        data_loss = custom_loss(y, s14) / n
        kl = gaussian_kl(t14, s14)
        hint = tf.reduce_mean(tf.square(self.proj(h_s) - h_t))
        self.loss_data_tracker.update_state(data_loss)
        self.kl_tracker.update_state(kl)
        self.hint_tracker.update_state(hint)
        return {'loss_data': self.loss_data_tracker.result(),
                'kl': self.kl_tracker.result(),
                'hint': self.hint_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_data_tracker, self.kl_tracker,
                self.hint_tracker, self.lam_tracker]
