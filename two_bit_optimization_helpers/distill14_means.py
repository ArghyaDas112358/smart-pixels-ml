"""
Means-only distillation for direct-14-output students.

Diagnosis from the full sweep: the tiny student cannot match the teacher's
full predictive distribution (KL stuck ~2; forcing it via MDMM/KL or TAID
hurts). But the teacher's predicted MEANS may still be a useful soft
target. This recipe transfers ONLY the 4 means and lets the student learn
its own covariance from the data NLL -- decoupling the (reachable) mean
transfer from the (unreachable) covariance match.

    total = data_NLL + beta * MSE(teacher_means, student_means)

teacher_means / student_means are the 4 mean slots (indices 0,2,4,6) of the
14-output vector. Optionally anneal beta down late so the data NLL takes
over (the teacher's means are at best as good as the true labels).

Eval metric (`val_loss_data`) is the standard unweighted per-event NLL.
"""
import tensorflow as tf

from loss import custom_loss


def _means(v14):
    return v14[..., 0:8:2]   # (B, 4): x, y, cotA, cotB


class Distiller14Means(tf.keras.Model):
    def __init__(self, student, teacher, beta=1.0, anneal_steps=0, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self.teacher.trainable = False
        for w in self.teacher.weights:
            w._trainable = False
        self.beta = float(beta)
        self.anneal_steps = float(anneal_steps)   # 0 = constant beta
        self._step = tf.Variable(0.0, dtype=tf.float32, trainable=False, name='step')

        self.loss_data_tracker = tf.keras.metrics.Mean(name='loss_data')
        self.mse_tracker = tf.keras.metrics.Mean(name='mse_means')
        self.beta_tracker = tf.keras.metrics.Mean(name='beta_eff')

    def call(self, x, training=False):
        return self.student(x, training=training)

    def _beta_eff(self):
        if self.anneal_steps <= 0:
            return self.beta
        # linearly decay beta -> 0 over anneal_steps (teacher fades, data takes over)
        frac = tf.maximum(0.0, 1.0 - self._step / self.anneal_steps)
        return self.beta * frac

    def train_step(self, data):
        x, y = data
        t_means = _means(self.teacher(x, training=False))
        beta = self._beta_eff()
        with tf.GradientTape() as tape:
            s14 = self.student(x, training=True)
            n = tf.cast(tf.shape(x)[0], tf.float32)
            data_loss = custom_loss(y, s14) / n
            mse = tf.reduce_mean(tf.square(_means(s14) - t_means))
            total = data_loss + beta * mse
        grads = tape.gradient(total, self.student.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_weights))
        self._step.assign_add(1.0)
        self.loss_data_tracker.update_state(data_loss)
        self.mse_tracker.update_state(mse)
        self.beta_tracker.update_state(beta)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        t_means = _means(self.teacher(x, training=False))
        s14 = self.student(x, training=False)
        n = tf.cast(tf.shape(x)[0], tf.float32)
        data_loss = custom_loss(y, s14) / n
        mse = tf.reduce_mean(tf.square(_means(s14) - t_means))
        self.loss_data_tracker.update_state(data_loss)
        self.mse_tracker.update_state(mse)
        return {'loss_data': self.loss_data_tracker.result(),
                'mse_means': self.mse_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_data_tracker, self.mse_tracker, self.beta_tracker]
