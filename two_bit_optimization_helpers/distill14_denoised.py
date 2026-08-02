"""
Denoised-mean distillation (the research's highest-leverage lever) for
direct-14-output students.

    L = NLL(y, mu_S, Sigma_S)                      # true-label NLL, weight 1.0 (bias control)
      + gamma * Huber(mu_S - mu_T)                 # teacher mean as a DENOISED target
      + beta  * KL_forward[ T(tau) || S ]          # temperature-softened covariance shaping

Why the mean term is the main driver: the teacher's predicted mean mu_T is a
lower-variance estimate of the true track parameters than the single noisy label
y (it is ~E[params|cluster] learned over the whole dataset). Pulling mu_S toward
mu_T sharpens the student mean WITHOUT the covariance-inflation side effect of
KL, while the true-label NLL then contracts Sigma_S -- both NLL terms improve.

Temperature: tau>1 inflates the teacher covariance (Sigma_T(tau)=tau^2 Sigma_T),
softening the razor-sharp teacher a tiny student cannot represent. Applied by
scaling the teacher Cholesky factor by tau; the mean is untouched.

Huber (smooth-L1) on the mean is robust to the rare events where the teacher is
off; optional teacher-bounded mask applies the pull only where the teacher's
per-event NLL beats the student's (trust the teacher only where it is right).

Designed for WARM-START use (load converged standalone weights first) so the
forward-KL term cannot explode. Eval metric val_loss_data = true-label per-event
NLL, comparable to every other run.
"""
import tensorflow as tf
import tensorflow_probability as tfp

from loss import custom_loss, custom_loss_perevent
from distill import make_tril


def _means(v14):
    return v14[..., 0:8:2]


def _forward_kl_temp(t14, s14, tau):
    """forward KL[ T(tau) || S ] with the teacher covariance softened by tau."""
    mu_T, L_T = make_tril(t14)
    mu_S, L_S = make_tril(s14)
    L_T = L_T * tau                       # Sigma_T -> tau^2 Sigma_T (mean unchanged)
    d_T = tfp.distributions.MultivariateNormalTriL(loc=mu_T, scale_tril=L_T)
    d_S = tfp.distributions.MultivariateNormalTriL(loc=mu_S, scale_tril=L_S)
    return tf.reduce_mean(tfp.distributions.kl_divergence(d_T, d_S))


class Distiller14Denoised(tf.keras.Model):
    def __init__(self, student, teacher, gamma=2.0, beta=0.3, tau=2.0,
                 huber_delta=1.0, teacher_bounded=False, margin=0.3, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self.teacher.trainable = False
        for w in self.teacher.weights:
            w._trainable = False
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.tau = float(tau)
        self.huber = tf.keras.losses.Huber(delta=float(huber_delta),
                                           reduction=tf.keras.losses.Reduction.NONE)
        self.teacher_bounded = bool(teacher_bounded)
        self.margin = float(margin)

        self.loss_data_tracker = tf.keras.metrics.Mean(name='loss_data')
        self.mean_tracker = tf.keras.metrics.Mean(name='mean_huber')
        self.kl_tracker = tf.keras.metrics.Mean(name='kl')

    def call(self, x, training=False):
        return self.student(x, training=training)

    def _terms(self, x, y, s14, t14):
        t14 = tf.stop_gradient(t14)
        mu_T = _means(t14)
        mu_S = _means(s14)
        # per-event Huber over the 4 mean dims -> (B,)
        he = self.huber(mu_T, mu_S)                       # (B,) reduced over last axis
        if self.teacher_bounded:
            s_nll = tf.stop_gradient(custom_loss_perevent(y, s14))
            t_nll = tf.stop_gradient(custom_loss_perevent(y, t14))
            mask = tf.cast(t_nll + self.margin < s_nll, tf.float32)
            mean_term = tf.reduce_mean(mask * he)
        else:
            mean_term = tf.reduce_mean(he)
        kl = _forward_kl_temp(t14, s14, self.tau)
        return mean_term, kl

    def train_step(self, data):
        x, y = data
        t14 = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            s14 = self.student(x, training=True)
            n = tf.cast(tf.shape(x)[0], tf.float32)
            data_loss = custom_loss(y, s14) / n
            mean_term, kl = self._terms(x, y, s14, t14)
            total = data_loss + self.gamma * mean_term + self.beta * kl
        grads = tape.gradient(total, self.student.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_weights))
        self.loss_data_tracker.update_state(data_loss)
        self.mean_tracker.update_state(mean_term)
        self.kl_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        t14 = self.teacher(x, training=False)
        s14 = self.student(x, training=False)
        n = tf.cast(tf.shape(x)[0], tf.float32)
        data_loss = custom_loss(y, s14) / n
        mean_term, kl = self._terms(x, y, s14, t14)
        self.loss_data_tracker.update_state(data_loss)
        self.mean_tracker.update_state(mean_term)
        self.kl_tracker.update_state(kl)
        return {'loss_data': self.loss_data_tracker.result(),
                'mean_huber': self.mean_tracker.result(),
                'kl': self.kl_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_data_tracker, self.mean_tracker, self.kl_tracker]
