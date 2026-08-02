"""
TAID-style distillation (Temporally Adaptive Interpolated Distillation,
arXiv 2501.16937) for direct-14-output students.

The fixed-KL experiment showed the tiny student cannot match the sharp
418K-param teacher: KL gets stuck ~2.1, MDMM cranks lambda to ~191, and
forcing the infeasible KL=0 constraint crushes the data fit.

TAID avoids the infeasible jump. Instead of distilling toward the full
teacher, it distills toward an INTERPOLATED target that starts at the
student's own current distribution and is gradually annealed toward the
teacher:

    target_14 = (1 - alpha) * stopgrad(student_14) + alpha * teacher_14
    alpha: 0 -> 1 linearly over `anneal_steps`

At alpha=0 the target IS the student (KL=0, no pull). As alpha grows the
target inches toward the teacher, so the student is always chasing a
reachable, nearby target. No MDMM needed -- a fixed beta on the KD term.

Eval metric (`val_loss_data`) is the standard unweighted per-event NLL,
comparable to every other run.
"""
import tensorflow as tf
import tensorflow_probability as tfp

from loss import custom_loss
from distill import gaussian_kl, make_tril


def _interp_target_dist(s14, t14, alpha):
    """TAID target as a Gaussian interpolated in DISTRIBUTION space:
    mu_t  = (1-a) mu_S + a mu_T   (student side detached)
    Sig_t = (1-a) Sig_S + a Sig_T  (blend COVARIANCES, not Cholesky entries)
    then recompute scale_tril via cholesky. This is the correct TAID
    interpolation (arXiv:2501.16937); the previous raw-14-param blend was wrong.
    Returns a tfp MultivariateNormalTriL.
    """
    mu_S, L_S = make_tril(tf.stop_gradient(s14))
    mu_T, L_T = make_tril(t14)
    Sig_S = tf.matmul(L_S, L_S, transpose_b=True)
    Sig_T = tf.matmul(L_T, L_T, transpose_b=True)
    mu_t = (1.0 - alpha) * mu_S + alpha * mu_T
    Sig_t = (1.0 - alpha) * Sig_S + alpha * Sig_T
    eye = tf.eye(4, batch_shape=tf.shape(Sig_t)[:1]) * 1e-6
    L_t = tf.linalg.cholesky(Sig_t + eye)
    return tfp.distributions.MultivariateNormalTriL(loc=mu_t, scale_tril=L_t)


class Distiller14TAID(tf.keras.Model):
    def __init__(self, student, teacher,
                 beta=1.0, anneal_steps=20000,
                 dist_space=True, kl_reverse=True, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self.teacher.trainable = False
        for w in self.teacher.weights:
            w._trainable = False

        # dist_space=True: interpolate (mu, Sigma) in distribution space (correct
        #   TAID). False: legacy raw-14-param blend (kept for ablation).
        # kl_reverse=True: minimize reverse KL[student || target] (mode-seeking).
        self.dist_space = bool(dist_space)
        self.kl_reverse = bool(kl_reverse)
        self.beta = float(beta)
        self.anneal_steps = float(anneal_steps)
        self._step = tf.Variable(0.0, dtype=tf.float32, trainable=False, name='train_step')

        self.loss_data_tracker = tf.keras.metrics.Mean(name='loss_data')
        self.kl_tracker = tf.keras.metrics.Mean(name='kl')
        self.alpha_tracker = tf.keras.metrics.Mean(name='alpha')

    def call(self, x, training=False):
        return self.student(x, training=training)

    def train_step(self, data):
        x, y = data
        t14 = self.teacher(x, training=False)
        alpha = tf.minimum(1.0, self._step / self.anneal_steps)
        with tf.GradientTape() as tape:
            s14 = self.student(x, training=True)
            n = tf.cast(tf.shape(x)[0], tf.float32)
            data_loss = custom_loss(y, s14) / n
            if self.dist_space:
                # correct TAID: interpolate (mu, Sigma) in distribution space
                d_target = _interp_target_dist(s14, t14, alpha)
                mu_S, L_S = make_tril(s14)
                d_S = tfp.distributions.MultivariateNormalTriL(loc=mu_S, scale_tril=L_S)
                if self.kl_reverse:   # reverse KL[student || target]
                    kl = tf.reduce_mean(tfp.distributions.kl_divergence(d_S, d_target))
                else:
                    kl = tf.reduce_mean(tfp.distributions.kl_divergence(d_target, d_S))
            else:
                # legacy raw-14-param blend (ablation)
                target14 = (1.0 - alpha) * tf.stop_gradient(s14) + alpha * t14
                kl = gaussian_kl(target14, s14, reverse=self.kl_reverse)
            total = data_loss + self.beta * kl
        grads = tape.gradient(total, self.student.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_weights))
        self._step.assign_add(1.0)
        self.loss_data_tracker.update_state(data_loss)
        self.kl_tracker.update_state(kl)
        self.alpha_tracker.update_state(alpha)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        t14 = self.teacher(x, training=False)
        s14 = self.student(x, training=False)
        n = tf.cast(tf.shape(x)[0], tf.float32)
        data_loss = custom_loss(y, s14) / n
        # eval KL is to the FULL teacher (alpha=1), for monitoring
        kl = gaussian_kl(t14, s14)
        self.loss_data_tracker.update_state(data_loss)
        self.kl_tracker.update_state(kl)
        return {'loss_data': self.loss_data_tracker.result(),
                'kl': self.kl_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_data_tracker, self.kl_tracker, self.alpha_tracker]
