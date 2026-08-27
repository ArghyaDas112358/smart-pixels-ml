"""
Teacher-Bounded Regression (TBR) mean distillation for direct-14-output
students. (Chen et al., arXiv:2002.12597, ported to a Gaussian-NLL head.)

Why this and not soft Gaussian-KL: the full multi-agent research + a numeric
teacher audit established that
  * the frozen ViT_Max teacher is genuinely strong AND well-calibrated on the
    EXACT hard {0,1,2,3} 2-bit input the student sees (-8.92 NLL/event vs the
    student's -6.10), so there is ~2.8 NLL/event of real transferable signal;
  * but soft forward-KL[T||S] precision-weights the mean error by the teacher's
    inverse covariance -- a sharp teacher makes that gradient pathological and it
    fights the data-NLL -- and it also forces the tiny student to copy a 4x4
    covariance it cannot represent. Every soft variant (forward/reverse KL,
    TAID, confidence-weighting, always-on means-MSE) lost to standalone.

TBR fixes both: transfer ONLY the teacher MEANS, and ONLY on events where the
teacher is actually better than the student right now. Once the student matches
or beats the teacher on an event, that event's distillation gradient switches
off and only the data-NLL remains. So in the limit this degenerates to
standalone training + a helpful nudge and, weighted sanely, CANNOT underperform
standalone. The student keeps fitting its own 4x4 Cholesky to ground truth via
the always-on data-NLL.

    mask_i   = 1[ teacher_NLL_i + margin < student_NLL_i ]      (stop-grad)
    tbr_i    = mean_d ( student_mean_i,d - teacher_mean_i,d )^2  (teacher detached)
    total    = data_NLL/n  +  w * mean_i ( mask_i * tbr_i )

The masked-mean denominator is the full batch B (not the masked count), so as
the student wins on more events the TBR term shrinks on its own -- a built-in
anneal, no schedule needed.

Eval metric (`val_loss_data`) is the standard unweighted per-event NLL,
directly comparable to standalone (-6.105) and every other run.
"""
import tensorflow as tf

from loss import custom_loss, custom_loss_perevent


def _means(v14):
    return v14[..., 0:8:2]   # (B, 4): x, y, cotA, cotB


class Distiller14TBR(tf.keras.Model):
    def __init__(self, student, teacher, w=1.0, margin=0.0, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self.teacher.trainable = False
        for wt in self.teacher.weights:
            wt._trainable = False
        self.w = float(w)
        self.margin = float(margin)

        self.loss_data_tracker = tf.keras.metrics.Mean(name='loss_data')
        self.tbr_tracker = tf.keras.metrics.Mean(name='tbr')
        self.frac_tracker = tf.keras.metrics.Mean(name='frac_teacher_wins')

    def call(self, x, training=False):
        return self.student(x, training=training)

    def _tbr_terms(self, x, y, s14, t14):
        """Returns (tbr_term, frac_masked). Teacher side fully detached."""
        t14 = tf.stop_gradient(t14)
        t_means = _means(t14)
        s_means = _means(s14)
        # per-event NLL for the hinge gate (both detached -- the mask is a
        # routing decision, never differentiated through)
        s_nll = tf.stop_gradient(custom_loss_perevent(y, s14))          # (B,)
        t_nll = tf.stop_gradient(custom_loss_perevent(y, t14))          # (B,)
        mask = tf.cast(t_nll + self.margin < s_nll, tf.float32)         # (B,) 1 = teacher wins
        per_event_se = tf.reduce_mean(tf.square(s_means - t_means), axis=-1)  # (B,)
        tbr_term = tf.reduce_mean(mask * per_event_se)                  # denom = B (self-anneal)
        frac = tf.reduce_mean(mask)
        return tbr_term, frac

    def train_step(self, data):
        x, y = data
        t14 = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            s14 = self.student(x, training=True)
            n = tf.cast(tf.shape(x)[0], tf.float32)
            data_loss = custom_loss(y, s14) / n
            tbr_term, frac = self._tbr_terms(x, y, s14, t14)
            total = data_loss + self.w * tbr_term
        grads = tape.gradient(total, self.student.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_weights))
        self.loss_data_tracker.update_state(data_loss)
        self.tbr_tracker.update_state(tbr_term)
        self.frac_tracker.update_state(frac)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        t14 = self.teacher(x, training=False)
        s14 = self.student(x, training=False)
        n = tf.cast(tf.shape(x)[0], tf.float32)
        data_loss = custom_loss(y, s14) / n
        tbr_term, frac = self._tbr_terms(x, y, s14, t14)
        self.loss_data_tracker.update_state(data_loss)
        self.tbr_tracker.update_state(tbr_term)
        self.frac_tracker.update_state(frac)
        return {'loss_data': self.loss_data_tracker.result(),
                'tbr': self.tbr_tracker.result(),
                'frac_teacher_wins': self.frac_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_data_tracker, self.tbr_tracker, self.frac_tracker]
