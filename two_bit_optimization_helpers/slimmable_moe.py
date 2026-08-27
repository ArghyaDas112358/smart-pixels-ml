"""
Slimmable / in-place-distillation MoE (flexible-teacher chain in its strongest
form; Yu 2019 US-Nets sandwich rule + Gaussian-KL in-place KD + a frozen ViT_Max
apex teacher).

ONE weight-shared super-net runs at several WIDTHS; a narrow width uses a PREFIX
of every weight of the wide width, so the small deployable slice literally
shares weights and gradients with the big in-net assistant. Each step:
  - the BIG width trains on data NLL + forward-KL from the frozen ViT_Max apex,
  - every narrower width trains on data NLL + forward-KL from the big width's
    DETACHED Gaussian (in-place distillation).
Chain:  ViT_Max(419K) -> big slice -> ... -> tiny deploy slice.

A width is (n_experts, hidden, emb, proj): proj is the slimmable width of the two
fixed 1D-marginal projections, which lets the smallest slice reach ~1K params
(the projections were the previous ~1K floor). Output is always the 14-vector.
Float weights; deployable slice can be re-quantized to 8-bit afterward.
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from qkeras import quantized_bits, quantized_tanh

from loss import custom_loss
from distill import gaussian_kl, make_tril


def _forward_kl_temp(t14, s14, tau):
    """forward KL[ T(tau) || S ] with the teacher covariance softened by tau
    (scale teacher Cholesky by tau -> Sigma_T -> tau^2 Sigma_T, mean unchanged).
    A softened teacher is far more stable: the student is not pushed to match the
    razor-sharp teacher covariance and collapse to a singular one."""
    mu_T, L_T = make_tril(t14)
    mu_S, L_S = make_tril(s14)
    d_T = tfp.distributions.MultivariateNormalTriL(loc=mu_T, scale_tril=L_T * tau)
    d_S = tfp.distributions.MultivariateNormalTriL(loc=mu_S, scale_tril=L_S)
    return tf.reduce_mean(tfp.distributions.kl_divergence(d_T, d_S))

# (n_experts, hidden, emb, proj) -- each entry must be a prefix of MAX below.
DEFAULT_WIDTHS = [
    (2,  8,  8,  6),    # ~1.06K  (smallest deployable)
    (3,  12, 12, 8),    # ~2.25K
    (4,  16, 16, 16),   # ~4.78K
    (8,  32, 32, 16),   # ~22.97K
    (12, 48, 48, 16),   # ~67.9K
    (16, 64, 64, 16),   # ~151.9K (big in-net assistant; gets the ViT apex)
]
MAX_N, MAX_H, MAX_EMB, MAX_PROJ = 16, 64, 64, 16


def slice_params(n, h, emb, proj):
    enc = (32*proj + proj)*2 + (proj*emb)*2 + emb        # Wx,Wy + Wex,Wey + be
    exp = n*(emb*h + h + h*h + h + h*14 + 14)
    gate = emb*n + n
    return enc + exp + gate


def _diag_pos_bias(max_n):
    b = np.zeros((max_n, 14), np.float32)
    b[:, 1:8:2] = 0.5
    return b


class SlimmableMoE(tf.keras.Model):
    def __init__(self, widths=DEFAULT_WIDTHS, lam_kd=1.0,
                 teacher=None, lam_vit=0.3, vit_warmup_steps=200, vit_tau=2.0,
                 quantize=False, **kwargs):
        super().__init__(**kwargs)
        self.widths = [tuple(w) for w in widths]
        self.lam_kd = float(lam_kd)
        self.vit_tau = float(vit_tau)
        # 8-bit QAT: same scheme as the QMlp models (quantized_bits(8,0) on every
        # weight/bias, quantized_tanh(8,0,1) on every activation). Fake-quant with
        # straight-through gradient, so the deployed weights are genuinely 8-bit.
        self.quantize = bool(quantize)
        if self.quantize:
            self.wq = quantized_bits(8, 0, alpha=1)
            self.aq = quantized_tanh(8, 0, 1)
        self.teacher = teacher
        if teacher is not None:
            teacher.trainable = False
            for w in teacher.weights:
                w._trainable = False
        self.lam_vit = float(lam_vit)
        self.vit_warmup_steps = int(vit_warmup_steps)
        self._step = tf.Variable(0, dtype=tf.int64, trainable=False, name='step')
        gi = tf.keras.initializers.GlorotUniform(seed=0)

        self.Wx = self.add_weight('Wx', (32, MAX_PROJ), initializer=gi)
        self.bx = self.add_weight('bx', (MAX_PROJ,), initializer='zeros')
        self.Wy = self.add_weight('Wy', (32, MAX_PROJ), initializer=gi)
        self.by = self.add_weight('by', (MAX_PROJ,), initializer='zeros')
        self.Wex = self.add_weight('Wex', (MAX_PROJ, MAX_EMB), initializer=gi)
        self.Wey = self.add_weight('Wey', (MAX_PROJ, MAX_EMB), initializer=gi)
        self.be = self.add_weight('be', (MAX_EMB,), initializer='zeros')
        self.D1 = self.add_weight('D1', (MAX_N, MAX_EMB, MAX_H), initializer=gi)
        self.b1 = self.add_weight('b1', (MAX_N, MAX_H), initializer='zeros')
        self.D2 = self.add_weight('D2', (MAX_N, MAX_H, MAX_H), initializer=gi)
        self.b2 = self.add_weight('b2', (MAX_N, MAX_H), initializer='zeros')
        self.OUT = self.add_weight('OUT', (MAX_N, MAX_H, 14), initializer=gi)
        self.ob = self.add_weight('ob', (MAX_N, 14),
                                  initializer=tf.keras.initializers.Constant(_diag_pos_bias(MAX_N)))
        self.GW = self.add_weight('GW', (MAX_EMB, MAX_N), initializer=gi)
        self.gb = self.add_weight('gb', (MAX_N,), initializer='zeros')
        self.trackers = {}

    def _q(self, w):
        return self.wq(w) if self.quantize else w

    def _act(self, z):
        return self.aq(z) if self.quantize else tf.tanh(z)

    def forward_cfg(self, x, cfg):
        n, h, emb, proj = cfg
        Q, A = self._q, self._act
        pool_x = tf.reshape(tf.reduce_mean(x, axis=2), (-1, 32))
        pool_y = tf.reshape(tf.reduce_mean(x, axis=1), (-1, 32))
        px = A(pool_x @ Q(self.Wx[:, :proj]) + Q(self.bx[:proj]))               # (B,proj)
        py = A(pool_y @ Q(self.Wy[:, :proj]) + Q(self.by[:proj]))
        e = A(px @ Q(self.Wex[:proj, :emb]) + py @ Q(self.Wey[:proj, :emb]) + Q(self.be[:emb]))  # (B,emb)
        h1 = A(tf.einsum('be,neh->bnh', e, Q(self.D1[:n, :emb, :h])) + Q(self.b1[:n, :h]))
        h2 = A(tf.einsum('bnh,nhg->bng', h1, Q(self.D2[:n, :h, :h])) + Q(self.b2[:n, :h]))
        o = tf.einsum('bnh,nho->bno', h2, Q(self.OUT[:n, :h, :])) + Q(self.ob[:n, :])  # (B,n,14) linear out
        gate = tf.nn.softmax(e @ Q(self.GW[:emb, :n]) + Q(self.gb[:n]), axis=-1)
        return tf.einsum('bn,bno->bo', gate, o)

    def call(self, x, training=False):
        return self.forward_cfg(x, self.widths[0])

    def _tracker(self, name):
        if name not in self.trackers:
            self.trackers[name] = tf.keras.metrics.Mean(name=name)
        return self.trackers[name]

    def train_step(self, data):
        x, y = data
        n = tf.cast(tf.shape(x)[0], tf.float32)
        t14 = None if self.teacher is None else tf.stop_gradient(self.teacher(x, training=False))
        with tf.GradientTape() as tape:
            big14 = self.forward_cfg(x, self.widths[-1])
            loss_big = custom_loss(y, big14) / n
            if t14 is not None:
                warm = tf.cast(self._step >= self.vit_warmup_steps, tf.float32)
                loss_big += warm * self.lam_vit * _forward_kl_temp(t14, big14, self.vit_tau)
            big_det = tf.stop_gradient(big14)
            total = loss_big
            for cfg in self.widths[:-1]:
                s14 = self.forward_cfg(x, cfg)
                total += custom_loss(y, s14) / n + self.lam_kd * gaussian_kl(big_det, s14)
        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self._step.assign_add(1)
        logs = {}
        for i, cfg in enumerate(self.widths):
            t = self._tracker(f'nll_w{i}'); t.update_state(custom_loss(y, self.forward_cfg(x, cfg)) / n)
            logs[f'nll_w{i}'] = t.result()
        return logs

    def test_step(self, data):
        x, y = data
        n = tf.cast(tf.shape(x)[0], tf.float32)
        out = {}
        for i, cfg in enumerate(self.widths):
            t = self._tracker(f'val_nll_w{i}'); t.update_state(custom_loss(y, self.forward_cfg(x, cfg)) / n)
            out[f'nll_w{i}'] = t.result()
        # Monitor the BIG (lead) slice for early-stopping, NOT the tiny one: the
        # tiny slice plateaus first and would stop the run before the big slice
        # (taught by the ViT) converges. As the big slice improves, the small
        # slices' in-place target improves too, so the whole chain trains fully.
        out['loss_data'] = out[f'nll_w{len(self.widths) - 1}']
        return out

    @property
    def metrics(self):
        return list(self.trackers.values())
