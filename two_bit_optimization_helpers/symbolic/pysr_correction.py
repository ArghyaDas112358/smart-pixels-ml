"""
PySR-discovered correction terms transcribed to TF, used by the
PhysicsAnsatz(variant='pysr_aug') branch.

Source equations were discovered by physics-informed PySR on
(teacher_means - barycenter_means) residuals (LABEL UNITS) over the 3sr
centeredIncidence dataset, using per-output templates:

  x: base(rx, skew_x, kurt_x, q_frac_max, wx) + lorentz(tasym, tasym_x, cota_geom)
     + edge(edge_x, wx)
  y: same with y variants
  cot_a: sat(cota_geom, wx, q_frac_max) + asym(rx, tasym)
  cot_b: sat(cotb_geom, wy, q_frac_max) + asym(ry, tasym_y)

The discovered equations are baked in here with literal constants exposed as
trainable tf.Variables. SGD can refine them during distillation.
"""
import tensorflow as tf
from tensorflow.keras import layers

# Geometry constants matching ansatz.py.
P_X = 50.0
P_Y = 12.5
T_SENSOR = 100.0
N = 16


def _pixel_centers_tf(n, pitch):
    idx = tf.range(n, dtype=tf.float32)
    return (idx - (n - 1) / 2.0) * pitch


def cluster_features_tf(charge):
    """Compute the 24-feature dict used by the PySR equations.

    charge: (B, 16, 16, 2). Axis 1 = x (pitch 50um), axis 2 = y (pitch 12.5um).
    Mirrors `cluster_features_physics` in pysr_residuals_physics.py.
    """
    q = tf.reduce_sum(charge, axis=-1)               # (B, H, W)
    prof_x = tf.reduce_sum(q, axis=2)                # (B, H)
    prof_y = tf.reduce_sum(q, axis=1)                # (B, W)
    active_x = tf.cast(prof_x > 0, tf.float32)
    active_y = tf.cast(prof_y > 0, tf.float32)

    wx = tf.reduce_sum(active_x, axis=1)
    wy = tf.reduce_sum(active_y, axis=1)
    cota_geom = tf.maximum(wx - 1.0, 0.0) * P_X / T_SENSOR
    cotb_geom = tf.maximum(wy - 1.0, 0.0) * P_Y / T_SENSOR

    q_tot = tf.reduce_sum(q, axis=[1, 2])
    q_max = tf.reduce_max(q, axis=[1, 2])
    q_frac_max = q_max / (q_tot + 1e-9)

    # head/tail charges on each profile
    first_x = tf.argmax(active_x, axis=1, output_type=tf.int32)
    last_x = (N - 1) - tf.argmax(tf.reverse(active_x, axis=[1]), axis=1, output_type=tf.int32)
    first_y = tf.argmax(active_y, axis=1, output_type=tf.int32)
    last_y = (N - 1) - tf.argmax(tf.reverse(active_y, axis=[1]), axis=1, output_type=tf.int32)
    qFx = tf.gather(prof_x, first_x, batch_dims=1)
    qLx = tf.gather(prof_x, last_x, batch_dims=1)
    qFy = tf.gather(prof_y, first_y, batch_dims=1)
    qLy = tf.gather(prof_y, last_y, batch_dims=1)
    rx = (qLx - qFx) / (qLx + qFx + 1e-9)
    ry = (qLy - qFy) / (qLy + qFy + 1e-9)

    # moments
    x_centers = _pixel_centers_tf(N, P_X)            # (N,)
    y_centers = _pixel_centers_tf(N, P_Y)
    sum_x = tf.reduce_sum(prof_x, axis=1) + 1e-9
    sum_y = tf.reduce_sum(prof_y, axis=1) + 1e-9
    xbary = tf.reduce_sum(prof_x * x_centers, axis=1) / sum_x
    ybary = tf.reduce_sum(prof_y * y_centers, axis=1) / sum_y
    diff_x = x_centers[None, :] - xbary[:, None]
    diff_y = y_centers[None, :] - ybary[:, None]
    var_x = tf.reduce_sum(prof_x * diff_x * diff_x, axis=1) / sum_x
    var_y = tf.reduce_sum(prof_y * diff_y * diff_y, axis=1) / sum_y
    std_x = tf.sqrt(var_x + 1e-9)
    std_y = tf.sqrt(var_y + 1e-9)
    skew_x = tf.reduce_sum(prof_x * tf.pow(diff_x, 3), axis=1) / sum_x / (tf.pow(std_x, 3) + 1e-9)
    skew_y = tf.reduce_sum(prof_y * tf.pow(diff_y, 3), axis=1) / sum_y / (tf.pow(std_y, 3) + 1e-9)
    kurt_x = tf.reduce_sum(prof_x * tf.pow(diff_x, 4), axis=1) / sum_x / (tf.pow(std_x, 4) + 1e-9) - 3.0
    kurt_y = tf.reduce_sum(prof_y * tf.pow(diff_y, 4), axis=1) / sum_y / (tf.pow(std_y, 4) + 1e-9) - 3.0

    # time-resolved drift centroids
    q_t0 = tf.reduce_sum(charge[..., 0], axis=[1, 2])
    q_t1 = tf.reduce_sum(charge[..., 1], axis=[1, 2])
    tasym = (q_t1 - q_t0) / (q_t1 + q_t0 + 1e-9)
    prof_x_t0 = tf.reduce_sum(charge[..., 0], axis=2)
    prof_x_t1 = tf.reduce_sum(charge[..., 1], axis=2)
    prof_y_t0 = tf.reduce_sum(charge[..., 0], axis=1)
    prof_y_t1 = tf.reduce_sum(charge[..., 1], axis=1)
    cx_t0 = tf.reduce_sum(prof_x_t0 * x_centers, axis=1) / (tf.reduce_sum(prof_x_t0, axis=1) + 1e-9)
    cx_t1 = tf.reduce_sum(prof_x_t1 * x_centers, axis=1) / (tf.reduce_sum(prof_x_t1, axis=1) + 1e-9)
    cy_t0 = tf.reduce_sum(prof_y_t0 * y_centers, axis=1) / (tf.reduce_sum(prof_y_t0, axis=1) + 1e-9)
    cy_t1 = tf.reduce_sum(prof_y_t1 * y_centers, axis=1) / (tf.reduce_sum(prof_y_t1, axis=1) + 1e-9)
    tasym_x = cx_t1 - cx_t0
    tasym_y = cy_t1 - cy_t0

    edge_x = tf.cast((first_x == 0) | (last_x == N - 1), tf.float32)
    edge_y = tf.cast((first_y == 0) | (last_y == N - 1), tf.float32)

    return dict(
        xbary=xbary, ybary=ybary,
        wx=wx, wy=wy,
        cota_geom=cota_geom, cotb_geom=cotb_geom,
        q_tot=q_tot, q_max=q_max, q_frac_max=q_frac_max,
        qFx=qFx, qLx=qLx, qFy=qFy, qLy=qLy,
        rx=rx, ry=ry,
        skew_x=skew_x, skew_y=skew_y, kurt_x=kurt_x, kurt_y=kurt_y,
        tasym=tasym, tasym_x=tasym_x, tasym_y=tasym_y,
        edge_x=edge_x, edge_y=edge_y,
    )


class PySRCorrection(layers.Layer):
    """4-element correction (one per output) from PySR-discovered equations.

    Constants are tf.Variables initialised to the PySR-found values so SGD
    can refine them during distillation. Output is in LABEL units (matches
    the residual PySR fit), so the caller should add this AFTER the
    `raw / labels_scale` step.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # x correction: base = c_x0; lorentz = c_x1 * tasym_x; edge = c_x2
        self.c_x_base = self.add_weight(
            name='c_x_base', shape=(),
            initializer=tf.constant_initializer(0.32042542), trainable=True)
        self.c_x_lorentz = self.add_weight(
            name='c_x_lorentz', shape=(),
            initializer=tf.constant_initializer(0.001411708), trainable=True)
        self.c_x_edge = self.add_weight(
            name='c_x_edge', shape=(),
            initializer=tf.constant_initializer(0.22818604), trainable=True)
        # y correction: base = c_y0 * skew_y * wy; lorentz = c_y1; edge = c_y2
        self.c_y_base = self.add_weight(
            name='c_y_base', shape=(),
            initializer=tf.constant_initializer(0.08006955), trainable=True)
        self.c_y_lorentz = self.add_weight(
            name='c_y_lorentz', shape=(),
            initializer=tf.constant_initializer(0.40809396), trainable=True)
        self.c_y_edge = self.add_weight(
            name='c_y_edge', shape=(),
            initializer=tf.constant_initializer(-0.8528123), trainable=True)
        # cot_a sat: tanh(cota_geom - off) * square(square(q_frac_max*cota_geom) * s)
        self.c_a_tanh_off = self.add_weight(
            name='c_a_tanh_off', shape=(),
            initializer=tf.constant_initializer(6.824275), trainable=True)
        self.c_a_inner_s = self.add_weight(
            name='c_a_inner_s', shape=(),
            initializer=tf.constant_initializer(0.35386667), trainable=True)
        self.c_a_asym = self.add_weight(
            name='c_a_asym', shape=(),
            initializer=tf.constant_initializer(0.0047847615), trainable=True)
        # cot_b sat: square(wy * (off - q_frac_max * cotb_geom)); asym constant
        self.c_b_off = self.add_weight(
            name='c_b_off', shape=(),
            initializer=tf.constant_initializer(0.1110652), trainable=True)
        self.c_b_asym = self.add_weight(
            name='c_b_asym', shape=(),
            initializer=tf.constant_initializer(-0.046504434), trainable=True)
        super().build(input_shape)

    def call(self, charge):
        f = cluster_features_tf(charge)
        # x output
        x_corr = self.c_x_base + self.c_x_lorentz * f['tasym_x'] + self.c_x_edge
        # y output
        y_corr = self.c_y_base * f['skew_y'] * f['wy'] + self.c_y_lorentz + self.c_y_edge
        # cot_a output
        inner_a = tf.square(f['q_frac_max'] * f['cota_geom']) * self.c_a_inner_s
        a_corr = tf.tanh(f['cota_geom'] - self.c_a_tanh_off) * tf.square(inner_a) + self.c_a_asym
        # cot_b output
        inner_b = f['wy'] * (self.c_b_off - f['q_frac_max'] * f['cotb_geom'])
        b_corr = tf.square(inner_b) + self.c_b_asym
        return tf.stack([x_corr, y_corr, a_corr, b_corr], axis=-1)
