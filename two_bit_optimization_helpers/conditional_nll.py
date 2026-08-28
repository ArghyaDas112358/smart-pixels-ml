"""
The 4D Gaussian NLL, itemised per target.

Same number as loss.custom_loss -- this is an exact identity, not a new
objective. What it buys is that the per-variable costs become addressable, so
they can be logged, weighted (beta-NLL), or constrained individually. You cannot
put an MDMM lambda on "the angle NLL" while the loss returns one fused scalar.

WHY IT DECOMPOSES
    The model emits mu and a lower-triangular L (Sigma = L L^T). Writing
    y = mu + L z with z ~ N(0, I), triangularity means each target gets exactly
    one NEW random number:

        y_k = mu_k + sum_{j<k} L_kj z_j  +  L_kk z_k

    Knowing y_1..y_{k-1} pins z_1..z_{k-1} (forward substitution), so

        y_k | y_{<k} ~ N( mu_k + sum_{j<k} L_kj z_j ,  L_kk^2 )

    and its NLL is the 1-D Gaussian one, 0.5*log(2pi) + log L_kk + 0.5 z_k^2.
    By the chain rule those four terms sum to the joint NLL. Verified against
    tfp.MultivariateNormalTriL to ~1e-15 in test_loss_split.py.

COLUMN ORDER
    Whatever order the LABELS are in. Today that is (x, y, cotA, cotB), so
    terms[:, 2] is cot alpha GIVEN position and terms[:, 3] is cot beta given
    position AND cot alpha. The last slot is conditioned on more, so a single
    angle's number is order-dependent -- but their SUM is not:

        terms[:, 2] + terms[:, 3] = -log p(cotA, cotB | x, y)

    which is identical whichever angle you put first (checked numerically). Use
    `angle_block` when you want an order-free quantity to constrain or report.

NOTE ON THE CLIP -- READ BEFORE SWAPPING THE TRAINING LOSS
    loss.custom_loss builds a density, clips it to [1e-9, 1e9], then logs, which
    pins the per-event NLL to [-20.72, +20.72]. That is NOT cosmetic:

      * At initialisation nearly every event sits on the +20.72 ceiling
        (recorded initial val ~99,113 over a 5,000-event batch = 19.8/event), so
        the NLL contributes almost no gradient and the run has to be dragged off
        the plateau by the constraint terms.
      * On a TRAINED model it truncates the worst-fit tail. Measured on seed
        2042, batch 0: 4,992 of 5,000 events agree with this module to 3.6e-15,
        and the 8 that do not (0.16%) are capped at 20.72 while their true NLL is
        27-211. The current loss is therefore implicitly Huberised at the top,
        and those 8 events carry the largest gradients.

    So `custom_loss_split(..., clip=True)` (the default) reproduces the current
    objective exactly, tail truncation included. Pass clip=False to get the
    untruncated NLL -- a real change of objective, not a refactor, which will
    weight catastrophic events roughly 10x harder. `nll_terms` is always
    unclipped, because clipping a sum cannot be attributed back to its parts.
"""
import numpy as np
import tensorflow as tf

# Layout of the 14-vector the model emits (same lineage as loss.custom_loss and
# MDMM_OUTPUT_COLUMNS): means interleaved at 0,2,4,6; L diagonal at 1,3,5,7;
# the six strictly-lower entries of L in 8..13, row-major.
MU_SLICE = slice(0, 8, 2)
DIAG_SLICE = slice(1, 8, 2)
COV_SLICE = slice(8, None)
HALF_LOG_2PI = 0.5 * np.log(2.0 * np.pi)


def scale_tril(p_base, minval=1e-9, diag_mode="relu"):
    """Assemble L. Returns (L, diag).

    diag_mode is the DIALECT of the raw diagonal outputs and must match how the
    checkpoint was TRAINED -- scoring across dialects is pure artifact
    (~7 nats/event measured on QConv2D_Max by the Symb_ASIC side):
      * "relu"     -- minval + max(p, 0), the loss.custom_loss lineage
                      (O11..O21). Dead gradient below 0; diag can pin at the
                      1e-9 floor, which is what creates the clip plateau and
                      the z-cascade gradient spikes.
      * "softplus" -- minval + softplus(p), the v2 lineage (O21.v2+): smooth,
                      strictly positive, softplus(0)=0.693 so a cold init
                      starts at a healthy scale instead of the floor.
    """
    if diag_mode == "softplus":
        diag = minval + tf.math.softplus(p_base[:, DIAG_SLICE])
    else:
        diag = minval + tf.math.maximum(p_base[:, DIAG_SLICE], 0.0)
    cov = p_base[:, COV_SLICE]
    zeros = tf.zeros_like(diag[:, 0])
    row1 = tf.stack([diag[:, 0], zeros, zeros, zeros])
    row2 = tf.stack([cov[:, 0], diag[:, 1], zeros, zeros])
    row3 = tf.stack([cov[:, 1], cov[:, 2], diag[:, 2], zeros])
    row4 = tf.stack([cov[:, 3], cov[:, 4], cov[:, 5], diag[:, 3]])
    L = tf.transpose(tf.stack([row1, row2, row3, row4]), perm=[2, 0, 1])
    return L, diag


def nll_terms(y, p_base, minval=1e-9, diag_mode="relu"):
    """Per-event, per-target conditional NLL. Shape (B, 4); rows sum to the joint."""
    y = tf.cast(y, p_base.dtype)
    L, diag = scale_tril(p_base, minval, diag_mode)
    r = tf.expand_dims(y - p_base[:, MU_SLICE], -1)
    # forward substitution, not an explicit inverse
    z = tf.linalg.triangular_solve(L, r, lower=True)[..., 0]
    return 0.5 * tf.square(z) + tf.math.log(diag) + HALF_LOG_2PI


def custom_loss_split(y, p_base, minval=1e-9, maxval=1e9, clip=True):
    """Summed joint NLL. With clip=True this equals loss.custom_loss exactly.

    clip=True bounds each event to [-log(maxval), -log(minval)], matching the
    clip_by_value on the density in loss.custom_loss -- including the truncation
    of the worst-fit tail (see the module docstring). clip=False removes that
    bound and IS a change of objective.
    """
    per_event = tf.reduce_sum(nll_terms(y, p_base, minval), axis=-1)
    if clip:
        per_event = tf.clip_by_value(per_event,
                                     -tf.math.log(tf.cast(maxval, per_event.dtype)),
                                     -tf.math.log(tf.cast(minval, per_event.dtype)))
    return tf.reduce_sum(per_event)


def custom_loss_v2(y, p_base, minval=1e-9):
    """Loss v2 (O21.v2+): softplus diagonal, NO clip, log-domain, batch SUM.

    tfp-free port of the Symb_ASIC side's custom_loss_fixed (they use tfp +
    per-event MEAN; we keep the campaign lineage's batch-SUM so MDMM constraint
    scales, AbortOnStuck thresholds, and every logged magnitude stay in the
    same units -- v2 numbers differ from theirs by exactly the batch factor).
    Same 14-output layout; ONLY the diag dialect and the missing clip differ
    from custom_loss. Numbers are NOT comparable with the clip-dialect ledger
    (O11..O21) and v2 checkpoints must never be scored under the relu mapping
    (or vice versa) without retraining -- see scale_tril.
    """
    per_event = tf.reduce_sum(nll_terms(y, p_base, minval, diag_mode="softplus"), axis=-1)
    return tf.reduce_sum(per_event)


def angle_block(y, p_base, minval=1e-9, diag_mode="relu"):
    """-log p(cotA, cotB | x, y) per event. Order-invariant in the two angles."""
    t = nll_terms(y, p_base, minval, diag_mode)
    return t[:, 2] + t[:, 3]


def position_block(y, p_base, minval=1e-9, diag_mode="relu"):
    """-log p(x, y) per event."""
    t = nll_terms(y, p_base, minval, diag_mode)
    return t[:, 0] + t[:, 1]


def beta_nll_terms(y, p_base, betas, minval=1e-9):
    """beta-NLL (Seitzer et al.): per-target term scaled by stop_grad(L_kk^(2*beta)).

    beta = 0 recovers plain NLL (the mean's gradient carries the 1/sigma^2 factor
    that lets the network abandon hard targets); beta = 1 removes that factor
    entirely, weighting every event alike. `betas` is a length-4 sequence, so
    position can stay at 0 while only the angles are re-weighted.

    The weight is DETACHED on purpose: sigma keeps learning from the unweighted
    terms, so calibration is preserved -- only the mean's learning signal stops
    being gated by the variance.
    """
    _, diag = scale_tril(p_base, minval)
    b = tf.constant(np.asarray(betas, dtype=np.float32).reshape(1, 4))
    w = tf.stop_gradient(tf.pow(diag, 2.0 * tf.cast(b, diag.dtype)))
    return w * nll_terms(y, p_base, minval)


# ---------------------------------------------------------------------------
# O22: quasi-binning in the TRUE target value.
#
# Fix 1 of the residual-bias plan. The plain loss is a SUM over the batch, so
# whichever region of the label space is densest dominates every gradient step
# and the sparse tails get outvoted -- the network buys a small gain in the bulk
# by accepting a systematic bias out at the edges. Weighting each event by the
# inverse occupancy of its bin makes every region of the label space pull with
# equal force, i.e. trains against a flat prior instead of the sampled one.
#
# "Quasi" because the bins are Gaussian kernels, not hard edges: hard binning is
# not differentiable in the label and puts arbitrary discontinuities at the bin
# boundaries.
# ---------------------------------------------------------------------------

def soft_bin_membership(v, centers, sigma):
    """K[e, b], rows summing to 1: how much event e belongs to kernel b."""
    d = (tf.expand_dims(tf.cast(v, tf.float32), -1)
         - tf.reshape(tf.cast(centers, tf.float32), (1, -1))) / tf.cast(sigma, tf.float32)
    logk = -0.5 * tf.square(d)
    return tf.nn.softmax(logk, axis=-1)


def soft_bin_weights(y, specs, clip_lo=0.25, clip_hi=4.0):
    """Per-event weight that flattens the marginal of every target in `specs`.

    specs: list of (label_column, centers, sigma). One weight has to serve all
    of them -- an event sits in some beta bin AND some alpha bin -- so the
    per-target inverse-occupancy weights are multiplied and renormalised to mean
    1. The clip is the safety valve: without it a single event in a sparse
    corner of the joint space can carry a whole batch's gradient.
    """
    w = tf.ones(tf.shape(y)[0], dtype=tf.float32)
    n = tf.cast(tf.shape(y)[0], tf.float32)
    for col, centers, sigma in specs:
        K = soft_bin_membership(y[:, col], centers, sigma)          # (N, B)
        occ = tf.reduce_sum(K, axis=0)                              # (B,)
        nb = tf.cast(tf.shape(K)[1], tf.float32)
        per_ev = tf.reduce_sum(K * (n / (nb * (occ + 1e-6))), axis=-1)
        w = w * per_ev
    w = w / (tf.reduce_mean(w) + 1e-12)
    w = tf.clip_by_value(w, clip_lo, clip_hi)
    return w / (tf.reduce_mean(w) + 1e-12)


def binbalanced_loss_v2(y, p_base, specs, minval=1e-9, clip_lo=0.25, clip_hi=4.0):
    """Loss v2, but every bin of every target pulls with equal weight.

    REPLACES custom_loss_v2 -- it is not an extra term. Bin-balancing is a
    statement about how the batch is averaged, so adding it alongside the plain
    sum would leave the plain sum free to keep outvoting the tails.

    NOTE: this changes the objective, so the reported NLL is NOT comparable with
    the O11..O21 ledger or with any unweighted loss-v2 run.
    """
    per_event = tf.reduce_sum(nll_terms(y, p_base, minval, diag_mode="softplus"), axis=-1)
    w = soft_bin_weights(y, specs, clip_lo, clip_hi)
    return tf.reduce_sum(w * per_event)


# --- physical-space helpers -------------------------------------------------
# The angle targets are stored as scaled cotangents, but the bias lgray is
# reading off is in DEGREES. inverse_cot is continuous through c = 0 once the
# branch is wrapped (c -> 0+ gives pi/2, c -> 0- gives -pi/2 + pi = pi/2), and
# its derivative -1/(1+c^2) is smooth everywhere, so this is safe to put inside
# the training graph.

def cot_to_deg(c_scaled, scale):
    # atan2(1, c), NOT atan(1/c). Same value everywhere -- atan2 already lands
    # in (0, pi), which is the branch the eval plot wraps to by hand -- but the
    # division form has an infinite derivative at c = 0, so a single event with
    # a near-zero cotangent NaNs the whole batch gradient. atan2 never divides,
    # and (1, c) never reaches the origin, so it is smooth throughout.
    c = tf.cast(c_scaled, tf.float32) * tf.cast(scale, tf.float32)
    return tf.atan2(tf.ones_like(c), c) * (180.0 / np.pi)


def per_target_bin_weight(v, centers, sigma, clip_lo=0.25, clip_hi=4.0):
    """Inverse-occupancy weight for ONE target. Mean 1, clipped."""
    K = soft_bin_membership(v, centers, sigma)                  # (N, B)
    occ = tf.reduce_sum(K, axis=0)                              # (B,)
    n = tf.cast(tf.shape(K)[0], tf.float32)
    nb = tf.cast(tf.shape(K)[1], tf.float32)
    w = tf.reduce_sum(K * (n / (nb * (occ + 1e-6))), axis=-1)
    w = w / (tf.reduce_mean(w) + 1e-12)
    w = tf.clip_by_value(w, clip_lo, clip_hi)
    return w / (tf.reduce_mean(w) + 1e-12)


def binbalanced_loss_v2_byterm(y, p_base, specs, minval=1e-9,
                               clip_lo=0.25, clip_hi=4.0):
    """Loss v2 with EACH conditional term balanced in its OWN target's bins.

    nll_terms already returns the chain-rule decomposition
        -log p(y) = sum_k -log p(y_k | y_1..y_{k-1}),
    shape (B, 4), so term k can carry its own weight and no single per-event
    weight has to serve four different binnings at once. That removes the whole
    product-of-inverse-occupancies construction, and with it the multiplicative
    blow-up in sparse corners of the joint space.

    READ THIS BEFORE TRUSTING THE DECOMPOSITION. The terms are CONDITIONAL, not
    marginal. L is lower-triangular, so forward substitution gives
        z_k = (r_k - sum_{j<k} L_kj z_j) / L_kk,
    i.e. term k is target k AFTER conditioning on targets 1..k-1. Weighting it
    by target k's own occupancy is still the right thing for target k's bias,
    but the treatment is asymmetric: term 0 (x) is unconditional while term 3
    (cotB) sits downstream of three conditionings, and the strictly-lower L
    entries are shared across terms, so reweighting the terms also reweights
    how hard each correlation is fitted.

    Consequence: the weighted total is a composite objective, not the log
    likelihood of any single distribution. It is a further dialect change on
    top of loss v2 -- do not compare its value with anything.

    specs: list of (term_index, label_column, centers, sigma). Terms not listed
    keep uniform weight.
    """
    terms = nll_terms(y, p_base, minval, diag_mode="softplus")   # (B, 4)
    by_term = {int(t): (c, cen, sg) for (t, c, cen, sg) in specs}
    cols = []
    for k in range(terms.shape[-1]):
        if k in by_term:
            col, centers, sigma = by_term[k]
            cols.append(per_target_bin_weight(tf.cast(y[:, col], tf.float32),
                                              centers, sigma, clip_lo, clip_hi))
        else:
            cols.append(tf.ones(tf.shape(y)[0], dtype=tf.float32))
    W = tf.stack(cols, axis=-1)                                  # (B, 4)
    return tf.reduce_sum(tf.cast(W, terms.dtype) * terms)
