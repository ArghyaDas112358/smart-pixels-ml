import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from SoftQuantizeLayer import SoftQuantizeLayer
from SoftRouterLayer import SoftRouterLayer
from SimpleRouterLayer import SimpleRouterLayer
from PairLatticeRouterLayer import PairLatticeRouterLayer
from TwoRouterLayer import TwoRouterLayer
from UCBRouterLayer import UCBRouterLayer

# Vision Transformer (ViT) ported verbatim from the legacy repo
# (legacy/smart_pixels_ml/train_loop.py). The PatchExtractor / PatchEncoder /
# transformer_encoder building blocks and the create_vit_model head are kept
# identical; only the wrapping is adapted to this repo's model conventions
# (plain vs _SoftQuantizer variants, Max/Full/Slim heads).


class PatchExtractor(layers.Layer):
    """Extract 2D patches from images."""

    def __init__(self, patch_size=(3, 7)):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        patch_h, patch_w = self.patch_size
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, patch_h, patch_w, 1),
            strides=(1, patch_h, patch_w, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",
        )
        patch_dims = tf.shape(patches)[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    """Linear embedding + learnable positional encoding."""

    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(embed_dim)
        self.pos_embed = tf.Variable(
            initial_value=tf.zeros((1, num_patches, embed_dim)),
            trainable=True,
            name="pos_embedding",
        )

    def call(self, patch_batch):
        projected = self.projection(patch_batch)
        return projected + self.pos_embed


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=head_size, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1], activation="linear")(x)
    x = layers.Dropout(dropout)(x)

    return x + res


def _vit_backbone(
    x,
    input_shape,
    output,
    patch_size=(3, 4),
    embed_dim=64,
    num_heads=4,
    ff_dim=128,
    num_layers=4,
    dropout=0.1,
    head_dims=(64,),
):
    """Patch -> encode -> N transformer blocks -> regression head.

    Mirrors create_vit_model in legacy train_loop.py. `x` is the (already
    soft-quantized, or raw) feature tensor; `input_shape` is (H, W, C).
    """
    H, W, C = input_shape
    ph, pw = patch_size
    num_patches = (H // ph) * (W // pw)

    patches = PatchExtractor(patch_size=patch_size)(x)
    encoded_patches = PatchEncoder(num_patches, embed_dim)(patches)

    x = encoded_patches
    for _ in range(num_layers):
        x = transformer_encoder(
            x, head_size=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout
        )
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Flatten()(x)
    # Regression head. The default (64,) is the legacy single-bottleneck head:
    # Flatten(1280) -> Dense(64) -> Dense(output), i.e. ONE hidden layer holding
    # 19.6% of the model's parameters in a single matrix, shared by all 14
    # Gaussian parameters. head_dims=(256,128,64) is the Deep variant (option
    # O17): same trunk, deeper funnel, testing whether the shallow shared
    # bottleneck is what makes good basins rare (measured on frozen [11,26]:
    # only ~1 in 5 seeds reaches the deep NLL basin).
    for w in head_dims:
        x = layers.Dense(w, activation="relu")(x)
    outputs = layers.Dense(output, activation="linear")(x)
    return outputs


def _vit_plain(shape, output):
    x_in = layers.Input(shape=shape, name="raw_input")
    outputs = _vit_backbone(x_in, shape, output)
    return Model(inputs=x_in, outputs=outputs, name="smrtpxl_vit")


def _vit_softquantizer(shape, output, initial_thresholds, threshold_offset,
                       initial_levels=None, trainable_thresholds=True):
    x_in = layers.Input(shape=shape, name="raw_input")
    x = SoftQuantizeLayer(
        n_bits=2,
        initial_thresholds=initial_thresholds,
        threshold_offset=threshold_offset,
        initial_levels=initial_levels,
        trainable_levels=False,
        trainable_thresholds=trainable_thresholds,
        initial_k=1.0,
        trainable_k=True,
        name="soft_quantizer_output",
    )(x_in)
    outputs = _vit_backbone(x, shape, output)
    return Model(inputs=x_in, outputs=outputs, name="smrtpxl_vit")


def _vit_softrouter(shape, output, initial_thresholds, threshold_offset,
                    initial_levels=None, trainable_thresholds=True, num_slots=2):
    """JOINT slice + threshold discovery model (docs/soft_router_plan.md, Fig A).

    `shape` is the all-slice input (H, W, T=101). SoftRouterLayer selects
    `num_slots` slices, SoftQuantizeLayer digitizes them, and BOTH are annealed
    (two AnnealingSchedulers, same cosine) so slices and thresholds co-adapt.
    The backbone is byte-identical to the production 2-slice model.
    """
    x_in = layers.Input(shape=shape, name="raw_input")
    x = SoftRouterLayer(
        num_slots=num_slots,
        initial_k=1.0,
        trainable_k=True,
        name="soft_router_output",
    )(x_in)
    x = SoftQuantizeLayer(
        n_bits=2,
        initial_thresholds=initial_thresholds,
        threshold_offset=threshold_offset,
        initial_levels=initial_levels,
        trainable_levels=False,
        trainable_thresholds=trainable_thresholds,
        initial_k=1.0,
        trainable_k=True,
        name="soft_quantizer_output",
    )(x)
    backbone_shape = (shape[0], shape[1], num_slots)
    outputs = _vit_backbone(x, backbone_shape, output)
    return Model(inputs=x_in, outputs=outputs, name="smrtpxl_vit_router")


def _vit_simplerouter(shape, output, initial_thresholds, threshold_offset,
                      initial_levels=None, trainable_thresholds=True, num_slots=2,
                      anneal_beta=False, smooth_logits=False, head_dims=(64,)):
    """JOINT slice + threshold discovery model, SIMPLE variant (Option D).

    Same anatomy as _vit_softrouter but the slice selector is SimpleRouterLayer
    (exact pair-sampling gradient, Ahmed et al. ICLR 2023): NO router annealer —
    only the downstream SoftQuantizeLayer is annealed. The backbone is
    byte-identical to the production 2-slice model.
    """
    x_in = layers.Input(shape=shape, name="raw_input")
    x = SimpleRouterLayer(
        num_slots=num_slots,
        anneal_beta=anneal_beta,
        smooth_logits=smooth_logits,
        name="simple_router_output",
    )(x_in)
    x = SoftQuantizeLayer(
        n_bits=2,
        initial_thresholds=initial_thresholds,
        threshold_offset=threshold_offset,
        initial_levels=initial_levels,
        trainable_levels=False,
        trainable_thresholds=trainable_thresholds,
        initial_k=1.0,
        trainable_k=True,
        name="soft_quantizer_output",
    )(x)
    backbone_shape = (shape[0], shape[1], num_slots)
    outputs = _vit_backbone(x, backbone_shape, output, head_dims=head_dims)
    return Model(inputs=x_in, outputs=outputs, name="smrtpxl_vit_simplerouter")


# ---- Max (14 outputs, full covariance -> custom_loss) ----
def ViT_Max(shape):
    return _vit_plain(shape, output=14)

def ViT_Max_SoftQuantizer(shape, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    return _vit_softquantizer(shape, 14, initial_thresholds, threshold_offset, initial_levels, trainable_thresholds)

def ViT_Max_SoftRouter(shape, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    return _vit_softrouter(shape, 14, initial_thresholds, threshold_offset, initial_levels, trainable_thresholds)

def ViT_Max_SimpleRouter(shape, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    return _vit_simplerouter(shape, 14, initial_thresholds, threshold_offset, initial_levels, trainable_thresholds)

def ViT_MaxDeep_SimpleRouter(shape, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    """SimpleRouter + the DEEP regression head (option O17):
    Flatten(1280) -> 256 -> 128 -> 64 -> 14, versus the legacy 1280 -> 64 -> 14.
    Separate entry point because the head weights make checkpoints incompatible
    with every shallow-head run."""
    return _vit_simplerouter(shape, 14, initial_thresholds, threshold_offset, initial_levels,
                             trainable_thresholds, head_dims=(256, 128, 64))

def ViT_MaxDeep_SimpleRouterSmooth(shape, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    """DEEP head + annealed smoothing kernel on the router (option O18):
    O15's free-router search with O17's regression head. Tests whether the deep
    basin (reachable only by the deep head, O17) flips the plain objective's
    preference to EARLY slices -- if it does, the router should now find them
    without any angle constraint doing the pushing."""
    return _vit_simplerouter(shape, 14, initial_thresholds, threshold_offset, initial_levels,
                             trainable_thresholds, smooth_logits=True, head_dims=(256, 128, 64))

def ViT_Max_SimpleRouterSmooth(shape, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    """SimpleRouter + an ANNEALED Gaussian kernel on the pair logits (option O15).

    theta carries no metric on the slice axis -- the layer is invariant to
    shuffling slice indices, so a gradient can only re-vote slice by slice and
    never slide a preference along time. The kernel couples neighbours during
    early training and anneals to exactly 0, after which the layer is identical
    to plain ViT_Max_SimpleRouter. Separate entry point because the
    smooth_sigma weight makes checkpoints incompatible with the plain runs."""
    return _vit_simplerouter(shape, 14, initial_thresholds, threshold_offset, initial_levels,
                             trainable_thresholds, smooth_logits=True)

def ViT_Max_SimpleRouterBeta(shape, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    """SimpleRouter + annealable inverse temperature beta on the pair logits
    (option O4). Separate entry point because the beta weight makes checkpoints
    incompatible with plain ViT_Max_SimpleRouter runs."""
    return _vit_simplerouter(shape, 14, initial_thresholds, threshold_offset, initial_levels,
                             trainable_thresholds, anneal_beta=True)


# ---- Full (8 outputs, diagonal covariance -> custom_diag_loss) ----
def ViT_Full(shape):
    return _vit_plain(shape, output=8)

def ViT_Full_SoftQuantizer(shape, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    return _vit_softquantizer(shape, 8, initial_thresholds, threshold_offset, initial_levels, trainable_thresholds)


# ---- Slim (3 outputs -> custom_sse_loss) ----
def ViT_Slim(shape):
    return _vit_plain(shape, output=3)

def ViT_Slim_SoftQuantizer(shape, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    return _vit_softquantizer(shape, 3, initial_thresholds, threshold_offset, initial_levels, trainable_thresholds)



def _vit_pairlattice(shape, output, initial_thresholds, threshold_offset,
                     initial_levels=None, trainable_thresholds=True, num_slots=2,
                     smooth_logits=False, head_dims=(64,)):
    """JOINT slice + threshold discovery model, PAIR-LATTICE variant (O21a).

    Same anatomy as _vit_simplerouter but the slice selector is
    PairLatticeRouterLayer: one free logit per C(T,2) pair instead of the
    factorised exp(theta_a + theta_b), with coarse-to-fine smoothing in PAIR
    space rather than along the slice axis. The layer keeps the name
    'simple_router_output' so SimpleRouterLogger, SmoothSigmaScheduler and the
    driver's get_layer plumbing work unchanged. NO router annealer -- only the
    downstream SoftQuantizeLayer is annealed. The backbone is byte-identical
    to the production 2-slice model.
    """
    x_in = layers.Input(shape=shape, name="raw_input")
    x = PairLatticeRouterLayer(
        num_slots=num_slots,
        smooth_logits=smooth_logits,
        name="simple_router_output",
    )(x_in)
    x = SoftQuantizeLayer(
        n_bits=2,
        initial_thresholds=initial_thresholds,
        threshold_offset=threshold_offset,
        initial_levels=initial_levels,
        trainable_levels=False,
        trainable_thresholds=trainable_thresholds,
        initial_k=1.0,
        trainable_k=True,
        name="soft_quantizer_output",
    )(x)
    backbone_shape = (shape[0], shape[1], num_slots)
    outputs = _vit_backbone(x, backbone_shape, output, head_dims=head_dims)
    return Model(inputs=x_in, outputs=outputs, name="smrtpxl_vit_pairlattice")


def ViT_MaxDeep_PairLattice(shape, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    """PairLattice router + the DEEP regression head (O21a = O18's slot in the
    ledger with the slice-axis kernel swapped for the pair-space one).
    smooth_logits=True on purpose: the 2D pair-space kernel is the arm's whole
    point, and the smooth_sigma weight it adds makes checkpoints incompatible
    with any un-smoothed variant -- hence a separate entry point, like every
    other extra-weight variant in this file. The 5050-logit psi also makes
    checkpoints incompatible with EVERY SimpleRouter run regardless of head."""
    return _vit_pairlattice(shape, 14, initial_thresholds, threshold_offset, initial_levels,
                            trainable_thresholds, smooth_logits=True, head_dims=(256, 128, 64))


# router models -- there is no quantizer-less variant):


def _vit_tworouter(shape, output, initial_thresholds, threshold_offset,
                   initial_levels=None, trainable_thresholds=True, num_slots=2,
                   overlap_scale=100.0, head_dims=(64,)):
    """JOINT slice + threshold discovery, TWO-ROUTER variant (option O21b).

    Same anatomy as _vit_simplerouter but the selector is TwoRouterLayer: two
    INDEPENDENT softmaxes (one per readout slot, collision-masked at sample
    time) instead of one pair distribution, so 'one early slice + one later
    slice' is the default expressive mode rather than a two-bump contortion of
    a single theta. The smoothing kernel is built in (one shared smooth_sigma
    weight for both routers, SmoothSigmaScheduler hook) and an anti-overlap
    penalty enters via layer add_loss — model.compute_loss folds it in, no
    extra callback. The layer KEEPS the name 'simple_router_output' so
    SimpleRouterLogger, SmoothSigmaScheduler and the fix-slices/warm-start
    plumbing all work unchanged. The backbone is byte-identical to the
    production 2-slice model.
    """
    x_in = layers.Input(shape=shape, name="raw_input")
    x = TwoRouterLayer(
        num_slots=num_slots,
        overlap_scale=overlap_scale,
        name="simple_router_output",
    )(x_in)
    x = SoftQuantizeLayer(
        n_bits=2,
        initial_thresholds=initial_thresholds,
        threshold_offset=threshold_offset,
        initial_levels=initial_levels,
        trainable_levels=False,
        trainable_thresholds=trainable_thresholds,
        initial_k=1.0,
        trainable_k=True,
        name="soft_quantizer_output",
    )(x)
    backbone_shape = (shape[0], shape[1], num_slots)
    outputs = _vit_backbone(x, backbone_shape, output, head_dims=head_dims)
    return Model(inputs=x_in, outputs=outputs, name="smrtpxl_vit_tworouter")


def ViT_MaxDeep_TwoRouterSmooth(shape, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    """Two independent routers + DEEP head (option O21b):
    Flatten(1280) -> 256 -> 128 -> 64 -> 14, one softmax per readout slot.
    Separate entry point because thetaA/thetaB (and the head) make checkpoints
    incompatible with every single-theta run."""
    return _vit_tworouter(shape, 14, initial_thresholds, threshold_offset, initial_levels,
                          trainable_thresholds, head_dims=(256, 128, 64))


def _vit_ucbrouter(shape, output, initial_thresholds, threshold_offset,
                   initial_levels=None, trainable_thresholds=True, num_slots=2,
                   head_dims=(64,), ucb_c=1.0):
    """JOINT slice + threshold discovery model, BANDIT variant (option O21c).

    Same anatomy as _vit_simplerouter but the slice selector is UCBRouterLayer
    (discounted-UCB over the 5050 slice pairs): NO gradient search on the
    router at all — the pair choice each step is a bandit argmax, the head
    trains through a plain gather of the chosen slices, and the training
    wrapper must call router.bandit_update(loss) after each train step (the
    mdmm.py hook is applied by the orchestrator). Only the downstream
    SoftQuantizeLayer is annealed. The backbone is byte-identical to the
    production 2-slice model.
    """
    x_in = layers.Input(shape=shape, name="raw_input")
    x = UCBRouterLayer(
        num_slots=num_slots,
        c=ucb_c,
        name="simple_router_output",
    )(x_in)
    x = SoftQuantizeLayer(
        n_bits=2,
        initial_thresholds=initial_thresholds,
        threshold_offset=threshold_offset,
        initial_levels=initial_levels,
        trainable_levels=False,
        trainable_thresholds=trainable_thresholds,
        initial_k=1.0,
        trainable_k=True,
        name="soft_quantizer_output",
    )(x)
    backbone_shape = (shape[0], shape[1], num_slots)
    outputs = _vit_backbone(x, backbone_shape, output, head_dims=head_dims)
    return Model(inputs=x_in, outputs=outputs, name="smrtpxl_vit_ucbrouter")


def ViT_MaxDeep_UCBRouter(shape, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    """Discounted-UCB bandit router + the DEEP regression head (O17's
    256->128->64 funnel): option O21c. The router keeps the layer name
    'simple_router_output' on purpose — SimpleRouterLogger, the checkpoint
    plumbing, and the eval scripts address the router by that name and the
    UCB layer implements the full logger contract (theta/mu/visits/sigma/
    selected_indices). Separate entry point because the bandit state weights
    (q/n/t/tried/EMA) make checkpoints incompatible with every SimpleRouter
    run."""
    return _vit_ucbrouter(shape, 14, initial_thresholds, threshold_offset, initial_levels,
                          trainable_thresholds, head_dims=(256, 128, 64))
