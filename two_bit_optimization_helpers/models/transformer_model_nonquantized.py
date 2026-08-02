import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from SoftQuantizeLayer import SoftQuantizeLayer
from SoftRouterLayer import SoftRouterLayer
from SimpleRouterLayer import SimpleRouterLayer

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
    x = layers.Dense(64, activation="relu")(x)
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
                      anneal_beta=False):
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
    outputs = _vit_backbone(x, backbone_shape, output)
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
