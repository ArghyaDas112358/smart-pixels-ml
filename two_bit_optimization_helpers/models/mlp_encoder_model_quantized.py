import tensorflow as tf
from keras.layers import (
    Input, Flatten, AveragePooling2D,
    Reshape, Concatenate
)
from keras.models import Model
from qkeras import (
    QDense, QConv1D, QConv2D, QActivation, quantized_bits
)
from SoftQuantizeLayer import SoftQuantizeLayer

def _var_network(var, hidden=10, output=2):
    var = Flatten(name="flatten_var")(var)
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="dense_1"
    )(var)
    #var = keras.activations.tanh(var)
    var = QActivation("quantized_tanh(8, 0, 1)", name="activation_tanh_2")(var)
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="dense_2"
    )(var)
    #var = keras.activations.tanh(var)
    var = QActivation("quantized_tanh(8, 0, 1)", name="activation_tanh_3")(var)
    return QDense(
        output,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        name="dense_3"
    )(var)

def _mlp_encoder_network(var, hidden=16, hidden_dimx=16, hidden_dimy=16):
    proj_x = AveragePooling2D(
        pool_size=(1, hidden_dimx), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(var)
    proj_x = Flatten()(proj_x)

    proj_y = AveragePooling2D(
        pool_size=(hidden_dimy, 1), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(var)
    proj_y = Flatten()(proj_y)

    proj_x = QDense(
        hidden_dimx,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(proj_x)
    proj_x = QActivation("quantized_relu(bits=13, integer=5)")(proj_x)

    proj_y = QDense(
        hidden_dimy,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(proj_y)
    proj_y = QActivation("quantized_relu(bits=13, integer=5)")(proj_y)

    var = Concatenate(axis=1)([proj_x, proj_y])

    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)

    var = QActivation("quantized_tanh(8, 0, 1)")(var)
    return var

def _moe_expert_head(emb, hidden=16, output=14, idx=0):
    """One expert regression head on the shared embedding (unique layer names)."""
    v = QDense(hidden,
               kernel_quantizer=quantized_bits(8, 0, alpha=1),
               bias_quantizer=quantized_bits(8, 0, alpha=1),
               kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
               activity_regularizer=tf.keras.regularizers.L2(0.01),
               name=f"exp{idx}_d1")(emb)
    v = QActivation("quantized_tanh(8, 0, 1)", name=f"exp{idx}_a1")(v)
    v = QDense(hidden,
               kernel_quantizer=quantized_bits(8, 0, alpha=1),
               bias_quantizer=quantized_bits(8, 0, alpha=1),
               kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
               activity_regularizer=tf.keras.regularizers.L2(0.01),
               name=f"exp{idx}_d2")(v)
    v = QActivation("quantized_tanh(8, 0, 1)", name=f"exp{idx}_a2")(v)
    return QDense(output,
                  kernel_quantizer=quantized_bits(8, 0, alpha=1),
                  bias_quantizer=quantized_bits(8, 0, alpha=1),
                  kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
                  name=f"exp{idx}_out")(v)

def QMlp_MoE_Max(shape, n_experts=4):
    """Mixture-of-Experts QMlp_Max: shared physics (x/y-projection) encoder,
    n_experts regression heads, and a softmax gate that routes per-event.
    Output is the gate-weighted sum of the expert 14-vectors (soft MoE)."""
    x_in = Input(shape, name="input_pxls")
    emb = _mlp_encoder_network(x_in)                       # shared 16-dim physics embedding
    experts = [_moe_expert_head(emb, output=14, idx=i) for i in range(n_experts)]
    gate = QDense(n_experts,
                  kernel_quantizer=quantized_bits(8, 0, alpha=1),
                  bias_quantizer=quantized_bits(8, 0, alpha=1),
                  name="gate")(emb)
    gate = tf.keras.layers.Softmax(name="gate_softmax")(gate)        # (B, n_experts)
    stacked = tf.keras.layers.Lambda(
        lambda t: tf.stack(t, axis=1), name="stack_experts")(experts)   # (B, n_experts, 14)
    out = tf.keras.layers.Lambda(
        lambda a: tf.reduce_sum(a[0] * tf.expand_dims(a[1], -1), axis=1),
        name="moe_combine")([stacked, gate])                # (B, 14)
    return Model(inputs=x_in, outputs=out, name="smrtpxl_moe")

def _convstem_encoder(x_in, conv_ch=8, emb=24):
    """Conv-stem + ATTENTION-POOLING encoder. Replaces the 1D average-pool
    encoder, whose pooling collapsed the 16x16 cluster into two 1D marginals
    and destroyed the 2D shape that encodes the incidence angles (the
    data-processing bottleneck that capped capacity scaling). This keeps a 2D
    feature map, then learns WHICH spatial cells matter via a softmax attention
    pool (the cheap stand-in for the teacher's attention), then projects to a
    `emb`-dim embedding. hls4ml/QKeras-friendly (QConv2D + Dense + Softmax)."""
    qb = dict(kernel_quantizer=quantized_bits(8, 0, alpha=1),
              bias_quantizer=quantized_bits(8, 0, alpha=1))
    v = QConv2D(conv_ch, (3, 3), padding='valid', name='cs_conv1', **qb)(x_in)   # 14x14xC
    v = QActivation("quantized_tanh(8, 0, 1)", name='cs_a1')(v)
    v = QConv2D(conv_ch, (3, 3), padding='valid', name='cs_conv2', **qb)(v)       # 12x12xC
    v = QActivation("quantized_tanh(8, 0, 1)", name='cs_a2')(v)
    tokens = tf.keras.layers.Reshape((-1, conv_ch), name='cs_tokens')(v)          # (144, C)
    score = QDense(1, name='cs_attn_score', **qb)(tokens)                          # (144, 1)
    weights = tf.keras.layers.Softmax(axis=1, name='cs_attn_softmax')(score)       # over tokens
    pooled = tf.keras.layers.Lambda(
        lambda a: tf.reduce_sum(a[0] * a[1], axis=1), name='cs_attn_pool')([tokens, weights])  # (C,)
    e = QDense(emb, name='cs_emb', kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
               **qb)(pooled)
    e = QActivation("quantized_tanh(8, 0, 1)", name='cs_emb_a')(e)
    return e


def QMlp_MoE_ConvStem(shape, n_experts=6, expert_hidden=24, conv_ch=8, enc_emb=24):
    """MoE with the conv-stem + attention-pool encoder (Agent-2 Proposal B).
    ~10K params at the defaults. Same 14-output gate-weighted MoE head as
    QMlp_MoE_Max, so it drops into every distiller unchanged."""
    x_in = Input(shape, name="input_pxls")
    emb = _convstem_encoder(x_in, conv_ch=conv_ch, emb=enc_emb)
    experts = [_moe_expert_head_w(emb, expert_hidden, output=14, idx=i) for i in range(n_experts)]
    gate = QDense(n_experts, kernel_quantizer=quantized_bits(8, 0, alpha=1),
                  bias_quantizer=quantized_bits(8, 0, alpha=1), name="gate")(emb)
    gate = tf.keras.layers.Softmax(name="gate_softmax")(gate)
    stacked = tf.keras.layers.Lambda(lambda t: tf.stack(t, axis=1), name="stack_experts")(experts)
    out = tf.keras.layers.Lambda(
        lambda a: tf.reduce_sum(a[0] * tf.expand_dims(a[1], -1), axis=1),
        name="moe_combine")([stacked, gate])
    return Model(inputs=x_in, outputs=out, name="smrtpxl_moe_convstem")


# Output-layer bias init for the 14-vector: means/off-diag at 0, covariance
# DIAGONAL (idx 1,3,5,7) at +0.5 so the predicted covariance is non-singular at
# init. Without this the diag starts ~0 (relu of small randoms), the Gaussian is
# singular, the likelihood pins at the clip floor (NLL ~20.7), and Nadam can get
# stuck at the +103,616/batch ceiling -- the stuck-init the retry harness was
# gambling against. This makes the MoE escape deterministically (no seed lottery).
_DIAG_POS_BIAS = tf.keras.initializers.Constant(
    [0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def _moe_expert_head_w(emb, hidden, output=14, idx=0):
    """Width-parametrized expert head (for the capacity-scaling sweep)."""
    v = QDense(hidden, kernel_quantizer=quantized_bits(8, 0, alpha=1),
               bias_quantizer=quantized_bits(8, 0, alpha=1),
               kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
               activity_regularizer=tf.keras.regularizers.L2(0.01),
               name=f"exp{idx}_d1")(emb)
    v = QActivation("quantized_tanh(8, 0, 1)", name=f"exp{idx}_a1")(v)
    v = QDense(hidden, kernel_quantizer=quantized_bits(8, 0, alpha=1),
               bias_quantizer=quantized_bits(8, 0, alpha=1),
               kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
               activity_regularizer=tf.keras.regularizers.L2(0.01),
               name=f"exp{idx}_d2")(v)
    v = QActivation("quantized_tanh(8, 0, 1)", name=f"exp{idx}_a2")(v)
    bias_init = _DIAG_POS_BIAS if output == 14 else 'zeros'
    return QDense(output, kernel_quantizer=quantized_bits(8, 0, alpha=1),
                  bias_quantizer=quantized_bits(8, 0, alpha=1),
                  bias_initializer=bias_init,
                  kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
                  name=f"exp{idx}_out")(v)


def QMlp_MoE_Max_Scaled(shape, n_experts=4, expert_hidden=16, enc_hidden=16):
    """Capacity-parametrized MoE (same topology as QMlp_MoE_Max, scalable width).
    Knobs: n_experts, expert_hidden (each expert's two hidden layers),
    enc_hidden (shared embedding width). Output is the 14-vector gate-weighted
    sum. Used by the capacity-scaling sweep to find the smallest student that
    reaches the teacher's -8.75/event."""
    x_in = Input(shape, name="input_pxls")
    emb = _mlp_encoder_network(x_in, hidden=enc_hidden)
    experts = [_moe_expert_head_w(emb, expert_hidden, output=14, idx=i) for i in range(n_experts)]
    gate = QDense(n_experts, kernel_quantizer=quantized_bits(8, 0, alpha=1),
                  bias_quantizer=quantized_bits(8, 0, alpha=1), name="gate")(emb)
    gate = tf.keras.layers.Softmax(name="gate_softmax")(gate)
    stacked = tf.keras.layers.Lambda(lambda t: tf.stack(t, axis=1), name="stack_experts")(experts)
    out = tf.keras.layers.Lambda(
        lambda a: tf.reduce_sum(a[0] * tf.expand_dims(a[1], -1), axis=1),
        name="moe_combine")([stacked, gate])
    return Model(inputs=x_in, outputs=out, name="smrtpxl_moe_scaled")


# Fixed-size variants for the sweep (create_model passes only `shape`).
def QMlp_MoE_M(shape):   return QMlp_MoE_Max_Scaled(shape, n_experts=6,  expert_hidden=24, enc_hidden=24)
def QMlp_MoE_L(shape):   return QMlp_MoE_Max_Scaled(shape, n_experts=8,  expert_hidden=40, enc_hidden=32)
def QMlp_MoE_XL(shape):  return QMlp_MoE_Max_Scaled(shape, n_experts=10, expert_hidden=64, enc_hidden=48)


def QMlp_MoE_Max_Distill(shape, n_experts=4):
    """E2 (DeiT separate-head) variant of QMlp_MoE_Max. Identical MoE trunk
    (so the first 14 outputs match QMlp_MoE_Max exactly and weights are
    transferable), PLUS a separate 4-dim 'distillation head' off the shared
    embedding that predicts ONLY the means. Total output is 18 =
    [moe_14 ; distill_means_4].

      moe_14[means]  -> trained by ground-truth NLL (the deployed covariance too)
      distill_4      -> trained by MSE to the teacher's means (DeiT distill token)

    At inference the deployed means are the AVERAGE of moe_14's means and the
    distill head (DeiT averages its two heads); the covariance is moe_14's.
    Keeping the two signals in SEPARATE heads is the whole point -- it avoids
    the GT-vs-teacher conflict that sank the single-head means-only run.
    +~68 params (one QDense(16->4))."""
    x_in = Input(shape, name="input_pxls")
    emb = _mlp_encoder_network(x_in)
    experts = [_moe_expert_head(emb, output=14, idx=i) for i in range(n_experts)]
    gate = QDense(n_experts,
                  kernel_quantizer=quantized_bits(8, 0, alpha=1),
                  bias_quantizer=quantized_bits(8, 0, alpha=1),
                  name="gate")(emb)
    gate = tf.keras.layers.Softmax(name="gate_softmax")(gate)
    stacked = tf.keras.layers.Lambda(
        lambda t: tf.stack(t, axis=1), name="stack_experts")(experts)
    moe14 = tf.keras.layers.Lambda(
        lambda a: tf.reduce_sum(a[0] * tf.expand_dims(a[1], -1), axis=1),
        name="moe_combine")([stacked, gate])                # (B, 14)
    distill_means = QDense(4,
                           kernel_quantizer=quantized_bits(8, 0, alpha=1),
                           bias_quantizer=quantized_bits(8, 0, alpha=1),
                           name="distill_mean")(emb)         # (B, 4)
    out = tf.keras.layers.Concatenate(axis=1, name="moe_distill_concat")(
        [moe14, distill_means])                              # (B, 18)
    return Model(inputs=x_in, outputs=out, name="smrtpxl_moe_distill")

def QMlp_Max(shape):
    x_base = x_in = Input(shape, name="input_pxls")
    stack = _mlp_encoder_network(x_base)
    stack = _var_network(stack, hidden=16, output=14)
    model = Model(inputs=x_in, outputs=stack, name="smrtpxl_regression")
    return model

def QMlp_Full(shape):
    x_base = x_in = Input(shape, name="input_pxls")
    stack = _mlp_encoder_network(x_base)
    stack = _var_network(stack, hidden=16, output=8)
    model = Model(inputs=x_in, outputs=stack, name="smrtpxl_regression")
    return model
    
def QMlp_Slim(shape):
    x_base = x_in = Input(shape, name="input_pxls")
    stack = _mlp_encoder_network(x_base)
    stack = _var_network(stack, hidden=16, output=3)
    model = Model(inputs=x_in, outputs=stack, name="smrtpxl_regression")
    return model

def QMlp_Max_SoftQuantizer(shape, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    x_base = x_in = Input(shape, name="input_pxls")
    x_base = SoftQuantizeLayer(
        n_bits=2,
        initial_thresholds=initial_thresholds,
        initial_levels=initial_levels,
        threshold_offset=threshold_offset,
        trainable_levels=False,
        trainable_thresholds=trainable_thresholds,
        initial_k=1.0,
        trainable_k=True,
        name='soft_quantizer_output'
    )(x_base)
    stack = _mlp_encoder_network(x_base)
    stack = _var_network(stack, hidden=16, output=14)
    model = Model(inputs=x_in, outputs=stack, name="smrtpxl_regression")
    return model

def QMlp_Full_SoftQuantizer(shape, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    x_base = x_in = Input(shape, name="input_pxls")
    x_base = SoftQuantizeLayer(
        n_bits=2,
        initial_thresholds=initial_thresholds,
        initial_levels=initial_levels,
        threshold_offset=threshold_offset,
        trainable_levels=False,
        trainable_thresholds=trainable_thresholds,
        initial_k=1.0,
        trainable_k=True,
        name='soft_quantizer_output'
    )(x_base)
    stack = _mlp_encoder_network(x_base)
    stack = _var_network(stack, hidden=16, output=8)
    model = Model(inputs=x_in, outputs=stack, name="smrtpxl_regression")
    return model
    
def QMlp_Slim_SoftQuantizer(shape, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    x_base = x_in = Input(shape, name="input_pxls")
    x_base = SoftQuantizeLayer(
        n_bits=2,                     
        initial_thresholds=initial_thresholds,
        initial_levels=initial_levels,
        threshold_offset=threshold_offset,    
        trainable_levels=False,
        trainable_thresholds=trainable_thresholds,
        initial_k=1.0,                
        trainable_k=True,             
        name='soft_quantizer_output'  
    )(x_base)
    stack = _mlp_encoder_network(x_base)
    stack = _var_network(stack, hidden=16, output=3)
    model = Model(inputs=x_in, outputs=stack, name="smrtpxl_regression")
    return model