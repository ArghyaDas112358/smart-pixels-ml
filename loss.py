import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
pi = 3.14159265359

def build_scale_tril(p_base, minval=1e-9):
    mu = p_base[:, 0:8:2]
    Mdia = minval + tf.math.maximum(p_base[:, 1:8:2], 0.0)
    Mcov = p_base[:, 8:]

    zeros = tf.zeros_like(Mdia[:, 0])

    row1 = tf.stack([Mdia[:, 0], zeros, zeros, zeros], axis=-1)
    row2 = tf.stack([Mcov[:, 0], Mdia[:, 1], zeros, zeros], axis=-1)
    row3 = tf.stack([Mcov[:, 1], Mcov[:, 2], Mdia[:, 2], zeros], axis=-1)
    row4 = tf.stack([Mcov[:, 3], Mcov[:, 4], Mcov[:, 5], Mdia[:, 3]], axis=-1)

    scale_tril = tf.stack([row1, row2, row3, row4], axis=1)  # shape: [batch, 4, 4]
    return mu, scale_tril

def loss_nll(y, p_base, minval=1e-9, maxval=1e9):
    mu, scale_tril = build_scale_tril(p_base, minval)
    dist = tfp.distributions.MultivariateNormalTriL(loc=mu, scale_tril=scale_tril)
    likelihood = dist.prob(y)
    likelihood = tf.clip_by_value(likelihood, minval, maxval)
    NLL = -1 * tf.math.log(likelihood)
    return tf.reduce_sum(NLL)

def multivariate_crps_mc(y, dist, n_samples):
    batch = tf.shape(y)[0]
    d = tf.shape(y)[1]
    half = n_samples // 2

    # Antithetic samples
    z = tf.random.normal((half, batch, d), dtype=y.dtype)
    z = tf.concat([z, -z], axis=0)  # [n_samples, batch, d]
    z = tf.expand_dims(z, -1)       # [n_samples, batch, d, 1]

    L = dist.scale.to_dense()       # [batch, d, d]
    L = tf.expand_dims(L, 0)        # [1, batch, d, d]

    samples = tf.matmul(L, z)       # [n_samples, batch, d, 1]
    samples = tf.squeeze(samples, -1) + dist.mean()[None, :, :]  # [n_samples, batch, d]

    # E[||x - y||]
    diff = tf.norm(samples - y[None, :, :], axis=-1)  # [n_samples, batch]
    E1 = tf.reduce_mean(diff, axis=0)

    # E[||x - x'||]
    s1 = tf.expand_dims(samples, 0)  # [1, n_samples, batch, d]
    s2 = tf.expand_dims(samples, 1)  # [n_samples, 1, batch, d]
    pairwise = tf.norm(s1 - s2, axis=-1)  # [n_samples, n_samples, batch]
    E2 = 0.5 * tf.reduce_mean(pairwise, axis=[0, 1])  # [batch]

    tf.debugging.assert_all_finite(L, "NaNs/Infs in Cholesky factor L")
    tf.debugging.assert_all_finite(samples, "NaNs/Infs in sampled values")
    tf.debugging.assert_all_finite(diff, "NaNs/Infs in E1 distance calculation")
    tf.debugging.assert_all_finite(pairwise, "NaNs/Infs in E2 pairwise distances")

    return tf.reduce_mean(E1 - E2)


@tf.function
def empirical_kl(p_vals, q_vals, epsilon=1e-10):
    # p_vals: (dims, n_grid)
    # q_vals: (n_grid,)
    # Normalize p_vals per dim
    p_sum = tf.reduce_sum(p_vals, axis=1, keepdims=True)
    p_sum = tf.where(p_sum < epsilon, tf.ones_like(p_sum), p_sum)
    
    p_vals = p_vals / p_sum # (dims, n_grid)
    q_vals = q_vals / tf.reduce_sum(q_vals) # (n_grid,)
    
    # Clip everything to avoid log(0) or log(negative)
    safe_p = tf.clip_by_value(p_vals, epsilon, 1.0)
    safe_q = tf.clip_by_value(q_vals, epsilon, 1.0)

    kl_term = tf.reduce_sum(safe_p * tf.math.log(safe_p / safe_q), axis=1)
    # Replace NaN with 0 just in case (better be safe than be nan XD)
    kl_term = tf.where(tf.math.is_nan(kl_term), tf.zeros_like(kl_term), kl_term)
    return kl_term # (dims,)

# 
@tf.function
def kde_tf(samples, grid, bandwidth):
    samples_T = tf.transpose(samples) 
    grid_exp = tf.reshape(grid, [1, -1, 1])  
    
    samples_exp = tf.expand_dims(samples_T, 1)  
    diff = grid_exp - samples_exp
    
    kernel_vals = tf.exp(-0.5 * tf.square(diff / bandwidth))
    kernel_vals /= (bandwidth * tf.sqrt(2. * tf.constant(pi, dtype=tf.float32)))
    
    # Average over samples (axis=2) to get KDE values per dimension:
    return tf.reduce_mean(kernel_vals, axis=2)  # (dims, n_grid)

@tf.function
def kl_div_term(y, p_base):
    batch_size = tf.cast(tf.shape(p_base)[0], tf.float32)
    mu = p_base[:, 0:8:2]  # shape (batch_size, 4)
    Mdia = 1e-9 + tf.math.maximum(p_base[:, 1:8:2], 0.0)  # shape (batch_size, 4)
    Mcov = p_base[:, 8:]  # shape (batch_size, 6)
    
    # Build the lower triangular matrix for each batch sample.
    zeros = tf.zeros_like(Mdia[:, 0])
    row1 = tf.stack([Mdia[:, 0], zeros,      zeros,      zeros], axis=1)  # (batch_size, 4)
    row2 = tf.stack([Mcov[:, 0], Mdia[:, 1],  zeros,      zeros], axis=1)
    row3 = tf.stack([Mcov[:, 1], Mcov[:, 2],  Mdia[:, 2], zeros], axis=1)
    row4 = tf.stack([Mcov[:, 3], Mcov[:, 4],  Mcov[:, 5], Mdia[:, 3]], axis=1)
    scale_tril = tf.stack([row1, row2, row3, row4], axis=1)  # (batch_size, 4, 4)

    scale_tril += 1e-6 * tf.eye(4, batch_shape=[tf.shape(scale_tril)[0]])
    
    # Solve for the pulls via the lower-triangular system.
    residual = y - mu  # (batch_size, 4)
    residual = tf.expand_dims(residual, -1)  # (batch_size, 4, 1)
    pull = tf.linalg.triangular_solve(scale_tril, residual, lower=True)
    pull = tf.squeeze(pull, axis=-1)  # (batch_size, 4)
    
    # Compute KDE for each of the 4 dimensions simultaneously.
    grid = tf.linspace(-5.0, 5.0, 500) 
    bandwidth = 0.3
    
    kde_vals = kde_tf(pull, grid = grid, bandwidth = bandwidth)  # (4, 500)
    
    # Evaluate the standard normal distribution on the grid (same for all dims).
    q_dist = tfd.Normal(loc=0., scale=1.)
    q_vals = q_dist.prob(grid)  # (500,)
    
    # Calculate the empirical KL divergence for each dimension.
    kl_values = empirical_kl(kde_vals, q_vals)  # (4,)
    total_kl = tf.reduce_mean(kl_values)
    return total_kl * batch_size


def custom_loss(loss_type='nll', minval=1e-9, maxval=1e9, n_samples=300):
    def loss_fn(y_true, y_pred):
        mu, scale_tril = build_scale_tril(y_pred, minval)
        dist = tfp.distributions.MultivariateNormalTriL(loc=mu, scale_tril=scale_tril)

        if loss_type.lower() == 'nll':
            likelihood = dist.prob(y_true)
            likelihood = tf.clip_by_value(likelihood, minval, maxval)
            NLL = -1 * tf.math.log(likelihood)
            return tf.reduce_sum(NLL)
        elif loss_type.lower() == 'crps':
            return multivariate_crps_mc(y_true, dist, n_samples)

        elif loss_type.lower() == 'kl_div':
            return kl_div_term(y_true, y_pred)
        else:
            raise ValueError(f"Invalid loss_type '{loss_type}'")
    return loss_fn



'''
import tensorflow as tf
import tensorflow_probability as tfp

def build_scale_tril(p_base, minval=1e-9):
    # p_base shape: [batch, 14]
    # Extract 4 means from indices 0,2,4,6.
    mu = p_base[:, 0:8:2]
    # Diagonals: indices 1,3,5,7; enforce positivity via relu.
    Mdia = minval + tf.nn.relu(p_base[:, 1:8:2])
    # Off-diagonals: indices 8: (6 values)
    Mcov = p_base[:, 8:]
    zeros = tf.zeros_like(Mdia[:, 0])
    row1 = tf.stack([Mdia[:, 0], zeros,      zeros,      zeros], axis=-1)
    row2 = tf.stack([Mcov[:, 0], Mdia[:, 1], zeros,      zeros], axis=-1)
    row3 = tf.stack([Mcov[:, 1], Mcov[:, 2], Mdia[:, 2], zeros], axis=-1)
    row4 = tf.stack([Mcov[:, 3], Mcov[:, 4], Mcov[:, 5], Mdia[:, 3]], axis=-1)
    scale_tril = tf.stack([row1, row2, row3, row4], axis=1)
    return mu, scale_tril

def loss_nll(y, p_base, minval=1e-9, maxval=1e9):
    # Build distribution from model output.
    mu, scale_tril = build_scale_tril(p_base, minval)
    dist = tfp.distributions.MultivariateNormalTriL(loc=mu, scale_tril=scale_tril)
    # Use log_prob for numerical stability.
    log_prob = dist.log_prob(y)
    log_prob = tf.clip_by_value(log_prob, tf.math.log(minval), tf.math.log(maxval))
    return -tf.reduce_mean(log_prob)

def multivariate_crps_mc(y, dist, n_samples=100):
    batch_size = tf.shape(y)[0]
    d = tf.shape(y)[1]  # Expecting d = 4.
    half = n_samples // 2
    # Generate antithetic samples for reduced variance.
    z = tf.random.normal((half, batch_size, d), dtype=y.dtype)
    z = tf.concat([z, -z], axis=0)  # [n_samples, batch, d]
    z = tf.expand_dims(z, -1)        # [n_samples, batch, d, 1]
    L = dist.scale.to_dense()        # [batch, d, d]
    L = tf.expand_dims(L, 0)         # [1, batch, d, d]
    # Compute samples: sample = mu + L @ z.
    samples = tf.matmul(L, z)        # [n_samples, batch, d, 1]
    samples = tf.squeeze(samples, axis=-1)  # [n_samples, batch, d]
    mu = dist.mean()                 # [batch, d]
    samples += tf.expand_dims(mu, 0)   # Broadcast mu to each sample.
    # Compute E1: average distance from each sample to true y.
    samples_t = tf.transpose(samples, perm=[1, 0, 2])  # [batch, n_samples, d]
    diff = samples_t - tf.expand_dims(y, 1)            # [batch, n_samples, d]
    E1 = tf.reduce_mean(tf.norm(diff, axis=-1), axis=1)  # [batch]
    # Compute E2: average pairwise distance among samples for each batch.
    diff_samples = tf.expand_dims(samples_t, 2) - tf.expand_dims(samples_t, 1)
    pairwise = tf.norm(diff_samples, axis=-1)          # [batch, n_samples, n_samples]
    E2 = 0.5 * tf.reduce_mean(pairwise, axis=[1, 2])     # [batch]
    return tf.reduce_mean(E1 - E2)

def custom_loss(loss_type='nll', minval=1e-9, maxval=1e9, n_samples=100):
    # Closure to comply with Keras's loss(y_true, y_pred) signature.
    def loss_fn(y_true, y_pred):
        mu, scale_tril = build_scale_tril(y_pred, minval)
        dist = tfp.distributions.MultivariateNormalTriL(loc=mu, scale_tril=scale_tril)
        if loss_type.lower() == 'nll':
            return loss_nll(y_true, y_pred, minval, maxval)
        elif loss_type.lower() == 'crps':
            return multivariate_crps_mc(y_true, dist, n_samples)
        else:
            raise ValueError(f"Invalid loss_type '{loss_type}'")
    return loss_fn


'''
