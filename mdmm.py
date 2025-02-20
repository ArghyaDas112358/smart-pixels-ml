import tensorflow as tf
from tensorflow.keras import Model, layers
import abc

@tf.keras.utils.register_keras_serializable(name='Constraint')
class Constraint(layers.Layer):
    """Base class for constraints."""
    def __init__(self, scale=1.0, damping=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer=tf.constant_initializer(scale),
            trainable=False
        )
        self.damping = self.add_weight(
            name='damping',
            shape=(),
            initializer=tf.constant_initializer(damping),
            trainable=False
        )
        self.lmbda = self.add_weight(
            name=self.name + '_lmbda',
            shape=(),
            initializer=tf.zeros_initializer(),
            trainable=True
        )

    def call(self, inputs):
        fn_value = self.fn(inputs)
        inf = self.infeasibility(fn_value)  # Add the absolute of the value
        l_term = tf.math.maximum(self.lmbda, 0.0) * inf  # make lmbda to be also positive
        damp_term = self.damping * tf.square(inf) / 2
        penalty = self.scale * (l_term + damp_term)
        return penalty

    @abc.abstractmethod
    def fn(self, inputs):
        raise NotImplementedError("Subclasses should implement fn() method")

    @abc.abstractmethod
    def infeasibility(self, fn_value):
        raise NotImplementedError("Subclasses should implement infeasibility() method")

    @abc.abstractmethod
    def compute_update_lmbda(self):
        raise NotImplementedError("Subclasses should implement compute_update_lmbda() method")

    def get_config(self):
        config = super().get_config()
        config.update({
            "scale": self.scale.numpy(),
            "damping": self.damping.numpy(),
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Equality Constraints
@tf.keras.utils.register_keras_serializable(name='EqConstraint')
class EqConstraint(Constraint):  # fn_value = target_value
    """Constraint that equals to a given value."""
    def __init__(self, layer, target_value, scale=1.0, damping=1.0, abs_inf=False):
        super().__init__(scale, damping)
        self.layer = layer
        self.target_value = target_value
        self.abs_inf = abs_inf

    def infeasibility(self, fn_value):
        if self.abs_inf:
            return abs(self.target_value - fn_value)
        return self.target_value - fn_value

    def get_config(self):
        config = super().get_config()
        config.update({
            "layer": self.layer,
            "target_value": self.target_value,
        })
        return config


class EqL1Constraint(EqConstraint):
    def __init__(self, layer, target_sparsity, scale=1.0, damping=1.0, epsilon=1e-5, lr_multiplier=1.0):
        super().__init__(layer, target_value=0.0, scale=scale, damping=damping, abs_inf=True)
        
        assert 0 <= target_sparsity <= 1, "target_sparsity must be between 0 and 1"
        self.target_sparsity = target_sparsity
        self.epsilon = epsilon
        self.lr_multiplier = lr_multiplier

        self.weights_list = []
        if isinstance(layer, list):
            for l in layer:
                self.weights_list.append(l.weights[0])
        else:
            self.weights_list.append(layer.weights[0])

    def fn(self, inputs):
        weights_concat = tf.concat([tf.reshape(w, [-1]) for w in self.weights_list], axis=0)
        num_weights = tf.cast(tf.size(weights_concat), tf.float32)
        zero_weights = tf.less_equal(tf.abs(weights_concat), self.epsilon)
        zero_count = tf.reduce_sum(tf.cast(zero_weights, tf.float32))
        l1_term = tf.reduce_mean(tf.abs(weights_concat))

        target_zero_count = tf.math.ceil(num_weights * self.target_sparsity)
        factor = (target_zero_count - zero_count) / num_weights

        fn_value = tf.math.maximum(factor, 0.0) * l1_term # abs(tf.nn.leaky_relu(factor)) *
        return fn_value

    def compute_update_lmbda(self):
        # update_lmbda here (call it in GradientTape)
        weights_concat = tf.concat([tf.reshape(w, [-1]) for w in self.weights_list], axis=0)
        num_weights = tf.cast(tf.size(weights_concat), tf.float32)
        zero_weights = tf.less_equal(tf.abs(weights_concat), self.epsilon)
        zero_count = tf.reduce_sum(tf.cast(zero_weights, tf.float32))

        target_zero_count = tf.math.ceil(num_weights * self.target_sparsity)
        factor = (target_zero_count - zero_count) / num_weights
        factor_2 = tf.math.pow(factor, 2)

        new_update_lmbda = tf.where(
            (factor >= 1e-6) & (factor_2 > 1e-6),
            self.lr_multiplier * factor_2,
            1e-6
        )
        return new_update_lmbda


class EqL2Constraint(EqConstraint):
    def __init__(self, layer, target_fraction_small, threshold=0.2, scale=1.0, damping=1.0):
        super().__init__(layer, target_value=0.0, scale=scale, damping=damping, abs_inf=True)
        
        assert 0 <= target_fraction_small <= 1, "target_fraction_small must be between 0 and 1"
        self.target_fraction = target_fraction_small
        self.threshold = threshold

        self.weights_list = []
        if isinstance(layer, list):
            for l in layer:
                self.weights_list.append(l.weights[0])
        else:
            self.weights_list.append(layer.weights[0])

    def fn(self, inputs):
        weights_concat = tf.concat([tf.reshape(w, [-1]) for w in self.weights_list], axis=0)
        num_weights = tf.cast(tf.size(weights_concat), tf.float32)
        small_weights = tf.less_equal(tf.abs(weights_concat), self.threshold)
        small_count = tf.reduce_sum(tf.cast(small_weights, tf.float32))

        target_small_count = tf.math.ceil(num_weights * self.target_fraction)
        factor = (target_small_count - small_count) / num_weights
        l2_term = tf.reduce_mean(tf.square(weights_concat))

        fn_value = tf.maximum(factor, 0.0) * l2_term
        return fn_value

    def compute_update_lmbda(self):
        weights_concat = tf.concat([tf.reshape(w, [-1]) for w in self.weights_list], axis=0)
        num_weights = tf.cast(tf.size(weights_concat), tf.float32)
        small_weights = tf.less_equal(tf.abs(weights_concat), self.threshold)
        small_count = tf.reduce_sum(tf.cast(small_weights, tf.float32))

        target_small_count = tf.math.ceil(num_weights * self.target_fraction)
        factor = (target_small_count - small_count) / num_weights
        factor_squared = tf.square(factor)

        new_update_lmbda = tf.where(
            (factor >= 1e-6) & (factor_squared > 1e-6),
            factor_squared,
            1e-6
        )
        return new_update_lmbda

# Model

# Graph execution turned off here.
# tf.config.run_functions_eagerly(True)

@tf.keras.utils.register_keras_serializable(name='MDMM')
class MDMM(Model):
    """MDMM model that applies multiple constraints."""
    def __init__(self, model, constraints=None, name='MDMM', **kwargs):
        super().__init__(name=name, **kwargs)
        self.model = model
        self.constraints = constraints if constraints is not None else []
        for constraint in self.constraints:
            setattr(self, constraint.name, constraint)

    @tf.function
    def call(self, inputs, training=False):
        x = inputs
        penalties = {}
        visited_constraints = []

        for layer in self.model.layers:
            x = layer(x)

            if training:
                for constraint in self.constraints:
                    if hasattr(constraint, 'layer') and constraint.layer == layer:
                        penalties["loss_" + constraint.name] = constraint(x)
                        visited_constraints.append(constraint.name)

        if training:
            for constraint in self.constraints:
                if constraint.name not in visited_constraints:
                    penalties["loss_" + constraint.name] = constraint(x)

        return (x, penalties) if penalties else x

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            if isinstance(outputs, tuple):
                logits, penalties = outputs
            else:
                logits = outputs
                penalties = {}

            valid_penalties = [v for v in penalties.values() if v is not None]
            if valid_penalties:
                penalty = tf.add_n(valid_penalties)
            else:
                penalty = tf.constant(0.0)

            loss_obj = self.compiled_loss(y, logits)
            loss = loss_obj + penalty

            # update_lmbda inside GradientTape
            update_lmbdas = {}
            for constraint in self.constraints:
                if hasattr(constraint, 'compute_update_lmbda'):
                    update_lmbda = constraint.compute_update_lmbda()
                    update_lmbdas[constraint.name] = update_lmbda

        grads = tape.gradient(loss, self.trainable_variables)
        adjusted_grads_and_vars = []

        for grad, var in zip(grads, self.trainable_variables):
            if 'lmbda' in var.name:
                for constraint in self.constraints:
                    if var is constraint.lmbda:
                        update_lmbda = update_lmbdas.get(constraint.name, 1.0)
                        adjusted_grads_and_vars.append((update_lmbda * -grad, var))
            else:
                adjusted_grads_and_vars.append((grad, var))

        self.optimizer.apply_gradients(adjusted_grads_and_vars)

        out = {"loss": loss, "loss_obj": loss_obj}
        out.update(penalties)

        return out

    def summary(self):
        return self.model.summary()

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.model.save_weights(filepath, overwrite, save_format, options)

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        self.model.load_weights(filepath, by_name, skip_mismatch, options)

    def get_config(self):
        model_config = self.model.get_config()
        constraints_config = [tf.keras.utils.serialize_keras_object(c) for c in self.constraints]
        config = super().get_config()
        config.update({
            "model": model_config,
            "constraints": constraints_config,
        })
        return config

    @classmethod
    def from_config(cls, config):
        model = tf.keras.models.Model.from_config(config['model'])
        constraints = [tf.keras.layers.deserialize(c) for c in config['constraints']]
        return cls(model=model, constraints=constraints)
