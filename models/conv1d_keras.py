import tensorflow as tf
from ray.rllib.utils.annotations import override
# from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf.misc import normc_initializer
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, GlobalAveragePooling1D, Dropout


class KerasConv1d(TFModelV2):
    pass


class KerasQConv1d(DistributionalQTFModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        super(KerasQConv1d, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)
        training = model_config['custom_model_config']['training']
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        
        conv_1 = tf.keras.layers.Conv1D(filters=128, kernel_size=5, input_shape=obs_space.shape, activation=tf.nn.leaky_relu, kernel_initializer=normc_initializer(1.0))(self.inputs)
        conv_2 = tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation=tf.nn.leaky_relu, kernel_initializer=normc_initializer(1.0))(conv_1)
        flatten_2 = tf.keras.layers.Flatten()(conv_2)
        dense3 = tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu, name="dense2", kernel_initializer=normc_initializer(1.0))(flatten_2)
        # if training:
        #     dense3 = tf.keras.layers.Dropout(rate=0.5)(dense3)
        # dense3 = tf.keras.layers.LayerNormalization()(dense3)
        """
        conv1 = Conv1D(filters=64, kernel_size=3, input_shape=obs_space.shape, activation=tf.nn.relu, kernel_initializer=normc_initializer(1.0))(self.inputs)
        conv1 = Conv1D(filters=64, kernel_size=3, activation=tf.nn.relu, kernel_initializer=normc_initializer(1.0))(conv1)
        conv1 = MaxPooling1D(pool_size=3)(conv1)
        conv2 = Conv1D(filters=128, kernel_size=3, activation=tf.nn.relu, kernel_initializer=normc_initializer(1.0))(conv1)
        conv2 = Conv1D(filters=128, kernel_size=3, activation=tf.nn.relu, kernel_initializer=normc_initializer(1.0))(conv2)
        dense3 = GlobalAveragePooling1D()(conv2)
        
        if training:
            dense3 = Dropout(rate=0.5)(dense3)
        """
        layer_out = tf.keras.layers.Dense(num_outputs, name="my_out", activation=tf.nn.leaky_relu, kernel_initializer=normc_initializer(1.0))(dense3)
        value_out = tf.keras.layers.Dense(1, name="value_out", activation=None, kernel_initializer=normc_initializer(0.01))(dense3)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.register_variables(self.base_model.variables)

    @override(DistributionalQTFModel)
    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    @override(DistributionalQTFModel)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    @override(DistributionalQTFModel)
    def metrics(self):
        return {"foo": tf.constant(42.0)}
