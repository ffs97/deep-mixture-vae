import tensorflow as tf

from includes.layers import *


_layers_id_mapping = {
    "fc": FullyConnected,
    "cn": Convolution,
    "mp": MaxPooling,
    "bn": BatchNormalization
}


class DeepNetwork:
    def __init__(self, name, layers, outputs, activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer):
        self.name = name

        self.layers = []
        with tf.variable_scope(self.name):
            for index, (layer_id, args) in enumerate(layers):
                name = "layer_%d" % (index + 1)

                if layer_id not in _layers_id_mapping:
                    raise NotImplementedError

                self.layers.append(
                    _layers_id_mapping[layer_id](
                        name, activation=activation, initializer=initializer, **args
                    )
                )

            self.output_layers = []
            for index, (layer_id, args) in enumerate(outputs):
                name = "output_%d" % (index + 1)

                if layer_id not in _layers_id_mapping:
                    raise NotImplementedError

                self.output_layers.append(
                    _layers_id_mapping[layer_id](
                        name, activation=None, initializer=initializer, **args
                    )
                )

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = inputs

            for layer in self.layers:
                outputs = layer(outputs)

            outputs = [layer(outputs) for layer in self.output_layers]

            if len(outputs) == 1:
                outputs = outputs[0]

            return outputs
