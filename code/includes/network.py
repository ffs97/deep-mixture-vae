import tensorflow as tf

from includes.layers import FullyConnected, Convolution, MaxPooling, BatchNormalization


_layers_id_mapping = {
    "fc": FullyConnected,
    "cv": Convolution,
    "mp": MaxPooling,
    "bn": BatchNormalization
}


class FeedForwardNetwork:
    def __init__(self, name, activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer, dropout=0.0):
        self.name = name
        self.keep_prob = 1 - dropout

        self.activation = activation
        self.initializer = initializer

    def build(self, output_dims, layer_sizes, input_var, reuse=False):
        layers = []
        with tf.variable_scope(self.name, reuse=reuse) as _:
            input_var = tf.layers.flatten(input_var)

            for index, layer_size in enumerate(layer_sizes):
                layers.append(
                    tf.layers.dense(
                        input_var if index == 0 else layers[index - 1],
                        layer_size,
                        activation=self.activation,
                        kernel_initializer=self.initializer(),
                        name="network_layer_" + str(index + 1)
                    )
                )

            self.outputs = []
            for name, output_dim in output_dims:
                self.outputs.append(
                    tf.layers.dense(
                        layers[-1],
                        output_dim,
                        kernel_initializer=self.initializer(),
                        name="network_output/" + name
                    )
                )

        self.layers = layers

        if len(self.outputs) == 1:
            return self.outputs[0]

        return self.outputs


class DeepNetwork:
    def __init__(self, name, layers, outputs, activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer):
        self.name = name

        self.layers = []
        with tf.variable_scope(self.name) as _:
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
