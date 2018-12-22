import re

import tensorflow as tf


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

    # def get_weight_decay_loss(self):
    #     params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    #     r1 = self.name + "\/.*\/kernel"
    #     r2 = self.name + "\/.*\/gamma"

    #     l2_norm_loss = 0
    #     for p in params:
    #         if re.search(r1, p.name) or re.search(r2, p.name):
    #             l2_norm_loss += tf.nn.l2_loss(p)

    #     return self.weight_decay_coeff * l2_norm_loss
