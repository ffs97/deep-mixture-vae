import tensorflow as tf

from models import MixtureVAE
import tensorflow as tf

from models import MixtureVAE
from network import FeedForwardNetwork


class GMMOE:
    def __init__(self, name, input_dim, output_dim, n_classes, activation=None, initializer=None):
        self.name = name

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.n_classes = n_classes

        self.activation = activation
        self.initializer = initializer

    def build_graph(self, network_layer_sizes):
        self.X = tf.placeholder(
            tf.float32, shape=(None, self.input_dim), name="X"
        )
        self.Y = tf.placeholder(
            tf.float32, shape=(None, self.output_dim), name="Y"
        )

        self.logits_network = FeedForwardNetwork(
            name="logits_network"
        )
        self.logits = self.logits_network.build(
            [("logits", self.n_classes)],
            network_layer_sizes, self.X
        )

        self.regression_biases = tf.get_variable(
            "regression_biases", dtype=tf.float32,
            initializer=tf.initializers.zeros,
            shape=(self.output_dim, self.n_classes)
        )
        self.regression_weights = tf.get_variable(
            "regression_weights", dtype=tf.float32,
            initializer=tf.initializers.random_normal,
            shape=(self.n_classes, self.output_dim, self.input_dim)
        )

        self.cluster_probs = tf.nn.softmax(self.logits, axis=-1)

        self.reconstructed_Y_k = tf.transpose(tf.matmul(
            self.regression_weights,
            tf.tile(
                tf.transpose(self.X)[None, :, :], [self.n_classes, 1, 1]
            )
        )) + self.regression_biases

        self.reconstructed_Y = tf.reduce_sum(
            self.reconstructed_Y_k * self.cluster_probs[:, None, :], axis=-1
        )

        self.error = tf.reduce_mean(tf.square(self.reconstructed_Y - self.Y))

    def square_error(self, session, data):
        return session.run(self.error, feed_dict={
            self.X: data.data,
            self.Y: data.labels
        })

    def define_train_loss(self):
        self.loss = - tf.log(tf.reduce_sum(
            self.cluster_probs * tf.exp(-0.5 * tf.reduce_sum(
                tf.square(self.reconstructed_Y_k - self.Y[:, :, None]), axis=1
            ))
        ))

    def define_train_step(self, init_lr, decay_steps, decay_rate=0.9):
        learning_rate = tf.train.exponential_decay(
            learning_rate=init_lr,
            global_step=0,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )

        self.define_train_loss()
        self.train_step = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        ).minimize(self.loss)

    def train_op(self, session, data):
        assert(self.train_step is not None)

        loss = 0.0
        for X_batch, Y_batch, _ in data.get_batches():
            feed = {
                self.X: X_batch,
                self.Y: Y_batch
            }

            batch_loss, _ = session.run(
                [self.loss, self.train_step],
                feed_dict=feed
            )
            loss += batch_loss / data.epoch_len

        return loss


class VMOE:
    def __init__(self, name, input_dim, latent_dim, output_dim, n_classes, activation=None, initializer=None):
        self.name = name

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.n_classes = n_classes

        self.activation = activation
        self.initializer = initializer

    def build_graph(self, encoder_layer_sizes, decoder_layer_sizes):
        self.vae = MixtureVAE(
            "mixture_vae", self.input_dim, self.latent_dim, self.n_classes,
            activation=self.activation, initializer=self.initializer
        )
        self.vae.build_graph(encoder_layer_sizes, decoder_layer_sizes)

        self.X = self.vae.X
        self.Y = tf.placeholder(
            tf.float32, shape=(None, self.output_dim), name="Y"
        )

        self.logits = self.vae.logits

        self.regression_biases = tf.get_variable(
            "regression_biases", dtype=tf.float32,
            initializer=tf.initializers.zeros,
            shape=(self.output_dim, self.n_classes)
        )
        self.regression_weights = tf.get_variable(
            "regression_weights", dtype=tf.float32,
            initializer=tf.initializers.random_normal,
            shape=(self.n_classes, self.output_dim, self.input_dim)
        )

        self.cluster_probs = tf.nn.softmax(self.logits, axis=-1)

        self.reconstructed_Y_k = tf.transpose(tf.matmul(
            self.regression_weights,
            tf.tile(
                tf.transpose(self.X)[None, :, :], [self.n_classes, 1, 1]
            )
        )) + self.regression_biases

        self.reconstructed_Y = tf.reduce_sum(
            self.reconstructed_Y_k * self.cluster_probs[:, None, :], axis=-1
        )

        self.error = tf.reduce_mean(tf.square(self.reconstructed_Y - self.Y))

    def square_error(self, session, data):
        return session.run(self.error, feed_dict={
            self.X: data.data,
            self.Y: data.labels
        })

    def define_train_loss(self):
        self.vae.define_train_loss()

        self.recon_loss = - tf.log(tf.reduce_sum(
            self.cluster_probs * tf.exp(-0.5 * tf.reduce_sum(
                tf.square(self.reconstructed_Y_k - self.Y[:, :, None]), axis=1
            ))
        ))

        self.loss = self.vae.loss + self.recon_loss

    def define_train_step(self, init_lr, decay_steps, decay_rate=0.9):
        learning_rate = tf.train.exponential_decay(
            learning_rate=init_lr,
            global_step=0,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )

        self.define_train_loss()
        self.train_step = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        ).minimize(self.loss)

    def train_op(self, session, data):
        assert(self.train_step is not None)

        loss = 0.0
        for X_batch, Y_batch, _ in data.get_batches():
            feed = {
                self.X: X_batch,
                self.Y: Y_batch
            }
            feed.update(
                self.vae.sample_reparametrization_variables(
                    len(X_batch), feed=False)
            )

            batch_loss, _ = session.run(
                [self.loss, self.train_step],
                feed_dict=feed
            )
            loss += batch_loss / data.epoch_len

        return loss
