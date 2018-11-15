import priors
import tensorflow as tf

from includes.network import FeedForwardNetwork


class VAE:
    def __init__(self, name, input_type, input_dim, latent_dim, activation=None, initializer=None):
        self.name = name

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.input_type = input_type

        self.activation = activation
        self.initializer = initializer

        self.X = None
        self.decoded_X = None
        self.train_step = None
        self.latent_variables = dict()

    def build_graph(self, encoder_layer_sizes, decoder_layer_sizes):
        raise NotImplementedError

    def sample_reparametrization_variables(self, n):
        samples = dict()
        for lv, eps, _ in self.latent_variables.values():
            samples[eps] = lv.sample_reparametrization_variable(n)

        return samples

    def sample_generative_feed(self, n, **kwargs):
        samples = dict()
        for name, (lv, _, _) in self.latent_variables.items():
            kwargs_ = dict() if name not in kwargs else kwargs[name]
            samples[name] = lv.sample_generative_feed(n, **kwargs_)

        return samples

    def define_latent_loss(self):
        self.latent_loss = tf.add_n(
            [lv.kl_from_prior(params)
             for lv, _, params in self.latent_variables.values()]
        )

    def define_recon_loss(self):
        if self.input_type == "binary":
            self.recon_loss = tf.reduce_mean(tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.X,
                    logits=self.decoded_X
                ), axis=1
            ))
        elif self.input_type == "real":
            self.recon_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(
                tf.square(self.X - self.decoded_X), axis=1
            ))
        else:
            raise NotImplementedError

    def define_train_loss(self):
        self.define_latent_loss()
        self.define_recon_loss()

        self.loss = tf.reduce_mean(self.recon_loss + self.latent_loss)

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
        for batch in data.get_batches():
            feed = {
                self.X: batch
            }
            feed.update(
                self.sample_reparametrization_variables(len(batch))
            )

            batch_loss, _ = session.run(
                [self.loss, self.train_step],
                feed_dict=feed
            )
            loss += batch_loss / data.epoch_len

        return loss


class CollapsedMixtureVAE(VAE):
    def __init__(self, name, input_type, input_dim, latent_dim, n_classes, activation=None, initializer=None):
        VAE.__init__(self, name, input_dim, latent_dim,
                     activation=activation, initializer=initializer)

        self.n_classes = n_classes

    def build_graph(self, encoder_layer_sizes, decoder_layer_sizes):
        with tf.variable_scope(self.name) as _:
            self.X = tf.placeholder(
                tf.float32, shape=(None, self.input_dim), name="X"
            )
            self.epsilon = tf.placeholder(
                tf.float32, shape=(None, self.latent_dim), name="epsilon_Z"
            )
            self.cluster = tf.placeholder(
                tf.float32, shape=(None, 1, self.n_classes), name="epsilon_C"
            )
            self.temperature = tf.placeholder_with_default(
                1.0, shape=None, name="temperature"
            )

            self.latent_variables = dict()

            self.c_encoder_network = FeedForwardNetwork(
                name="c/encoder_network"
            )
            self.logits = self.c_encoder_network.build(
                [("logits", self.n_classes)],
                encoder_layer_sizes["C"], self.X
            )

            self.latent_variables.update({
                "C": (
                    priors.DiscreteFactorial(
                        "cluster", 1, self.n_classes
                    ), self.cluster,
                    {
                        "logits": self.logits,
                        "temperature": self.temperature
                    }
                )
            })
            lv, eps, params = self.latent_variables["C"]
            self.C = lv.inverse_reparametrize(eps, params)

            self.means = list()
            self.log_vars = list()

            self.z_encoder_networks = [
                FeedForwardNetwork(name="z/encoder_network_%d" % k)
                for k in range(self.n_classes)
            ]
            for k in range(self.n_classes):
                mean, log_var = self.z_encoder_networks[k].build(
                    [("mean", self.latent_dim),
                     ("log_var", self.latent_dim)],
                    encoder_layer_sizes["Z"], self.X
                )

                self.means.append(mean)
                self.log_vars.append(log_var)

            self.mean = tf.add_n([
                self.means[i] * self.C[:, :, i]
                for i in range(self.n_classes)
            ])
            self.log_var = tf.log(tf.add_n([
                tf.exp(self.log_vars[i]) * self.C[:, :, i]
                for i in range(self.n_classes)
            ]))

            self.latent_variables.update({
                "Z": (
                    priors.NormalFactorial(
                        "representation", self.latent_dim
                    ), self.epsilon,
                    {
                        "mean": self.mean,
                        "log_var": self.log_var
                    }
                )
            })

            lv, eps, params = self.latent_variables["Z"]
            self.Z = lv.inverse_reparametrize(eps, params)

            self.decoder_network = FeedForwardNetwork(name="decoder_network")
            self.decoded_X = self.decoder_network.build(
                [("decoded_X", self.input_dim)], decoder_layer_sizes, self.Z
            )

            if self.input_type == "binary":
                self.reconstructed_X = tf.nn.sigmoid(self.decoded_X)
            elif self.input_type == "real":
                self.reconstructed_X = self.decoded_X
            else:
                raise NotImplementedError

        return self


class DiscreteMixtureVAE(VAE):
    def __init__(self, name, input_type, input_dim, latent_dim, n_classes, activation=None, initializer=None):
        VAE.__init__(self, name, input_type, input_dim, latent_dim,
                     activation=activation, initializer=initializer)

        self.n_classes = n_classes

    def build_graph(self, encoder_layer_sizes, decoder_layer_sizes):
        with tf.variable_scope(self.name) as _:
            self.X = tf.placeholder(
                tf.float32, shape=(None, self.input_dim), name="X"
            )
            self.epsilon = tf.placeholder(
                tf.float32, shape=(None, self.latent_dim), name="epsilon_Z"
            )
            self.cluster = tf.placeholder(
                tf.float32, shape=(None, 1, self.n_classes), name="epsilon_C"
            )
            self.temperature = tf.placeholder_with_default(
                0.2, shape=None, name="temperature"
            )

            self.latent_variables = dict()

            self.c_encoder_network = FeedForwardNetwork(
                name="c/encoder_network"
            )
            self.logits = self.c_encoder_network.build(
                [("logits", self.n_classes)],
                encoder_layer_sizes["C"], self.X
            )

            self.latent_variables.update({
                "C": (
                    priors.DiscreteFactorial(
                        "cluster", 1, self.n_classes
                    ), self.cluster,
                    {
                        "logits": self.logits,
                        "temperature": self.temperature
                    }
                )
            })
            lv, eps, params = self.latent_variables["C"]
            self.C = lv.inverse_reparametrize(eps, params)

            self.z_encoder_network = FeedForwardNetwork(
                name="z/encoder_network"
            )
            self.mean, self.log_var = self.z_encoder_network.build(
                [("mean", self.latent_dim),
                 ("log_var", self.latent_dim)],
                encoder_layer_sizes["Z"], self.X
            )

            self.latent_variables.update({
                "Z": (
                    priors.NormalMixtureFactorial(
                        "representation", self.latent_dim, self.n_classes
                    ), self.epsilon,
                    {
                        "mean": self.mean,
                        "log_var": self.log_var,
                        "weights": self.C
                    }
                )
            })

            lv, eps, params = self.latent_variables["Z"]
            self.Z = lv.inverse_reparametrize(eps, params)

            self.decoder_network = FeedForwardNetwork(name="decoder_network")
            self.decoded_X = self.decoder_network.build(
                [("decoded_X", self.input_dim)], decoder_layer_sizes, self.Z
            )

            if self.input_type == "binary":
                self.reconstructed_X = tf.nn.sigmoid(self.decoded_X)
            elif self.input_type == "real":
                self.reconstructed_X = self.decoded_X
            else:
                raise NotImplementedError

        return self


class DeepMoE:
    def __init__(self, name, input_dim, output_dim, n_classes, activation=None, initializer=None):
        self.name = name

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.n_classes = n_classes

        self.activation = activation
        self.initializer = initializer

    def build_graph(self, network_layer_sizes):
        with tf.variable_scope(self.name) as _:
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

            self.error = tf.reduce_mean(
                tf.square(self.reconstructed_Y - self.Y)
            )

            return self

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


class DVMoE:
    def __init__(self, name, input_type, input_dim, latent_dim, output_dim, n_classes, activation=None, initializer=None):
        self.name = name

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.input_type = input_type

        self.n_classes = n_classes

        self.activation = activation
        self.initializer = initializer

    def build_graph(self, encoder_layer_sizes, decoder_layer_sizes):
        with tf.variable_scope(self.name) as _:
            self.vae = DiscreteMixtureVAE(
                "mixture_vae", self.input_type, self.input_dim, self.latent_dim,
                self.n_classes, activation=self.activation, initializer=self.initializer
            )
            self.vae.build_graph(encoder_layer_sizes, decoder_layer_sizes)

            self.X = self.vae.X
            self.Z = self.vae.Z
            self.Y = tf.placeholder(
                tf.float32, shape=(None, self.output_dim), name="Y"
            )

            self.logits = self.vae.logits

            self.reconstructed_X = self.vae.reconstructed_X

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

            self.error = tf.reduce_mean(
                tf.square(self.reconstructed_Y - self.Y)
            )

            return self

    def sample_generative_feed(self, n, **kwargs):
        return self.vae.sample_generative_feed(n, **kwargs)

    def sample_reparametrization_variables(self, n):
        return self.vae.sample_reparametrization_variables(n)

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
                self.vae.sample_reparametrization_variables(len(X_batch))
            )

            batch_loss, _ = session.run(
                [self.loss, self.train_step],
                feed_dict=feed
            )
            loss += batch_loss / data.epoch_len

        return loss
