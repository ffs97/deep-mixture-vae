import priors
import tensorflow as tf

from includes.utils import Dataset
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

class VaDE(VAE):
    def __init__(self, name, input_type, input_dim, latent_dim, n_classes, activation=None, initializer=None):
        VAE.__init__(self, name, input_type, input_dim, latent_dim,
                     activation=activation, initializer=initializer)
        self.n_classes = n_classes
        self.batch_size = 200

    def build_graph(self, encoder_layer_sizes, decoder_layer_sizes):
        with tf.variable_scope(self.name) as _:
            self.X = tf.placeholder(
                tf.float32, shape=(None, self.input_dim), name="X"
            )
            self.epsilon = tf.placeholder(
                tf.float32, shape=(None, self.latent_dim), name="epsilon_Z"
            )
            # make the priors trainable
            self.prior_means = tf.Variable(
                tf.random_normal((self.n_classes, self.latent_dim), stddev=5.0),
                dtype=tf.float32,
                name="prior_means"
            )
            self.prior_vars = tf.Variable(
                tf.ones((self.n_classes, self.latent_dim)),
                dtype=tf.float32,
                name="prior_vars"
            )
            self.prior_weights = tf.Variable(
                tf.ones((self.n_classes)) / self.n_classes,
                dtype=tf.float32,
                name="prior_weights"
            )

            self.encoder_network = FeedForwardNetwork(name="vae_encoder")
            self.mean, self.log_var = self.encoder_network.build(
                [("mean", self.latent_dim),
                 ("log_var", self.latent_dim)],
                encoder_layer_sizes, self.X
            )

            self.latent_variables = dict()
            self.latent_variables.update({
                "Z": (
                    priors.NormalFactorial("representation", self.latent_dim),
                    self.epsilon,
                    {
                        "mean": self.mean,
                        "log_var": self.log_var,
                    }
                )
            })

            lv, eps, params = self.latent_variables["Z"]
            self.Z = lv.inverse_reparametrize(eps, params)

            self.cluster_weights = self.find_cluster_weights()

            self.decoder_network = FeedForwardNetwork(name="vae_decoder")
            self.decoded_X = self.decoder_network.build(
                [("vae_decoder", self.input_dim)], decoder_layer_sizes, self.Z
            )

            if self.input_type == "binary":
                self.reconstructed_X = tf.nn.sigmoid(self.decoded_X)
            elif self.input_type == "real":
                self.reconstructed_X = self.decoded_X
            else:
                raise NotImplementedError

    def find_cluster_weights(self):
        def fn_cluster(_, k):
            q = self.prior_weights[k] *\
                tf.contrib.distributions.MultivariateNormalDiag(
                        loc=self.prior_means[k],
                        scale_diag=self.prior_vars[k]
                ).prob(self.Z) + 1e-10
            return tf.reshape(q, [self.batch_size])

        clusters = tf.Variable(tf.range(self.n_classes))
        probs = tf.scan(fn_cluster, clusters, initializer=tf.ones([self.batch_size]))
        probs = tf.transpose(probs)
        probs = probs / tf.reshape(tf.reduce_sum(probs, 1), (-1, 1))
        return probs

    def define_train_loss(self):
        self.define_latent_loss()
        self.define_recon_loss()
        self.vae_loss = self.recon_loss + self.latent_loss

        J = 0
        J += self.vae_loss
        J -= tf.reduce_sum(self.cluster_weights * tf.log(self.prior_weights), axis=1)
        J += tf.reduce_sum(self.cluster_weights * tf.log(self.cluster_weights), axis=1)

        def fn_cluster(prev_out, curr_inp):
            k = curr_inp
            return prev_out + 0.5 * self.cluster_weights[:, k] * tf.reduce_sum(
                tf.log(self.prior_vars[k]) +
                (
                    tf.exp(self.log_var) +
                    tf.square(self.mean - self.prior_means[k])
                ) / self.prior_vars[k], axis=1
            )

        clusters = tf.Variable(tf.range(self.n_classes))
        J += tf.scan(fn_cluster, clusters, initializer=tf.zeros(self.batch_size))[-1, :]

        self.loss = tf.reduce_mean(J)

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
            ) * self.output_dim

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
            ) * self.output_dim

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

    def pretrain(self, session, data, n_epochs):
        data = Dataset(data.data, data.batch_size, data.shuffle)
        for _ in range():
            self.vae.train_op(session, data)

    def define_train_step(self, init_lr, decay_steps, decay_rate=0.9, pretrain_init_lr=None,
                          pretrain_decay_steps=None, pretrain_decay_rate=None):
        self.define_train_loss()

        if pretrain_init_lr is None:
            pretrain_init_lr = init_lr
        if pretrain_decay_rate is None:
            pretrain_decay_rate = decay_rate
        if pretrain_decay_steps is None:
            pretrain_decay_steps = decay_steps

        self.vae.define_train_step(
            pretrain_init_lr, pretrain_decay_steps, pretrain_decay_rate)

        learning_rate = tf.train.exponential_decay(
            learning_rate=pretrain_init_lr,
            global_step=0,
            decay_steps=pretrain_decay_steps,
            decay_rate=pretrain_decay_rate
        )

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


class VaDEMoE:
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
            self.vae = VaDE(
                "vade_vae", self.input_type, self.input_dim, self.latent_dim,
                self.n_classes, activation=self.activation, initializer=self.initializer
            )
            self.vae.build_graph(encoder_layer_sizes, decoder_layer_sizes)

            self.X = self.vae.X
            self.Z = self.vae.Z
            self.Y = tf.placeholder(
                tf.float32, shape=(None, self.output_dim), name="Y"
            )

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

            self.cluster_probs = self.vae.cluster_weights

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
            ) * self.output_dim

            return self

    def sample_generative_feed(self, n, **kwargs):
        return self.vae.sample_generative_feed(n, **kwargs)

    def sample_reparametrization_variables(self, n):
        return self.vae.sample_reparametrization_variables(n)

    def square_error(self, session, data):
        error = 0
        for X_batch, Y_batch, _ in data.get_batches():
            feed = {
                self.X: X_batch,
                self.Y: Y_batch,
            }
            feed.update(
                self.sample_reparametrization_variables(len(X_batch))
            )
            batch_error = session.run(self.error, feed_dict=feed)
            error += batch_error
        return error

    def define_train_loss(self):
        self.vae.define_train_loss()

        self.recon_loss = - tf.log(tf.reduce_sum(
            self.cluster_probs * tf.exp(-0.5 * tf.reduce_sum(
                tf.square(self.reconstructed_Y_k - self.Y[:, :, None]), axis=1
            ))
        ))

        self.loss = self.vae.loss + self.recon_loss

    def pretrain(self, session, data, n_epochs):
        data = Dataset(data.data, data.batch_size, data.shuffle)
        for _ in range(n_epochs):
            self.vae.train_op(session, data)

    def define_train_step(self, init_lr, decay_steps, decay_rate=0.9, pretrain_init_lr=None,
                          pretrain_decay_steps=None, pretrain_decay_rate=None):
        self.define_train_loss()

        if pretrain_init_lr is None:
            pretrain_init_lr = init_lr
        if pretrain_decay_rate is None:
            pretrain_decay_rate = decay_rate
        if pretrain_decay_steps is None:
            pretrain_decay_steps = decay_steps

        self.vae.define_train_step(
            pretrain_init_lr, pretrain_decay_steps, pretrain_decay_rate)

        learning_rate = tf.train.exponential_decay(
            learning_rate=pretrain_init_lr,
            global_step=0,
            decay_steps=pretrain_decay_steps,
            decay_rate=pretrain_decay_rate
        )

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

    def testNow(self, session, data):
        for X_batch, Y_batch, _ in data.get_batches():
            feed = {
                self.X: X_batch,
                self.Y: Y_batch
            }
            feed.update(
                self.vae.sample_reparametrization_variables(len(X_batch))
            )
            print(session.run([self.cluster_probs], feed_dict=feed))
            break
