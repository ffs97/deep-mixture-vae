import priors
import numpy as np
import tensorflow as tf

from sklearn.mixture import GaussianMixture

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

    def sample_reparametrization_variables(self, n, variables=None):
        samples = dict()
        if variables is None:
            for lv, eps, _ in self.latent_variables.values():
                samples[eps] = lv.sample_reparametrization_variable(n)
        else:
            for var in variables:
                lv, eps, _ = self.latent_variables[var]
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
                        "weights": self.C,
                        "cluster_sample": True
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

    def define_pretrain_step(self, init_lr, decay_steps, decay_rate=0.9):
        assert(self.train_step is not None)

        learning_rate = tf.train.exponential_decay(
            learning_rate=init_lr,
            global_step=0,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )

        pretrain_loss = tf.reduce_mean(
            self.recon_loss + 0.5 * tf.reduce_sum(
                tf.exp(self.log_var) + self.mean ** 2 - self.log_var,
                axis=-1
            )
        )

        self.pretrain_step = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        ).minimize(pretrain_loss)

    def pretrain(self, session, data, n_epochs_vae, n_epochs_gmm):
        assert(self.pretrain_step is not None)

        for _ in range(n_epochs_vae):
            for batch in data.get_batches():
                feed = {
                    self.X: batch
                }
                feed.update(
                    self.sample_reparametrization_variables(
                        len(batch), variables=["Z"]
                    )
                )
                session.run(self.pretrain_step, feed_dict=feed)

        feed = {
            self.X: data.data
        }
        feed.update(
            self.sample_reparametrization_variables(
                len(data.data), variables=["Z"]
            )
        )
        Z = session.run(self.Z, feed_dict=feed)

        if n_epochs_gmm > 0:
            gmm_model = GaussianMixture(
                n_components=self.n_classes,
                covariance_type="diag",
                max_iter=n_epochs_gmm,
                n_init=5,
                weights_init=np.ones(self.n_classes) / self.n_classes,
            )
            gmm_model.fit(Z)

            lv = self.latent_variables["Z"][0]

            init_gmm_means = tf.assign(lv.means, gmm_model.means_)
            init_gmm_vars = tf.assign(
                lv.log_vars, np.log(gmm_model.covariances_ + 1e-10)
            )

            session.run([init_gmm_means, init_gmm_vars])

            labels = gmm_model.predict(Z)

            print(labels)

    def get_accuracy(self, session, data):
        logits = session.run(self.logits, feed_dict={self.X: data.data})

        clusters = np.argmax(logits, axis=-1)[:, None]
        classes = data.classes[:, None]

        np.save("comp.npy", np.concatenate([classes, clusters], axis=1))


class VaDE(VAE):
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
                tf.float32, shape=(None, self.latent_dim), name="epsilon"
            )

            self.latent_variables = dict()

            self.encoder_network = FeedForwardNetwork(
                name="z/encoder_network"
            )
            self.mean, self.log_var = self.encoder_network.build(
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
                        "cluster_sample": False
                    }
                )
            })

            lv, eps, params = self.latent_variables["Z"]
            self.Z = lv.inverse_reparametrize(eps, params)
            self.cluster_weights = lv.get_cluster_weights(self.Z)

            params["weights"] = self.cluster_weights

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

    def define_latent_loss(self):
        self.latent_loss = tf.add_n(
            [lv.kl_from_prior(params)
             for lv, _, params in self.latent_variables.values()]
        )
        self.latent_loss -= tf.reduce_mean(tf.reduce_sum(
            self.cluster_weights * tf.log(self.cluster_weights + 1e-10),
            axis=-1
        ))

    def define_pretrain_step(self, init_lr, decay_steps, decay_rate=0.9):
        assert(self.train_step is not None)

        learning_rate = tf.train.exponential_decay(
            learning_rate=init_lr,
            global_step=0,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )

        pretrain_loss = tf.reduce_mean(
            self.recon_loss + 0.5 * tf.reduce_sum(
                tf.exp(self.log_var) + self.mean ** 2 - self.log_var,
                axis=-1
            )
        )

        self.pretrain_step = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        ).minimize(pretrain_loss)

    def pretrain(self, session, data, n_epochs_vae, n_epochs_gmm):
        assert(self.pretrain_step is not None)

        for _ in range(n_epochs_vae):
            for batch in data.get_batches():
                feed = {
                    self.X: batch
                }
                feed.update(
                    self.sample_reparametrization_variables(
                        len(batch), variables=["Z"]
                    )
                )
                session.run(self.pretrain_step, feed_dict=feed)

        feed = {
            self.X: data.data
        }
        feed.update(
            self.sample_reparametrization_variables(
                len(data.data), variables=["Z"]
            )
        )
        Z = session.run(self.Z, feed_dict=feed)

        if n_epochs_gmm > 0:
            gmm_model = GaussianMixture(
                n_components=self.n_classes,
                covariance_type="diag",
                max_iter=n_epochs_gmm,
                n_init=5,
                weights_init=np.ones(self.n_classes) / self.n_classes,
            )
            gmm_model.fit(Z)

            lv = self.latent_variables["Z"][0]

            init_gmm_means = tf.assign(lv.means, gmm_model.means_)
            init_gmm_vars = tf.assign(
                lv.log_vars, np.log(gmm_model.covariances_ + 1e-10)
            )

            session.run([init_gmm_means, init_gmm_vars])

    def get_accuracy(self, session, data, k=10):
        weights = []
        for _ in range(k):
            feed = {self.X: data.data}
            feed.update(
                self.sample_reparametrization_variables(
                    len(data.data), variables=["Z"]
                )
            )
            weights.append(session.run(
                self.cluster_weights, feed_dict=feed
            ))

        weights = np.array(weights)
        weights = np.mean(weights, axis=0)

        clusters = np.argmax(weights, axis=-1)[:, None]
        classes = data.classes[:, None]

        np.save("comp.npy", np.concatenate([classes, clusters], axis=1))
