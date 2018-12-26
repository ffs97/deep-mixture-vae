import os
import priors
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from sklearn.mixture import GaussianMixture

from includes.network import FeedForwardNetwork, DeepNetwork
from includes.utils import get_clustering_accuracy
from includes.layers import Convolution, MaxPooling


class VAE:
    def __init__(self, name, input_type, input_dim, latent_dim, activation=None, initializer=None):
        self.name = name

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.input_type = input_type

        self.activation = activation
        self.initializer = initializer

        self.path = ""

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

    def debug(self, session, data):
        import pdb

        for batch in data.get_batches():
            feed = {
                self.X: batch
            }
            feed.update(
                self.sample_reparametrization_variables(len(batch))
            )

            pdb.set_trace()

            break


class DeepMixtureVAE(VAE):
    def __init__(self, name, input_type, input_dim, latent_dim, n_classes, activation=None, initializer=None):
        VAE.__init__(self, name, input_type, input_dim, latent_dim,
                     activation=activation, initializer=initializer)

        self.n_classes = n_classes

    def build_graph(self):
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
            self.dropout_c = tf.placeholder_with_default(
                0.0, shape=None, name="dropout_C"
            )
            self.dropout_z = tf.placeholder_with_default(
                0.0, shape=None, name="dropout_X"
            )

            self.lam = tf.placeholder_with_default(
                1.0, shape=None, name="lambda"
            )

            self.latent_variables = dict()

            with tf.variable_scope("encoder_network"):
                # Make this generic
                X_flat = tf.reshape(self.X, (-1, 28, 28, 1))

                # Make this configurable
                encoder_network = DeepNetwork(
                    "layers",
                    [
                        ("cn", {
                            "n_kernels": 32, "prev_n_kernels": 1, "kernel": (3, 3)
                        }),
                        ("cn", {
                            "n_kernels": 32, "prev_n_kernels": 32, "kernel": (3, 3)
                        }),
                        ("mp", {"k": 2}),
                        ("cn", {
                            "n_kernels": 64, "prev_n_kernels": 32, "kernel": (3, 3)
                        }),
                        ("cn", {
                            "n_kernels": 64, "prev_n_kernels": 64, "kernel": (3, 3)
                        }),
                        ("mp", {"k": 2}),
                        ("cn", {
                            "n_kernels": 128, "prev_n_kernels": 64, "kernel": (3, 3)
                        }),
                        ("cn", {
                            "n_kernels": 128, "prev_n_kernels": 128, "kernel": (3, 3)
                        }),
                        ("mp", {"k": 2}),
                        ("fc", {"input_dim": 2048, "output_dim": 128})
                    ],
                    # Following for fast testing
                    # [
                    #     ("cn", {
                    #         "n_kernels": 32, "prev_n_kernels": 1, "kernel": (3, 3)
                    #     }),
                    #     ("mp", {"k": 5}),
                    #     ("fc", {"input_dim": 1152, "output_dim": 128})
                    # ],
                    activation=self.activation,
                    initializer=self.initializer
                )
                hidden = encoder_network(X_flat)

                self.mean = tf.layers.dense(
                    hidden, self.latent_dim, activation=None, kernel_initializer=self.initializer()
                )
                self.log_var = tf.layers.dense(
                    hidden, self.latent_dim, activation=None, kernel_initializer=self.initializer()
                )

                self.logits = tf.layers.dense(
                    hidden, self.n_classes, activation=None, kernel_initializer=self.initializer()
                )
                self.cluster_weights = tf.nn.softmax(self.logits)

            self.latent_variables.update({
                "C": (
                    priors.DiscreteFactorial(
                        "cluster", 1, self.n_classes
                    ), self.cluster,
                    {"logits": self.logits}
                ),
                "Z": (
                    priors.NormalMixtureFactorial(
                        "representation", self.latent_dim, self.n_classes
                    ), self.epsilon,
                    {
                        "mean": self.mean,
                        "log_var": self.log_var,
                        "weights": self.cluster_weights,
                        "cluster_sample": False
                    }
                )
            })

            lv, eps, params = self.latent_variables["Z"]
            self.Z = lv.inverse_reparametrize(eps, params)

            with tf.variable_scope("decoder_network"):
                decoder_network = DeepNetwork(
                    "layers",
                    [
                        ("fc", {"input_dim": self.latent_dim, "output_dim": 128}),
                        ("fc", {"input_dim": 128, "output_dim": 256}),
                        ("fc", {"input_dim": 256, "output_dim": 512})
                    ],
                    activation=self.activation,
                    initializer=self.initializer
                )
                hidden = decoder_network(self.Z)

                self.decoded_X = tf.layers.dense(
                    hidden, self.input_dim, activation=None, kernel_initializer=self.initializer()
                )

            if self.input_type == "binary":
                self.reconstructed_X = tf.nn.sigmoid(self.decoded_X)
            elif self.input_type == "real":
                self.reconstructed_X = self.decoded_X
            else:
                raise NotImplementedError

        return self

    def define_train_loss(self):
        self.define_recon_loss()
        self.define_latent_loss()

        self.loss = tf.reduce_mean(
            self.lam * self.recon_loss + self.latent_loss
        )

    def get_accuracy(self, session, data):
        logits = []
        for batch in data.get_batches():
            logits.append(session.run(self.logits, feed_dict={self.X: batch}))

        logits = np.concatenate(logits, axis=0)

        return get_clustering_accuracy(logits, data.classes)


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
                name="z/encoder_network",
                activation=self.activation,
                initializer=self.initializer
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

            self.decoder_network = FeedForwardNetwork(
                name="decoder_network",
                activation=self.activation,
                initializer=self.initializer
            )
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
        self.latent_loss += tf.reduce_mean(tf.reduce_sum(
            self.cluster_weights * tf.log(self.cluster_weights + 1e-20),
            axis=-1
        ))
        # self.latent_loss *= 2.0

    def define_pretrain_step(self, init_lr, decay_steps, decay_rate=0.9, use_ae=True):
        learning_rate = tf.train.exponential_decay(
            learning_rate=init_lr,
            global_step=0,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )

        self.define_recon_loss()
        if use_ae:
            vae_pretrain_loss = tf.reduce_mean(
                self.recon_loss + 0.5 * tf.reduce_sum(
                    tf.exp(self.log_var) + self.mean ** 2 - self.log_var,
                    axis=-1
                )
            )
        else:
            vae_pretrain_loss = tf.reduce_mean(self.recon_loss)

        self.vae_pretrain_step = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        ).minimize(vae_pretrain_loss)

    def pretrain_vae(self, session, data, n_epochs_vae, use_ae=True):
        vae_var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + "/z/encoder_network"
        ) + tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + "/decoder_network"
        )

        vae_saver = tf.train.Saver(vae_var_list)

        vae_ckpt_path = self.path + "/vae/weights.ckpt"

        try:
            vae_saver.restore(session, vae_ckpt_path)
        except:
            print("Could not load pretrained vae model")

        min_loss = float("inf")
        with tqdm(range(n_epochs_vae)) as bar:
            for _ in bar:
                loss = 0
                for batch in data.get_batches():
                    if use_ae:
                        feed = {
                            self.X: batch,
                            self.epsilon: np.zeros(
                                (len(batch), self.latent_dim)
                            )
                        }
                    else:
                        feed = {
                            self.X: batch
                        }
                        feed.update(
                            self.sample_reparametrization_variables(
                                len(batch), variables=["Z"]
                            )
                        )

                    batch_loss, _ = session.run(
                        [self.recon_loss, self.vae_pretrain_step], feed_dict=feed
                    )
                    loss += batch_loss / data.epoch_len

                bar.set_postfix({"loss": "%.4f" % loss})

                if loss <= min_loss:
                    min_loss = loss
                    vae_saver.save(session, vae_ckpt_path)

    def pretrain_gmm(self, session, data, n_epochs_gmm):
        gmm_var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + "/representation"
        )
        gmm_saver = tf.train.Saver(gmm_var_list)
        gmm_ckpt_path = self.path + "/gmm/weights.ckpt"

        try:
            gmm_saver.restore(session, gmm_ckpt_path)
        except:
            print("Could not load pretrained gmm parameters")
            if n_epochs_gmm > 0:
                feed = {
                    self.X: data.data
                }
                Z = session.run(self.mean, feed_dict=feed)

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
                    lv.log_vars, np.log(gmm_model.covariances_ + 1e-20)
                )

                session.run([init_gmm_means, init_gmm_vars])
                gmm_saver.save(session, gmm_ckpt_path)

    def pretrain(self, session, data, n_epochs_vae, n_epochs_gmm, use_ae=True):
        assert(self.vae_pretrain_step is not None)

        self.pretrain_vae(session, data, n_epochs_vae, use_ae)
        self.pretrain_gmm(session, data, n_epochs_gmm)

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

        return get_clustering_accuracy(weights, data.classes)
