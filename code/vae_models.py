import os
import priors
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.utils.linear_assignment_ import linear_assignment

from includes.visualization import mnist_sample_plot as sample_plot
from includes.visualization import mnist_regeneration_plot as regeneration_plot

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

            self.cluster_weights = tf.nn.softmax(self.logits)

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
                        # "weights": self.C,
                        # "cluster_sample": True
                        "weights": self.cluster_weights,
                        "cluster_sample": False
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

    def define_pretrain_step(self, init_lr, decay_steps, decay_rate=0.9, use_ae=True):
        self.define_train_loss()

        # VAE Pretraining
        vae_learning_rate = tf.train.exponential_decay(
            learning_rate=init_lr,
            global_step=0,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )

        if not use_ae:
            latent_loss = tf.reduce_mean(0.5 * tf.reduce_sum(
                tf.exp(self.log_var) + tf.square(self.mean) - 1 - self.log_var,
                axis=1
            ))
            vae_pretrain_loss = tf.reduce_mean(self.recon_loss + latent_loss)
        else:
            vae_pretrain_loss = tf.reduce_mean(self.recon_loss)

        self.vae_pretrain_step = tf.train.AdamOptimizer(
            learning_rate=vae_learning_rate
        ).minimize(vae_pretrain_loss)

        # GMM Pretraining
        gmm_learning_rate = tf.train.exponential_decay(
            learning_rate=init_lr,
            global_step=0,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )

        self.cluster_labels = tf.placeholder(
            tf.int32, shape=(None, self.n_classes), name="cluster_labels"
        )
        gmm_pretrain_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.cluster_labels,
            logits=self.logits
        )
        gmm_pretrain_loss = tf.reduce_mean(gmm_pretrain_loss)
        self.gmm_pretrain_step = tf.train.AdamOptimizer(
            learning_rate=gmm_learning_rate
        ).minimize(gmm_pretrain_loss)

        # VAE-GMM Consistency Training
        vae_learning_rate = tf.train.exponential_decay(
            learning_rate=init_lr,
            global_step=0,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )

        gmm_fixed_var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + "/z/encoder_network"
        ) + tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + "/decoder_network"
        )
        vae_train_loss = tf.reduce_mean(self.recon_loss + 2 * self.latent_loss)
        self.vae_train_step = tf.train.AdamOptimizer(
            learning_rate=vae_learning_rate
        ).minimize(vae_train_loss, var_list=gmm_fixed_var_list)

    def pretrain_vae(self, session, data, n_epochs_vae, use_ae=True):
        vae_var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + "/z/encoder_network"
        ) + tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + "/decoder_network"
        )

        vae_saver = tf.train.Saver(vae_var_list)

        vae_ckpt_path = "saved_models/%s/vae/weights.ckpt" % self.name

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
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + "/c/encoder_network"
        ) + tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + "/representation"
        )
        gmm_saver = tf.train.Saver(gmm_var_list)
        gmm_ckpt_path = "saved_models/%s/gmm/weights.ckpt" % self.name

        try:
            gmm_saver.restore(session, gmm_ckpt_path)
        except:
            print("Could not load pretrained gmm parameters")
            if n_epochs_gmm > 0:
                size = len(data.data)

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
                labels = gmm_model.fit_predict(Z)

                lv = self.latent_variables["Z"][0]

                init_gmm_means = tf.assign(lv.means, gmm_model.means_)
                init_gmm_vars = tf.assign(
                    lv.log_vars, np.log(gmm_model.covariances_ + 1e-10)
                )

                session.run([init_gmm_means, init_gmm_vars])
                gmm_saver.save(session, gmm_ckpt_path)

                labels_one_hot = np.zeros((size, self.n_classes))
                labels_one_hot[np.arange(size), labels] = 1

                max_accuracy = 0
                with tqdm(range(n_epochs_gmm)) as bar:
                    for _ in bar:
                        feed = {
                            self.X: data.data,
                            self.cluster_labels: labels_one_hot
                        }
                        weights, _ = session.run([
                            self.cluster_weights, self.gmm_pretrain_step
                        ], feed_dict=feed)

                        accuracy = np.sum(
                            np.argmax(weights, axis=1) == labels) / len(labels) * 100

                        bar.set_postfix({"acc": "%.2f%%" % accuracy})

                        if accuracy >= max_accuracy:
                            max_accuracy = accuracy
                            gmm_saver.save(session, gmm_ckpt_path)

    def pretrain_dmvae(self, session, data, n_epochs_dmvae):
        self.get_accuracy(session, data)

        dmvae_var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + "/z/encoder_network"
        ) + tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + "/decoder_network"
        ) + tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + "/c/encoder_network"
        ) + tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + "/representation"
        )

        dmvae_saver = tf.train.Saver(dmvae_var_list)

        dmvae_ckpt_path = "saved_models/%s/dmvae/weights.ckpt" % self.name

        try:
            dmvae_saver.restore(session, dmvae_ckpt_path)
        except:
            print("Could not load pretrained dmvae model")

        min_loss = float("inf")
        with tqdm(range(n_epochs_dmvae)) as bar:
            for _ in bar:
                loss = 0
                for batch in data.get_batches():
                    feed = {
                        self.X: batch
                    }
                    feed.update(
                        self.sample_reparametrization_variables(
                            len(batch), variables=["Z"]
                        )
                    )

                    batch_loss, _ = session.run(
                        [self.loss, self.vae_train_step], feed_dict=feed
                    )
                    loss += batch_loss / data.epoch_len

                bar.set_postfix({"loss": "%.4f" % loss})

                if loss <= min_loss:
                    min_loss = loss
                    dmvae_saver.save(session, dmvae_ckpt_path)

    def pretrain(self, session, data, n_epochs_vae, n_epochs_gmm, n_epochs_dmvae, use_ae=True):
        assert(
            self.vae_pretrain_step is not None and
            self.gmm_pretrain_step is not None
        )

        self.pretrain_vae(session, data, n_epochs_vae, use_ae)
        # self.pretrain_gmm(session, data, n_epochs_gmm)
        # self.pretrain_dmvae(session, data, n_epochs_dmvae)

    def get_accuracy(self, session, data):
        logits = session.run(self.logits, feed_dict={self.X: data.data})

        clusters = np.argmax(logits, axis=-1)[:, None]
        classes = data.classes[:, None]

        np.save("comp.npy", np.concatenate([classes, clusters], axis=1))

        size = len(clusters)
        d = np.zeros((10, 10), dtype=np.int32)

        for i in range(size):
            d[clusters[i], classes[i]] += 1

        ind = linear_assignment(d.max() - d)
        return sum([d[i, j] for i, j in ind]) / size * 100


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
        self.latent_loss += tf.reduce_mean(tf.reduce_sum(
            self.cluster_weights * tf.log(self.cluster_weights + 1e-10),
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

        vae_ckpt_path = "saved_models/%s/vae/weights.ckpt" % self.name

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
        gmm_ckpt_path = "saved_models/%s/gmm/weights.ckpt" % self.name

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
                    lv.log_vars, np.log(gmm_model.covariances_ + 1e-10)
                )

                session.run([init_gmm_means, init_gmm_vars])
                gmm_saver.save(session, gmm_ckpt_path)

    def pretrain(self, session, data, n_epochs_vae, n_epochs_gmm, use_ae=True):
        assert(self.vae_pretrain_step is not None)

        if not os.path.exists("saved_models/" + self.name):
            os.makedirs("saved_models/" + self.name)

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

        clusters = np.argmax(weights, axis=-1)[:, None]
        classes = data.classes[:, None]

        np.save("comp.npy", np.concatenate([classes, clusters], axis=1))
