import os
import priors
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from sklearn.mixture import GaussianMixture

from includes.utils import get_clustering_accuracy
from includes.layers import Convolution, MaxPooling
from includes.network import DeepNetwork, get_encoder_networks


class VAE:
    def __init__(self, name, input_type, input_dim, input_shape,
                 latent_dim, activation=None, initializer=None):

        self.name = name

        self.input_dim = input_dim
        self.input_shape = input_shape

        self.latent_dim = latent_dim

        self.input_type = input_type

        self.activation = activation
        self.initializer = initializer

        self.path = ""

        self.is_training = tf.placeholder_with_default(
            True, shape=None, name="is_training"
        )
        self.kl_ratio = tf.placeholder_with_default(
            1.0, shape=None, name="kl_ratio"
        )

        self.X = None
        self.decoded_X = None
        self.train_step = None
        self.latent_variables = dict()

    def build_graph(self):
        raise NotImplementedError

    def sample_reparametrization_variables(self, n, variables=None):
        samples = dict()
        if variables is None:
            for lv, eps, _ in self.latent_variables.values():
                if eps is not None:
                    samples[eps] = lv.sample_reparametrization_variable(n)
        else:
            for var in variables:
                lv, eps, _ = self.latent_variables[var]
                if eps is not None:
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

        self.loss = tf.reduce_mean(
            self.recon_loss + self.kl_ratio * self.latent_loss
        )

    def define_train_step(self, init_lr, decay_steps, decay_rate=0.9):
        learning_rate = tf.train.exponential_decay(
            learning_rate=init_lr,
            global_step=0,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        )

        self.define_train_loss()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = optimizer.minimize(self.loss)

    def train_op(self, session, data, kl_ratio=1.0):
        assert(self.train_step is not None)

        loss = 0.0
        for batch in data.get_batches():
            feed = {
                self.X: batch,
                self.is_training: True,
                self.kl_ratio: kl_ratio
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
    def __init__(self, name, input_type, input_dim, input_shape, latent_dim,
                 n_classes, activation=None, initializer=None):

        VAE.__init__(self, name, input_type, input_dim, input_shape, latent_dim,
                     activation=activation, initializer=initializer)

        self.n_classes = n_classes

        self.shared_encoder = None

    def build_graph(self, cnn=False):
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

            self.latent_variables = dict()

            with tf.variable_scope("encoder_network"):
                self.encoder_z, self.encoder_c = get_encoder_networks(self)

                self.mean, self.log_var = self.encoder_z(self.X)

                self.logits = self.encoder_c(self.X)
                self.cluster_probs = tf.nn.softmax(self.logits)

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
                        "weights": self.cluster_probs,
                        "cluster_sample": False
                    }
                )
            })

            lv, eps, params = self.latent_variables["Z"]
            self.Z = lv.inverse_reparametrize(eps, params)

            with tf.variable_scope("decoder_network"):
                self.decoder = DeepNetwork(
                    "layers",
                    [
                        ("fc", {"input_dim": self.latent_dim, "output_dim": 2000}),
                        ("fc", {"input_dim": 2000, "output_dim": 500}),
                        ("fc", {"input_dim": 500, "output_dim": 500})
                    ],
                    [
                        ("fc", {"input_dim": 500, "output_dim": self.input_dim})
                    ],
                    activation=self.activation, initializer=self.initializer
                )
                self.decoded_X = self.decoder(self.Z)

            if self.input_type == "binary":
                self.reconstructed_X = tf.nn.sigmoid(self.decoded_X)
            elif self.input_type == "real":
                self.reconstructed_X = self.decoded_X
            else:
                raise NotImplementedError

        return self

    def get_accuracy(self, session, data):
        logits = []
        for batch in data.get_batches():
            logits.append(session.run(self.logits, feed_dict={self.X: batch}))

        logits = np.concatenate(logits, axis=0)

        return get_clustering_accuracy(logits, data.classes)


class VaDE(VAE):
    def __init__(self, name, input_type, input_dim, input_shape, latent_dim,
                 n_classes, activation=None, initializer=None):
        VAE.__init__(self, name, input_type, input_dim, input_shape, latent_dim,
                     activation=activation, initializer=initializer)

        self.n_classes = n_classes

    def build_graph(self, cnn=False):
        with tf.variable_scope(self.name) as _:
            self.X = tf.placeholder(
                tf.float32, shape=(None, self.input_dim), name="X"
            )
            self.epsilon = tf.placeholder(
                tf.float32, shape=(None, self.latent_dim), name="epsilon"
            )

            self.latent_variables = dict()

            if cnn:
                self.X = tf.reshape(self.X, self.input_shape)

            with tf.variable_scope("encoder_network"):
                encoder_network = self.define_encoder_network(cnn)
                hidden = encoder_network(self.X)
                hidden = tf.nn.relu(hidden)

                with tf.variable_scope("z"):
                    encoder_network_z = DeepNetwork(
                        "layers",
                        [
                            ("fc", {"input_dim": 500, "output_dim": 2000})
                        ],
                        [
                            ("fc", {
                                "input_dim": 2000, "output_dim": self.latent_dim
                            }),
                            ("fc", {
                                "input_dim": 2000, "output_dim": self.latent_dim
                            })
                        ],
                        activation=self.activation, initializer=self.initializer
                    )
                    self.mean, self.log_var = encoder_network_z(hidden)

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

            self.cluster_probs = lv.get_cluster_probs(self.Z)
            params["weights"] = self.cluster_probs

            self.latent_variables.update({
                "C": (
                    priors.DiscreteFactorial(
                        "cluster", 1, self.n_classes
                    ), None,
                    {"probs": self.cluster_probs}
                )
            })

            with tf.variable_scope("decoder_network"):
                decoder_network = DeepNetwork(
                    "layers",
                    [
                        ("fc", {"input_dim": self.latent_dim, "output_dim": 2000}),
                        ("fc", {"input_dim": 2000, "output_dim": 500})
                    ],
                    [
                        ("fc", {"input_dim": 500, "output_dim": self.input_dim})
                    ],
                    activation=self.activation, initializer=self.initializer
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

    def define_latent_loss(self):
        self.latent_loss = tf.add_n(
            [lv.kl_from_prior(params)
             for lv, _, params in self.latent_variables.values()]
        )
        self.latent_loss += tf.reduce_mean(tf.reduce_sum(
            self.cluster_probs * tf.log(self.cluster_probs + 1e-20),
            axis=-1
        ))

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
                self.cluster_probs, feed_dict=feed
            ))

        weights = np.array(weights)
        weights = np.mean(weights, axis=0)

        return get_clustering_accuracy(weights, data.classes)


class IMSAT:
    def __init__(self, name, input_type, input_dim, input_shape, n_classes, mu=4,
                 lam=0.2, xi=10, Ip=1, eps=1, activation=None, initializer=None):

        self.name = name

        self.input_dim = input_dim
        self.input_shape = input_shape

        self.input_type = input_type

        self.activation = activation
        self.initializer = initializer

        self.n_classes = n_classes

        self.path = ""

        self.is_training = tf.placeholder_with_default(
            True, shape=None, name="is_training"
        )

        self.mu = mu
        self.lam = lam

        self.xi = xi
        self.Ip = Ip
        self.eps = eps

        self.is_training = tf.placeholder_with_default(
            False, shape=None, name="is_training"
        )

    def build_graph(self):
        with tf.variable_scope(self.name) as _:
            self.X = tf.placeholder(
                tf.float32, shape=(None, self.input_dim), name="X"
            )

            with tf.variable_scope("network"):
                network = DeepNetwork(
                    "layers",
                    [
                        ("fc", {
                            "input_dim": self.input_dim,
                            "output_dim": 1200
                        }),
                        ("bn", {
                            "input_dim": 1200, "is_training": self.is_training
                        }),
                        ("fc", {
                            "input_dim": 1200, "output_dim": 1200
                        }),
                        ("bn", {
                            "input_dim": 1200, "is_training": self.is_training
                        })
                    ],
                    [
                        ("fc", {"input_dim": 1200, "output_dim": self.n_classes})
                    ],
                    activation=self.activation, initializer=self.initializer
                )
                self.logits = network(self.X)

                self.cluster_probs = tf.nn.softmax(self.logits)

            with tf.variable_scope("adversary"):
                ul_logits = network(self.X)

                d = tf.random_normal(shape=tf.shape(self.X))
                d /= (tf.reshape(tf.sqrt(tf.reduce_sum(tf.pow(d, 2.0),
                                                       axis=1)), [-1, 1]) + 1e-16)
                for _ in range(self.Ip):
                    y1 = ul_logits
                    y2 = network(self.X + self.xi * d)
                    kl_loss = tf.reduce_mean(self.compute_kld(y1, y2))
                    grad = tf.gradients(kl_loss, [d])[0]
                    d = tf.stop_gradient(grad)
                    d /= (tf.reshape(tf.sqrt(tf.reduce_sum(tf.pow(d, 2.0),
                                                           axis=1)), [-1, 1]) + 1e-16)

                self.orig_example = tf.stop_gradient(ul_logits)
                self.adversary = network(self.X + d * self.eps)

        return self

    def define_train_loss(self):
        mean_probs = tf.reduce_mean(self.cluster_probs, axis=0)
        H_Y = -tf.reduce_sum(mean_probs * tf.log(mean_probs + 1e-16))
        H_Y_X = tf.reduce_mean(self.entropy(self.cluster_probs))
        self.entropy_loss = H_Y_X - self.mu * H_Y

        self.adversary_loss = tf.reduce_mean(
            self.compute_kld(self.orig_example, self.adversary))

        self.loss = self.adversary_loss + self.lam * self.entropy_loss

    def compute_kld(self, p_logit, q_logit):
        p = tf.nn.softmax(p_logit)
        q = tf.nn.softmax(q_logit)
        return tf.reduce_sum(p * (tf.log(p + 1e-16) - tf.log(q + 1e-16)), axis=1)

    def entropy(self, p):
        return -tf.reduce_sum(p * tf.log(p + 1e-16), axis=1)

    def define_train_step(self, init_lr, decay_steps, decay_rate=0.9):
        learning_rate = tf.train.exponential_decay(
            learning_rate=init_lr,
            global_step=0,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )

        self.define_train_loss()

        # for batchnorm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_step = optimizer.minimize(self.loss)

    def train_op(self, session, data):
        assert self.train_step is not None
        loss = 0.0
        for batch in data.get_batches():
            feed = {
                self.X: batch,
                self.is_training: True
            }
            batch_loss, _ = session.run(
                [self.loss, self.train_step],
                feed_dict=feed
            )
            loss += batch_loss / data.epoch_len
        return loss

    def get_accuracy(self, session, data):
        logits = []
        for batch in data.get_batches():
            logits.append(session.run(self.logits, feed_dict={self.X: batch}))

        logits = np.concatenate(logits, axis=0)

        return get_clustering_accuracy(logits, data.classes)


class AdversarialDMVAE(DeepMixtureVAE):
    def __init__(self, name, input_type, input_dim, input_shape, latent_dim, n_classes,
                 lam=1.0, activation=None, initializer=None):
        DeepMixtureVAE.__init__(self, name, input_type, input_dim, input_shape, latent_dim,
                                n_classes, activation=activation, initializer=initializer)

        self.lam = lam

    def entropy(self, p):
        return -tf.reduce_sum(p * tf.log(p + 1e-16), axis=1)

    def compute_kld(self, p_logit, q_logit):
        p = tf.nn.softmax(p_logit)
        q = tf.nn.softmax(q_logit)
        return tf.reduce_sum(p * (tf.log(p + 1e-16) - tf.log(q + 1e-16)), axis=1)

    def define_adversarial_loss(self):
        def compute_logits(inputs):
            outputs = self.encoder_c(inputs)

            return outputs

        ul_logits = compute_logits(self.X)

        d = tf.random_normal(shape=tf.shape(self.X))
        d /= (tf.reshape(tf.sqrt(tf.reduce_sum(tf.pow(d, 2.0),
                                               axis=1)), [-1, 1]) + 1e-16)
        for _ in range(1):
            y1 = ul_logits
            y2 = compute_logits(self.X + 10 * d)
            kl_loss = tf.reduce_mean(self.compute_kld(y1, y2))
            grad = tf.gradients(kl_loss, [d])[0]
            d = tf.stop_gradient(grad)
            d /= (tf.reshape(tf.sqrt(tf.reduce_sum(tf.pow(d, 2.0),
                                                   axis=1)), [-1, 1]) + 1e-16)

        orig_example = tf.stop_gradient(ul_logits)
        adversary = compute_logits(self.X + d * 1)

        self.adversarial_loss = tf.reduce_mean(
            self.compute_kld(orig_example, adversary)
        )

    def define_train_loss(self):
        self.define_latent_loss()
        self.define_recon_loss()
        self.define_adversarial_loss()

        self.loss = tf.reduce_mean(
            self.lam * self.adversarial_loss +
            self.recon_loss + self.kl_ratio * self.latent_loss
        )
