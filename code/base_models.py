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
    def __init__(self, name, input_type, input_dim, latent_dim, activation=None, initializer=None, ss=False):
        self.name = name

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.ss = ss

        self.input_type = input_type

        self.activation = activation
        self.initializer = initializer

        self.path = ""

        self.kl_ratio = tf.placeholder_with_default(
            1.0, shape=None, name="kl_ratio"
        )

        self.is_training = tf.placeholder_with_default(
            True, shape=None, name="is_training"
        )

        self.X = None
        self.decoded_X = None
        self.train_step = None
        self.latent_variables = dict()
        if ss:
            self.latent_variables_unl = dict()
            self.decoded_X_unl = None
            self.X_unl = None


    def build_graph(self, encoder_layer_sizes, decoder_layer_sizes):
        raise NotImplementedError

    def sample_reparametrization_variables(self, n, ss=False, variables=None):
        samples = dict()
        
        if ss:
            if variables is None:
                for lv, eps, _ in self.latent_variables_unl.values():
                    if eps is not None:
                        samples[eps] = lv.sample_reparametrization_variable(n)
            else:
                for var in variables:
                    lv, eps, _ = self.latent_variables_unl[var]
                    if eps is not None:
                        samples[eps] = lv.sample_reparametrization_variable(n)

        else:
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
        self.latent_loss_lbl = tf.add_n(
            [lv.kl_from_prior(params)
             for lv, _, params in self.latent_variables.values()]
        )
        if self.ss:
          self.latent_loss_unl = tf.add_n(
            [lv.kl_from_prior(params)
             for lv, _, params in self.latent_variables_unl.values()]
          )

          self.latent_loss = self.latent_loss_lbl + self.latent_loss_unl
        else:
          self.latent_loss = self.latent_loss_lbl

    def define_recon_loss(self):
        
        if self.input_type == "binary":
            self.recon_loss_lbl = tf.reduce_mean(tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.X,
                    logits=self.decoded_X
                ), axis=1
            ))
            if self.ss:
                self.recon_loss_unl = tf.reduce_mean(tf.reduce_sum(
                   tf.nn.sigmoid_cross_entropy_with_logits(
                      labels=self.X_unl,
                      logits=self.decoded_X_unl
                   ), axis=1
                ))

        elif self.input_type == "real":
            self.recon_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(
                tf.square(self.X - self.decoded_X), axis=1
            ))
            if self.ss:
                self.recon_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(
                    tf.square(self.X_unl - self.decoded_X_unl), axis=1
                ))
        else:
            raise NotImplementedError
        if self.ss:
           self.recon_loss = self.recon_loss_lbl + self.recon_loss_unl
        else:
           self.recon_loss = self.recon_loss_lbl
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


def encoder_network(input, activation, initializer, reuse=None, cnn=True):

    with tf.variable_scope("encoder_network", reuse=tf.AUTO_REUSE):
        if cnn:
            X_flat = tf.reshape(input, (-1, 28, 28, 1))
            conv1 = tf.layers.conv2d(X_flat, filters=32, kernel_size=[5, 5], activation=tf.nn.relu, kernel_initializer=initializer)
            pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)   
            conv2 = tf.layers.conv2d(pool1, filters=32, kernel_size=[5, 5], activation=tf.nn.relu, kernel_initializer=initializer)
            pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)
            pool2_flat = tf.layers.flatten(pool2)
            hidden = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu, kernel_initializer=initializer)

        else:

            hidden = tf.layers.dense(input, 500, activation=activation, kernel_initializer=initializer)
            hidden = tf.layers.dense(hidden, 500, activation=activation, kernel_initializer=initializer)

    return hidden

def Z_network(input, activation, initializer, latent_dim, reuse=None, cnn=True):
    with tf.variable_scope("z", reuse=tf.AUTO_REUSE):
        hidden_z = tf.layers.dense(input, 128, activation=activation, kernel_initializer=initializer)
        mean = tf.layers.dense(hidden_z, latent_dim, activation=None, kernel_initializer=initializer)
        log_var = tf.layers.dense(hidden_z, latent_dim, activation=None, kernel_initializer=initializer)

    return mean, log_var

def C_network(input, activation, initializer, n_classes, reuse=None, cnn=True):
    with tf.variable_scope("c", reuse=tf.AUTO_REUSE):
        hidden_c = tf.layers.dense(input, 128, activation=activation, kernel_initializer=initializer)
        logits = tf.layers.dense(hidden_c, n_classes, activation=None, kernel_initializer=initializer)
        cluster_probs = tf.nn.softmax(logits)

    return logits, cluster_probs

def decoder_network(input, activation, initializer, input_dim, reuse=None, cnn=True):
    with tf.variable_scope("decoder_network", reuse=tf.AUTO_REUSE):
        hidden = tf.layers.dense(input, 256, activation=activation, kernel_initializer=initializer)
        hidden = tf.layers.dense(hidden, 256, activation=activation, kernel_initializer=initializer)
        hidden = tf.layers.dense(input, 512, activation=activation, kernel_initializer=initializer)
        decoded_X = tf.layers.dense(hidden, input_dim, activation=None, kernel_initializer=initializer)

    return decoded_X

class DeepMixtureVAE(VAE):
    def __init__(self, name, input_type, input_dim, latent_dim, n_classes, activation=None, initializer=None, cnn=False, noVAE=False, ss=False):
        VAE.__init__(self, name, input_type, input_dim, latent_dim,
                     activation=activation, initializer=initializer, ss=ss)

        self.n_classes = n_classes
        self.cnn = True#cnn
        self.prog = tf.placeholder_with_default(1.0, shape=()) 
        self.noVAE = noVAE
        self.ss = ss

    def build_graph(self):
        with tf.variable_scope(self.name) as _:
            self.X = tf.placeholder(
                tf.float32, shape=(None, self.input_dim), name="X"
            )

            if self.noVAE == False:
                self.epsilon = tf.placeholder(
                    tf.float32, shape=(None, self.latent_dim), name="epsilon_Z"
                )
                self.cluster = tf.placeholder(
                    tf.float32, shape=(None, 1, self.n_classes), name="epsilon_C"
                )

            if self.ss:
                self.X_unl = tf.placeholder(
                    tf.float32, shape=(None, self.input_dim), name="X_unl"
                )
                self.latent_variables_unl = dict()
                self.epsilon_unl = tf.placeholder(
                    tf.float32, shape=(None, self.latent_dim), name="epsilon_Z_unl"
                )
                self.cluster_unl = tf.placeholder(
                    tf.float32, shape=(None, 1, self.n_classes), name="epsilon_C_unl"
                )

            self.prob = tf.placeholder_with_default(1.0, shape=())
            self.latent_variables = dict()     

            self.hidden = encoder_network(self.X, self.activation, self.initializer(), reuse=None, cnn=self.cnn)

            if self.noVAE == False:
                
                self.mean, self.log_var = Z_network(self.hidden, self.activation, self.initializer(), self.latent_dim, reuse=None, cnn=self.cnn)
                self.logits, self.cluster_probs = C_network(self.hidden, self.activation, self.initializer(), self.n_classes, reuse=None, cnn=self.cnn)

                dropout = tf.layers.dropout(self.hidden, rate=self.prog)
                self.reconstructed_Y_soft = tf.nn.softmax(tf.layers.dense(inputs=dropout, units=self.n_classes))


            if self.noVAE == False:
                priorFac = priors.DiscreteFactorial(
                            "cluster", 1, self.n_classes
                        )
                priorNormal = priors.NormalMixtureFactorial(
                            "representation", self.latent_dim, self.n_classes
                        )
                self.latent_variables.update({
                    "C": (priorFac, self.cluster,{"logits": self.logits}),
                    "Z": (priorNormal, self.epsilon,
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

                self.decoded_X = decoder_network(self.Z, self.activation, self.initializer(), self.input_dim, reuse=None, cnn=self.cnn)

            if self.ss:
                self.hidden_unl = encoder_network(self.X_unl, self.activation, self.initializer(), cnn=self.cnn)
                self.mean_unl, self.log_var_unl = Z_network(self.hidden_unl, self.activation, self.initializer(), self.latent_dim, reuse=None, cnn=self.cnn)
                self.logits_unl, self.cluster_probs_unl = C_network(self.hidden_unl, self.activation, self.initializer(), self.n_classes, reuse=None, cnn=self.cnn)

                self.latent_variables_unl.update({
                    "C": (priorFac, self.cluster,{"logits": self.logits_unl}),
                    "Z": (priorNormal, self.epsilon_unl,
                        {
                            "mean": self.mean_unl,
                            "log_var": self.log_var_unl,
                            "weights": self.cluster_probs_unl,
                            "cluster_sample": False
                        }
                    )
                })

                lv, eps, params = self.latent_variables_unl["Z"]
                self.Z_unl = lv.inverse_reparametrize(eps, params)
                self.decoded_X_unl = decoder_network(self.Z_unl, self.activation, self.initializer(), self.input_dim, reuse=None, cnn=self.cnn)

        return self

    def define_pretrain_step(self, vae_lr, prior_lr):
        self.define_train_loss()

        self.vae_loss = self.recon_loss
        self.vae_train_step = tf.train.AdamOptimizer(
            learning_rate=vae_lr
        ).minimize(self.recon_loss)

        prior_var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + "/encoder_network/c"
        )
        # + tf.get_collection(
        #     tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + "/representation"
        # )

        self.prior_train_step = tf.train.AdamOptimizer(
            learning_rate=prior_lr
        ).minimize(self.latent_loss, var_list=prior_var_list)

    def pretrain_vae(self, session, data, n_epochs):
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        saver = tf.train.Saver(var_list)
        ckpt_path = self.path + "/vae/parameters.ckpt"

        try:
            saver.restore(session, ckpt_path)
        except:
            print("Could not load trained ae parameters")

        min_loss = float("inf")
        with tqdm(range(n_epochs)) as bar:
            for _ in bar:
                loss = 0
                for batch in data.get_batches():
                    feed = {
                        self.X: batch,
                        self.epsilon: np.zeros(
                            (len(batch), self.latent_dim)
                        ),
                        self.is_training: True
                    }

                    batch_loss, _ = session.run(
                        [self.recon_loss, self.vae_train_step], feed_dict=feed
                    )
                    loss += batch_loss / data.epoch_len
                bar.set_postfix({"loss": "%.4f" % loss})

                if loss <= min_loss:
                    min_loss = loss
                    saver.save(session, ckpt_path)

    def pretrain_prior(self, session, data, n_epochs):
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        saver = tf.train.Saver(var_list)
        ckpt_path = self.path + "/prior/parameters.ckpt"

        try:
            saver.restore(session, ckpt_path)
        except:
            print("Could not load trained prior parameters")

            if n_epochs > 0:
                feed = {
                    self.X: data.data
                }
                Z = session.run(self.mean, feed_dict=feed)

                gmm_model = GaussianMixture(
                    n_components=self.n_classes,
                    covariance_type="diag",
                    max_iter=n_epochs,
                    n_init=20,
                    weights_init=np.ones(self.n_classes) / self.n_classes,
                )
                gmm_model.fit(Z)

                lv = self.latent_variables["Z"][0]

                init_prior_means = tf.assign(lv.means, gmm_model.means_)
                init_prior_vars = tf.assign(
                    lv.log_vars, np.log(gmm_model.covariances_ + 1e-20)
                )

                session.run([init_prior_means, init_prior_vars])
                saver.save(session, ckpt_path)

        min_loss = float("inf")
        with tqdm(range(n_epochs)) as bar:
            for _ in bar:
                loss = 0
                for batch in data.get_batches():
                    feed = {
                        self.X: batch,
                        self.epsilon: np.zeros(
                            (len(batch), self.latent_dim)
                        ),
                        self.is_training: True
                    }

                    batch_loss, _ = session.run(
                        [self.latent_loss, self.prior_train_step], feed_dict=feed
                    )
                    loss += batch_loss / data.epoch_len

                bar.set_postfix({"loss": "%.4f" % loss})

                if loss <= min_loss:
                    min_loss = loss
                    saver.save(session, ckpt_path)

    def pretrain(self, session, data, n_epochs_vae, n_epochs_gmm):
        assert(
            self.vae_train_step is not None and
            self.prior_train_step is not None
        )

        self.pretrain_vae(session, data, n_epochs_vae)
        self.pretrain_prior(session, data, n_epochs_gmm)

    def get_accuracy(self, session, data):
        logits = []
        for batch in data.get_batches():
            logits.append(session.run(self.logits, feed_dict={self.X: batch}))

        logits = np.concatenate(logits, axis=0)

        return get_clustering_accuracy(logits, data.classes)


class VaDE(VAE):
    def __init__(self, name, input_type, input_dim, latent_dim, n_classes, activation=None, initializer=None, cnn=False):
        VAE.__init__(self, name, input_type, input_dim, latent_dim,
                     activation=activation, initializer=initializer)

        self.n_classes = n_classes
        self.cnn = cnn

    def build_graph(self):
        with tf.variable_scope(self.name) as _:
            self.X = tf.placeholder(
                tf.float32, shape=(None, self.input_dim), name="X"
            )
            self.epsilon = tf.placeholder(
                tf.float32, shape=(None, self.latent_dim), name="epsilon"
            )

            self.latent_variables = dict()

            X_flat = tf.reshape(self.X, (-1, 28, 28, 1))
            with tf.variable_scope("encoder_network"):

                if self.cnn:
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
                        activation=self.activation,
                        initializer=self.initializer
                    )
                    hidden = encoder_network(X_flat)
                else:
                    
                    encoder_network = DeepNetwork(
                    "layers",
                    [
                        ("fc", {"input_dim": self.input_dim, "output_dim": 2000}),
                        ("fc", {"input_dim": 2000, "output_dim": 500}),
                        ("fc", {"input_dim": 500, "output_dim": 500})
                    ],
                    activation=self.activation, initializer=self.initializer
                    )
                    hidden = encoder_network(self.X)


                with tf.variable_scope("z"):
                    self.mean = tf.layers.dense(
                        hidden, self.latent_dim, activation=None, kernel_initializer=self.initializer()
                    )
                    self.log_var = tf.layers.dense(
                        hidden, self.latent_dim, activation=None, kernel_initializer=self.initializer()
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
                        ("fc", {"input_dim": self.latent_dim, "output_dim": 500}),
                        ("fc", {"input_dim": 500, "output_dim": 500}),
                        ("fc", {"input_dim": 500, "output_dim": 2000})
                    ],
                    activation=self.activation, initializer=self.initializer
                )
                hidden = decoder_network(self.Z)

                self.decoded_X = tf.layers.dense(
                    hidden, self.input_dim, activation=None, kernel_initializer=self.initializer()
                )


            
           
          
         
        
       

        return self

    # def define_latent_loss(self):
    #     self.latent_loss = tf.add_n(
    #         [lv.kl_from_prior(params)
    #          for lv, _, params in self.latent_variables.values()]
    #     )
    #     self.latent_loss += tf.reduce_mean(tf.reduce_sum(
    #         self.cluster_probs * tf.log(self.cluster_probs + 1e-20),
    #         axis=-1
    #     ))

    def define_pretrain_step(self, vae_lr, _prior_lr=None):
        self.define_train_loss()

        self.vae_loss = self.recon_loss
        self.vae_train_step = tf.train.AdamOptimizer(
            learning_rate=vae_lr
        ).minimize(self.recon_loss)

    def pretrain_vae(self, session, data, n_epochs):
        saver = tf.train.Saver()
        ckpt_path = self.path + "/vae/parameters.ckpt"

        try:
            saver.restore(session, ckpt_path)
        except:
            print("Could not load pretrained vae model")

        min_loss = float("inf")
        with tqdm(range(n_epochs)) as bar:
            for _ in bar:
                loss = 0
                for batch in data.get_batches():
                    feed = {
                        self.X: batch,
                        self.epsilon: np.zeros(
                            (len(batch), self.latent_dim)
                        )
                    }

                    batch_loss, _ = session.run(
                        [self.recon_loss, self.vae_train_step], feed_dict=feed
                    )
                    loss += batch_loss / data.epoch_len

                bar.set_postfix({"loss": "%.4f" % loss})

                if loss <= min_loss:
                    min_loss = loss
                    saver.save(session, ckpt_path)

    def pretrain_prior(self, session, data, n_epochs):
        saver = tf.train.Saver()
        ckpt_path = self.path + "/prior/parameters.ckpt"

        try:
            saver.restore(session, ckpt_path)
        except:
            print("Could not load pretrained prior parameters")

            if n_epochs > 0:
                feed = {
                    self.X: data.data
                }
                Z = session.run(self.mean, feed_dict=feed)

                gmm_model = GaussianMixture(
                    n_components=self.n_classes,
                    covariance_type="diag",
                    max_iter=n_epochs,
                    n_init=5,
                    weights_init=np.ones(self.n_classes) / self.n_classes,
                )
                gmm_model.fit(Z)

                lv = self.latent_variables["Z"][0]

                init_prior_means = tf.assign(lv.means, gmm_model.means_)
                init_prior_vars = tf.assign(
                    lv.log_vars, np.log(gmm_model.covariances_ + 1e-20)
                )

                session.run([init_prior_means, init_prior_vars])
                saver.save(session, ckpt_path)

    def pretrain(self, session, data, n_epochs_vae, n_epochs_prior):
        assert(self.vae_train_step is not None)

        self.pretrain_vae(session, data, n_epochs_vae)
        self.pretrain_prior(session, data, n_epochs_prior)

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
