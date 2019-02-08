import os
import priors
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from sklearn.mixture import GaussianMixture

from includes.utils import get_clustering_accuracy
from vae import *
from includes.network import *
class clusterVAE(VAE):
    def __init__(self, name, input_type, input_dim, latent_dim, n_classes, activation=None, initializer=None, cnn=True, ss=False, noVAE=False):
        VAE.__init__(self, name, input_type, input_dim, latent_dim,
                     activation=activation, initializer=initializer, ss=ss)

        self.n_classes = n_classes
        self.cnn = cnn
        self.prog = tf.placeholder_with_default(1.0, shape=()) 
        self.noVAE = noVAE
        self.ss = ss
        if name == "vade" or name == "vademoe":
            self.vade = True
        else:
            self.vade = False            

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

            self.hidden = encoder_network(self.X, self.activation, self.initializer(), self.prob, reuse=None, cnn=self.cnn)
            self.reconstructed_Y_soft = tf.nn.softmax(tf.layers.dense(self.hidden, units=self.n_classes))

            if self.noVAE == False:
                
                self.mean, self.log_var = Z_network(self.hidden, self.activation, self.initializer(), self.latent_dim, reuse=None, cnn=self.cnn)

                priorFac = priors.DiscreteFactorial(
                            "cluster", 1, self.n_classes
                        )
                priorNormal = priors.NormalMixtureFactorial(
                            "representation", self.latent_dim, self.n_classes
                        )

                self.latent_variables.update({
                    "Z": (priorNormal, self.epsilon,
                        {
                            "mean": self.mean,
                            "log_var": self.log_var,
                            "cluster_sample": False
                        }
                    )
                })

                lv, eps, params = self.latent_variables["Z"]
                self.Z = lv.inverse_reparametrize(eps, params)

                if self.vade:
                    self.cluster_probs = lv.get_cluster_probs(self.Z)
                else:
                    self.cluster_probs = C_network(self.hidden, self.activation, self.initializer(), self.n_classes, reuse=None, cnn=self.cnn)
                
                params["weights"] = self.cluster_probs
                self.decoded_X = decoder_network(self.Z, self.activation, self.initializer(), self.input_dim, reuse=None, cnn=self.cnn)

                self.latent_variables.update({
                    "C": (priorFac, self.cluster,{"probs": self.cluster_probs}),
                })
                

            if self.ss:
                self.hidden_unl = encoder_network(self.X_unl, self.activation, self.initializer(), self.prob, cnn=self.cnn)
                self.mean_unl, self.log_var_unl = Z_network(self.hidden_unl, self.activation, self.initializer(), self.latent_dim, reuse=None, cnn=self.cnn)

                self.latent_variables_unl.update({
               
                    "Z": (priorNormal, self.epsilon_unl,
                        {
                            "mean": self.mean_unl,
                            "log_var": self.log_var_unl, 
                            "cluster_sample": False
                        }
                    )
                })

                lv, eps, params = self.latent_variables_unl["Z"]
                self.Z_unl = lv.inverse_reparametrize(eps, params)
                
                if self.vade:
                    self.cluster_probs_unl = lv.get_cluster_probs(self.Z_unl)
                else:
                    self.cluster_probs_unl = C_network(self.hidden_unl, self.activation, self.initializer(), self.n_classes, reuse=None, cnn=self.cnn)

                params["weights"] = self.cluster_probs_unl
                self.decoded_X_unl = decoder_network(self.Z_unl, self.activation, self.initializer(), self.input_dim, reuse=None, cnn=self.cnn)
        
                self.latent_variables_unl.update({
                    "C": (priorFac, self.cluster_unl,{"probs": self.cluster_probs_unl}),
                })
        return self

    def define_pretrain_step(self, vae_lr, prior_lr):
        self.define_train_loss()

        self.vae_loss = self.recon_loss
        self.vae_train_step = tf.train.AdamOptimizer(
            learning_rate=vae_lr
        ).minimize(self.recon_loss)

        # Only in DV-MOE
        # prior_var_list = tf.get_collection(
        #     tf.GraphKeys.TRAINABLE_VARIABLES, scope="dvmoe/dvmoe/dvmoe/" + self.name + "/c" ########################## Not back compatible       #################
        # )
        # # + tf.get_collection(
        # #     tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + "/representation"
        # # )

        # self.prior_train_step = tf.train.AdamOptimizer(
        #     learning_rate=prior_lr
        # ).minimize(self.latent_loss, var_list=prior_var_list)

    def pretrain_vae(self, session, data, n_epochs, ss):
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
                    if ss:
                        batch = batch[1][0]
                    feed = {
                        self.X: batch,
                        self.epsilon: np.zeros(
                            (len(batch), self.latent_dim)
                        ),
                    }

                    batch_loss, _ = session.run(
                        [self.recon_loss, self.vae_train_step], feed_dict=feed
                    )
                    loss += batch_loss / data.epoch_len
                bar.set_postfix({"loss": "%.4f" % loss})

                if loss <= min_loss:
                    min_loss = loss
                    saver.save(session, ckpt_path)

    def pretrain_prior(self, session, data, n_epochs, ss):
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        saver = tf.train.Saver(var_list)
        ckpt_path = self.path + "/prior/parameters.ckpt"

        try:
            saver.restore(session, ckpt_path)
        except:
            print("Could not load trained prior parameters")

            if n_epochs > 0:

                Z = []
                for batch in data.get_batches():
                    
                    # import pdb; pdb.set_trace()
                    # if ss:
                    #     data = batch[1][0]
                    # else:
                    #     data = batch.data

                    feed = {
                        self.X: batch
                    }
                    Z_batch = session.run(self.mean, feed_dict=feed)
                    Z.append(Z_batch)

                Z = np.concatenate(Z, axis=0)

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

        # min_loss = float("inf")
        # with tqdm(range(n_epochs)) as bar:
        #     for _ in bar:
        #         loss = 0
        #         for batch in data.get_batches():
        #             if ss:
        #                batch = batch[1][0] 
        #             feed = {
        #                 self.X: batch,
        #                 self.epsilon: np.zeros(
        #                     (len(batch), self.latent_dim)
        #                 ),
        #             }

        #             batch_loss, _ = session.run(
        #                 [self.latent_loss, self.prior_train_step], feed_dict=feed
        #             )
        #             loss += batch_loss / data.epoch_len

        #         bar.set_postfix({"loss": "%.4f" % loss})

        #         if loss <= min_loss:
        #             min_loss = loss
        #             saver.save(session, ckpt_path)

    def pretrain(self, session, data, n_epochs_vae, n_epochs_gmm, ss=0):
        assert(
            self.vae_train_step is not None# and
        #    self.prior_train_step is not None
        )

        self.pretrain_vae(session, data, n_epochs_vae, ss)
        self.pretrain_prior(session, data, n_epochs_gmm, ss)

    def get_accuracy(self, session, data, k=10):
        weights = []
        # for _ in range(k):
        clusterProb = []
        for batch in data.get_batches():
            feed = {self.X: batch,
                    self.prob: 0.0}
            feed.update(
                self.sample_reparametrization_variables(
                    len(batch), variables=["Z"]
                )
            )
            clusterProb.append(session.run(self.cluster_probs, feed_dict=feed))

        clusterProb = np.concatenate(clusterProb, axis=0)
        #     weights.append(clusterProb)

        # weights = np.array(weights)
        # weights = np.mean(weights, axis=0)

        return get_clustering_accuracy(clusterProb, data.classes)

