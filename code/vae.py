import os
import priors
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from sklearn.mixture import GaussianMixture

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
