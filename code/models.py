import numpy as np
import tensorflow as tf

from tqdm import tqdm
from includes.utils import Dataset
from includes.network import FeedForwardNetwork, DeepNetwork
from base_models import DeepMixtureVAE, VaDE
from includes.utils import get_clustering_accuracy

class MoE:
    def __init__(self, name, input_type, input_dim, latent_dim, n_classes, n_experts, classification, activation=None, initializer=None, lossVAE=1, featLearn=1, cnn=1, ss=0):
        self.name = name

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        self.input_type = input_type
        self.classification = classification

        self.n_experts = n_experts

        self.activation = activation
        self.initializer = initializer

        self.vae = None
        self.featLearn = featLearn
        self.lossVAE = lossVAE

        self.cnn = cnn
        self.ss = ss


    def _define_vae(self):
        raise NotImplementedError

    def define_vae(self):
        with tf.variable_scope(self.name) as _:
            self._define_vae()

    def build_graph(self):
        with tf.variable_scope(self.name) as _:
            self.define_vae()

            self.X = self.vae.X
            if self.ss:
               self.X_unl = self.vae.X_unl
            self.Z = self.vae.Z
            self.Y = tf.placeholder(
                tf.float32, shape=(None, self.n_classes), name="Y"
            )

            # self.logits = self.vae.logits

            #self.reconstructed_X = self.vae.reconstructed_X

            self.regression_biases = tf.get_variable(
                "regression_biases", dtype=tf.float32,
                initializer=tf.initializers.zeros,
                shape=(self.n_classes, self.n_experts)
            )
            if self.featLearn:
                # input_dim = self.latent_dim
                # inp2cls = tf.nn.relu(self.Z)
                inp2cls = tf.nn.relu(self.vae.hidden)
                input_dim = inp2cls.shape[-1]
                print("="*100)
            else:
                input_dim = self.input_dim
                inp2cls = self.X

            self.regression_weights = tf.get_variable(
                "regression_weights", dtype=tf.float32,
                initializer=tf.initializers.random_normal,
                shape=(self.n_experts, self.n_classes, input_dim)
            )

            self.expert_probs = self.vae.cluster_probs

            expert_predictions = tf.transpose(tf.matmul(
                self.regression_weights,
                tf.tile(
                    tf.transpose(inp2cls)[None, :, :], [self.n_experts, 1, 1]
                )
            )) + self.regression_biases

            if self.classification:
                expert_class_probs = tf.nn.softmax(
                    tf.transpose(expert_predictions, (0, 2, 1))
                )

                unnorm_class_probs = tf.reduce_sum(
                    expert_class_probs * self.expert_probs[:, :, None], axis=1
                )
                self.reconstructed_Y_soft = unnorm_class_probs / tf.reduce_sum(
                     unnorm_class_probs, axis=-1, keep_dims=True
                )

                self.reconstructed_Y = tf.one_hot(
                    tf.reshape(
                        tf.nn.top_k(self.reconstructed_Y_soft).indices, (-1,)
                    ), self.n_classes
                )

                self.error = tf.reduce_sum(
                    tf.abs(self.Y - self.reconstructed_Y)
                ) / 2
            else:
                self.reconstructed_Y = tf.reduce_sum(
                    expert_predictions * self.expert_probs[:, None, :], axis=-1
                )

                self.error = tf.reduce_mean(
                    tf.square(self.reconstructed_Y - self.Y)
                ) * self.n_classes

            return self

    def sample_generative_feed(self, n, **kwargs):
        return self.vae.sample_generative_feed(n, **kwargs)

    def sample_reparametrization_variables(self, n):
        return self.vae.sample_reparametrization_variables(n)

    def get_accuracy(self, session, data):
        error = 0.0
        logits = []
        for X_batch, Y_batch, _ in data.get_batches():
            feed = {
                self.X: X_batch,
                self.Y: Y_batch,
                self.vae.prob: 1.0
            }
            feed.update(
                self.vae.sample_reparametrization_variables(len(X_batch))
            )

            batchLogits, batchError = session.run([self.vae.logits, self.error], feed_dict=feed)

            error += batchError
            logits.append(batchLogits)

        logits = np.concatenate(logits, axis=0)
        try:
           accClustering = get_clustering_accuracy(logits, data.classes)
        except:
           accClustering = - 1.0
        if self.classification:
            error /= data.len
            return 1 - error, accClustering

        else:
            error /= data.epoch_len
            return -error, accClustering

    def define_train_loss(self):
        self.vae.define_train_loss()

        if self.classification:
            self.classificationLoss = -tf.reduce_mean(tf.reduce_sum(
                self.Y * tf.log(self.reconstructed_Y_soft + 1e-20), axis=-1
            ))*1000.0
        else:
            self.classificationLoss = 0.5 * tf.reduce_mean(
                tf.square(self.reconstructed_Y - self.Y)
            ) * self.n_classes

        self.loss = self.classificationLoss

        if self.lossVAE:
            
            self.loss += self.vae.loss

    def define_pretrain_step(self, init_lr, decay_steps, decay_rate=0.9):
        self.vae.define_train_step(
            init_lr, decay_steps, decay_rate
        )

    def define_train_step(self, init_lr, decay_steps, decay_rate=0.9, pretrain_init_lr=None,
                          pretrain_decay_steps=None, pretrain_decay_rate=None):
        self.define_train_loss()

        learning_rate = tf.train.exponential_decay(
            learning_rate=init_lr,
            global_step=0,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )

        self.train_step = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        ).minimize(self.loss)

    def pretrain(self, session, data, n_epochs):
        print("Pretraining Model")
        data = Dataset((data.data, data.classes),
                       data.batch_size, data.shuffle)

        with tqdm(range(n_epochs)) as bar:
            for _ in bar:
                self.vae.train_op(session, data)

    def train_op(self, session, data, kl_ratio=1.0):
        assert(self.train_step is not None)

        loss = 0.0
        lossCls = 0.0
        k = 0

        if self.ss:
            for ((X_batch, _, _), (X_batch_lbl, Y_batch, _)) in data.get_batches():
                feed = {
                    self.X: X_batch_lbl,
                    self.X_unl : X_batch,
                    self.Y: Y_batch,
                    self.vae.kl_ratio: kl_ratio,
                    self.vae.prob: .5 
                }
                feed.update(
                    self.vae.sample_reparametrization_variables(len(X_batch_lbl))
                )
                feed.update(
                    self.vae.sample_reparametrization_variables(len(X_batch), ss=True)
                )

                batch_error, batch_loss, _, batch_lossCls = session.run(
                    [self.error, self.loss, self.train_step, self.classificationLoss],
                    feed_dict=feed
                )

                lossCls +=  batch_lossCls / data.epoch_len
                loss += batch_loss / data.epoch_len

                k+=1
                if k > 3:
                    break

        else:

            for (X_batch, Y_batch, _) in data.get_batches():

                feed = {
                    self.X: X_batch,
                    self.Y: Y_batch,
                    self.vae.kl_ratio: kl_ratio,
                    self.vae.prob: .5 
                }
                feed.update(
                    self.vae.sample_reparametrization_variables(len(X_batch))
                )

                batch_error, batch_loss, _, batch_lossCls = session.run(
                    [self.error, self.loss, self.train_step, self.classificationLoss],
                    feed_dict=feed
                )
           
            
                lossCls +=  batch_lossCls / data.epoch_len
                loss += batch_loss / data.epoch_len

                k+=1
                if k > 3:
                    break

        if self.classification:
           batch_acc = 1 - batch_error/Y_batch.shape[0]
        else:
           batch_acc = - batch_error
        
        return loss, batch_acc, lossCls

    def debug(self, session, data, kl_ratio=1.0):
        import pdb

        for ((X_batch, dummy_y, _), (X_batch_lbl, Y_batch, _)) in data.get_batches():

            feed = {
                self.X: X_batch_lbl,
                self.Y: Y_batch,
                self.vae.kl_ratio: kl_ratio,
                self.is_unlabeled: -1,
                self.vae.prob: 1.0
            }
            feed.update(
                self.vae.sample_reparametrization_variables(len(X_batch_lbl))
            )

            pdb.set_trace()

            break

class handler:
    def __init__(self, name, input_type, input_dim, n_classes, activation=None, initializer=None, cnn=1):

        self.name = name

        self.input_dim = input_dim
        self.n_classes = n_classes

        self.input_type = input_type

        self.activation = activation
        self.initializer = initializer

        self.cnn = cnn
        self.vae = None

    def _define_vae(self):
        raise NotImplementedError

    def define_vae(self):
        with tf.variable_scope(self.name) as _:
            self._define_vae()

    def build_graph(self):

        with tf.variable_scope(self.name) as _:
            self.define_vae()

            self.X = self.vae.X
            self.Y = tf.placeholder(
                tf.float32, shape=(None, self.n_classes), name="Y"
            )

            self.reconstructed_Y_soft = self.vae.reconstructed_Y_soft
            print("="*100)
            self.reconstructed_Y = tf.one_hot(
                tf.reshape(
                    tf.nn.top_k(self.reconstructed_Y_soft).indices, (-1,)
                ), self.n_classes
            )

            self.error = tf.reduce_sum(
                tf.abs(self.Y - self.reconstructed_Y)
            ) / 2

            return self

    def get_accuracy(self, session, data):
        error = 0.0
        logits = []
        for X_batch, Y_batch, _ in data.get_batches():

            feed = {
                self.X: X_batch,
                self.Y: Y_batch,
                self.vae.prob: 1.0
            }

            batchError = session.run(self.error, feed_dict=feed)

            error += batchError

        accClustering = - 1.0

        error /= data.len
        return 1 - error, accClustering


    def define_train_loss(self):

        self.classificationLoss = -tf.reduce_mean(tf.reduce_sum(
            self.Y * tf.log(self.reconstructed_Y_soft + 1e-20), axis=-1
        ))*1000.0

        self.loss = self.classificationLoss

    def define_train_step(self, init_lr, decay_steps, decay_rate=0.9, pretrain_init_lr=None,
                          pretrain_decay_steps=None, pretrain_decay_rate=None):
        self.define_train_loss()

        learning_rate = tf.train.exponential_decay(
            learning_rate=init_lr,
            global_step=0,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )

        self.train_step = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        ).minimize(self.loss)


    def train_op(self, session, data, kl_ratio=1.0):

        loss = 0.0
        lossCls = 0.0
        k = 0
        for ((X_batch, dummy_y, _), (X_batch_lbl, Y_batch, _)) in data.get_batches():
           
            feed = {
                self.X: X_batch_lbl,
                self.Y: Y_batch,
                self.vae.kl_ratio: kl_ratio,
                self.vae.prob: .5
            }

            batch_error, batch_loss, _, batch_lossCls = session.run(
                [self.error, self.loss, self.train_step, self.classificationLoss],
                feed_dict=feed
            )
            
            lossCls +=  batch_lossCls / data.epoch_len
            loss += batch_loss / data.epoch_len
            k+=1

            if k > 3:
              break

            batch_acc = 1 - batch_error/Y_batch.shape[0]
        
        return loss, batch_acc, lossCls

    def debug(self, session, data, kl_ratio=1.0):
        import pdb

        for ((X_batch, dummy_y, _), (X_batch_lbl, Y_batch, _)) in data.get_batches():

            feed = {
                self.X: X_batch_lbl,
                self.Y: Y_batch,
                self.vae.kl_ratio: kl_ratio,
                self.vae.prob: 1.0
            }

            pdb.set_trace()
            break





class Supervised(handler):
    # if self.noVAE == False:
    def __init__(self, name, input_type, input_dim, n_classes, activation=None, initializer=None, cnn=1):
        handler.__init__(self, name, input_type, input_dim, n_classes, activation=activation, initializer=initializer, cnn=cnn)

    def _define_vae(self):
        with tf.variable_scope(self.name) as _:
            self.vae = DeepMixtureVAE(
                "null_vae", self.input_type, self.input_dim, -1, -1
                , activation=self.activation, initializer=self.initializer, cnn=self.cnn, noVAE=True
            ).build_graph()



class DeepMoE(MoE):
    def __init__(self, name, input_type, input_dim, n_classes, n_experts, classification, activation=None, initializer=None, featLearn=0, cnn=1):
        MoE.__init__(self, name, input_type, input_dim, 1, n_classes, n_experts,
                     classification, activation=activation, initializer=initializer, lossVAE=0, featLearn=featLearn, cnn=cnn)

    def _define_vae(self):
        with tf.variable_scope(self.name) as _:
            self.vae = DeepMixtureVAE(
                "null_vae", self.input_type, self.input_dim, self.latent_dim,
                self.n_experts, activation=self.activation, initializer=self.initializer, cnn=self.cnn
            ).build_graph()

class DeepVariationalMoE(MoE):
    def __init__(self, name, input_type, input_dim, latent_dim, n_classes, n_experts, classification, activation=None, initializer=None, featLearn=1, cnn=1, ss=0):
        MoE.__init__(self, name, input_type, input_dim, latent_dim, n_classes, n_experts,
                     classification, activation=activation, initializer=initializer, featLearn=featLearn, cnn=cnn, ss=ss)

    def _define_vae(self):
        with tf.variable_scope(self.name) as _:
            self.vae = DeepMixtureVAE(
                "deep_mixture_vae", self.input_type, self.input_dim, self.latent_dim,
                self.n_experts, activation=self.activation, initializer=self.initializer, cnn=self.cnn, ss=self.ss
            ).build_graph()


class VaDEMoE(MoE):
    def __init__(self, name, input_type, input_dim, latent_dim, n_classes, n_experts, classification, activation=None, initializer=None, featLearn=1, cnn=1):
        MoE.__init__(self, name, input_type, input_dim, latent_dim, n_classes, n_experts,
                     classification, activation=activation, initializer=initializer, featLearn=featLearn, cnn=cnn)

    def _define_vae(self):
        with tf.variable_scope(self.name) as _:
            self.vae = VaDE(
                "vade", self.input_type, self.input_dim, self.latent_dim,
                self.n_experts, activation=self.activation, initializer=self.initializer, cnn=self.cnn
            ).build_graph()

