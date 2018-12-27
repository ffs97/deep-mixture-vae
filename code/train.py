import os
import models
import argparse
import base_models
import numpy as np
import tensorflow as tf
import matplotlib as mpl
from visdom import Visdom

from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib import gridspec as grid

import includes.visualization as visualization
from includes.utils import load_data, generate_regression_variable

mpl.rc_file_defaults()

tf.logging.set_verbosity(tf.logging.ERROR)

parser = argparse.ArgumentParser(
    description="Training file for DMVAE and DVMOE"
)


parser.add_argument("--model", type=str, default="dmvae",
                    help="Model to use [dmvae, vade, dmoe, dvmoe, vademoe]")
parser.add_argument("--model_name", type=str, default="",
                    help="Name of the model")
parser.add_argument("--dataset", type=str, default="mnist",
                    help="Dataset to use [mnist, spiral, cifar10]")

parser.add_argument("--latent_dim", type=int, default=10,
                    help="Number of dimensions for latent variable Z")
parser.add_argument("--output_dim", type=int, default=1,
                    help="Output dimension for regression variable for ME models")

parser.add_argument("--n_clusters", type=int, default=-1,
                    help="Number of clusters to use")
parser.add_argument("--n_experts", type=int, default=5,
                    help="Number of experts to use for MoE models")

parser.add_argument("--classification", action="store_true", default=False,
                    help="Whether the objective is classification or regression (ME models)")

parser.add_argument("--n_epochs", type=int, default=500,
                    help="Number of epochs for training the model")
parser.add_argument("--pretrain_epochs_vae", type=int, default=200,
                    help="Number of epochs for pretraining the vae model")
parser.add_argument("--pretrain_epochs_gmm", type=int, default=200,
                    help="Number of epochs for pretraining the gmm model")

parser.add_argument("--init_lr", type=float, default=0.002,
                    help="Initial learning rate for training")
parser.add_argument("--decay_rate", type=float, default=0.9,
                    help="Decay rate for exponentially decaying learning rate (< 1.0)")
parser.add_argument("--decay_epochs", type=int, default=25,
                    help="Number of epochs between exponentially decay of learning rate")

parser.add_argument("--pretrain_init_lr", type=float, default=0.002,
                    help="Initial learning rate for pretraining")
parser.add_argument("--pretrain_decay_rate", type=float, default=0.9,
                    help="Decay rate for exponentially decaying learning rate (< 1.0) for pretraining")
parser.add_argument("--pretrain_decay_epochs", type=int, default=25,
                    help="Number of epochs between exponentially decay of learning rate for pretraining")

parser.add_argument("--plotting", action="store_true", default=False,
                    help="Whether to generate sampling and regeneration plots")
parser.add_argument("--plot_epochs", type=int, default=100,
                    help="Nummber of epochs before generating plots")

parser.add_argument("--save_epochs", type=int, default=10,
                    help="Nummber of epochs before saving model")

parser.add_argument("--debug", action="store_true", default=False,
                    help="Whether to debug the models or not")

parser.add_argument("--visdom", action="store_true", default=False,
                    help="Using visdom for plotting")

parser.add_argument("--featLearn", action="store_true", default=False,
                    help="Whether to use feature learning in MOE")

def main(argv):
    dataset = argv.dataset

    latent_dim = argv.latent_dim
    output_dim = argv.output_dim

    model_str = argv.model
    model_name = argv.model_name

    plot_epochs = argv.plot_epochs
    save_epochs = argv.save_epochs

    plotting = argv.plotting
    debug = argv.debug

    classification = argv.classification

    init_lr = argv.init_lr
    decay_rate = argv.decay_rate
    decay_epochs = argv.decay_epochs

    pretrain_init_lr = argv.pretrain_init_lr
    pretrain_decay_rate = argv.pretrain_decay_rate
    pretrain_decay_epochs = argv.pretrain_decay_epochs

    moe = model_str[-3:] == "moe"

    n_epochs = argv.n_epochs
    pretrain_epochs_vae = argv.pretrain_epochs_vae
    pretrain_epochs_gmm = argv.pretrain_epochs_gmm

    dataset = load_data(
        dataset, classification=classification, output_dim=output_dim
    )

    if model_name == "":
        model_name = model_str

    if moe:
        n_experts = argv.n_experts

        if classification:
            output_dim = dataset.n_classes

        from includes.utils import MEDataset as Dataset

        if model_str not in ["dmoe", "vademoe", "dvmoe"]:
            raise NotImplementedError

        if model_str == "dmoe":
            model = models.DeepMoE(
                model_str, dataset.input_type, dataset.input_dim, output_dim, n_experts, classification,
                activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer
            ).build_graph()
            plotting = False

        elif model_str == "dvmoe":
            model = models.DeepVariationalMoE(
                model_str, dataset.input_type, dataset.input_dim, latent_dim, output_dim, n_experts,
                classification, activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer, featLearn=argv.featLearn
            ).build_graph()

        elif model_str == "vademoe":
            model = models.VaDEMoE(
                model_str, dataset.input_type, dataset.input_dim, latent_dim, output_dim, n_experts,
                classification, activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer, featLearn=argv.featLearn
            ).build_graph()

        test_data = (
            dataset.test_data, dataset.test_classes, dataset.test_labels
        )
        train_data = (
            dataset.train_data, dataset.train_classes, dataset.train_labels
        )

    else:
        n_clusters = argv.n_clusters
        if n_clusters < 1:
            n_clusters = dataset.n_classes

        from includes.utils import Dataset

        if model_str not in ["dmvae", "vade"]:
            raise NotImplementedError

        if model_str == "dmvae":
            model = base_models.DeepMixtureVAE(
                model_name, dataset.input_type, dataset.input_dim, latent_dim, n_clusters,
                activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer
            ).build_graph()
        elif model_str == "vade":
            model = base_models.VaDE(
                model_name, dataset.input_type, dataset.input_dim, latent_dim, n_clusters,
                activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer
            ).build_graph()
            #     {"Z": [512, 256, 256]}, [256, 256, 512]
            # )

        dataset.train_data = np.concatenate(
            [dataset.train_data, dataset.test_data], axis=0
        )
        dataset.train_classes = np.concatenate(
            [dataset.train_classes, dataset.test_classes], axis=0
        )

        test_data = (dataset.test_data, dataset.test_classes)
        train_data = (dataset.train_data, dataset.train_classes)

    test_data = Dataset(test_data, batch_size=100)
    train_data = Dataset(train_data, batch_size=100)

    model.define_train_step(
        init_lr, train_data.epoch_len * decay_epochs, decay_rate)

    if model_str in ["vade", "dvmoe", "vademoe"]:
        model.define_pretrain_step(
            pretrain_init_lr, train_data.epoch_len *
            pretrain_decay_epochs, pretrain_decay_rate
        )

    model.path = "saved_models/%s/%s" % (dataset.datagroup, model.name)
    for path in [model.path + "/" + x for x in ["model", "vae", "prior"]]:
        if not os.path.exists(path):
            os.makedirs(path)

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)

    if model_str in ["dvmoe", "vademoe"]:
        model.pretrain(
            sess, train_data, pretrain_epochs_vae
        )
    elif model_str in ["vade"]:
        model.pretrain(
            sess, train_data, pretrain_epochs_vae, pretrain_epochs_gmm
        )

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    saver = tf.train.Saver(var_list)
    ckpt_path = model.path + "/model/parameters.ckpt"

    try:
        saver.restore(sess, ckpt_path)
    except:
        print("Could not load trained model")

    if argv.visdom:
        #######  Preparation  ############
        viz = Visdom()
        options=dict(
                    ytickmin=90,
                    ytickmax=100,
                    xlabel='This X Label',
                    ylabel='This Y Label',
                    title='The Title',
                )
        ##################################


    with tqdm(range(n_epochs), postfix={"loss": "inf", "accTrain": "0.00%", "accTest": "0.00%"}) as bar:
        accuracy = 0.0
        maxAcc = 0.0

        for epoch in bar:
            # import pdb;pdb.set_trace()
            if plotting and epoch % plot_epochs == 0:
                if dataset.sample_plot is not None:
                    dataset.sample_plot(model, sess)

                if dataset.regeneration_plot is not None:
                    dataset.regeneration_plot(model, test_data, sess)

            if epoch % save_epochs == 0:
                if debug:
                    model.debug(sess, train_data)

            if moe:
                loss, accTrain = model.train_op(sess, train_data)
            else:
                loss = model.train_op(sess, train_data)
                accTrain = model.get_accuracy(sess, train_data)

            accTest = model.get_accuracy(sess, test_data)

            if accTest > maxAcc:
                maxAcc = accTest
                saver.save(sess, ckpt_path)


            if argv.visdom:
                if epoch % 100 == 0:
                    ################## To Refresh after some time ##################
                    win = viz.line(X=np.arange(epoch, epoch +.1), Y=np.arange(0, .1))

                accTest_old, accTrain_old = accTest, accTrain
                if epoch > 0:
                    viz.line(X=np.linspace(epoch-1, epoch,50), Y=np.linspace(accTrain_old, accTrain,50), name='1', update='append', win=win, opts=options)
                    viz.line(X=np.linspace(epoch-1, epoch,50), Y=np.linspace(accTest_old, accTest,50), name='2', update='append', win=win, opts=options)

            bar.set_postfix({
                "loss": "%.4f" % loss,
                "accTrain": "%.4f" % accTrain,
                "accTest" : "%.4f" % accTest,  
                "maxAcc" : "%.4f" % maxAcc,  
            })

    if plotting:
        dataset.sample_plot(model, sess)
        dataset.regeneration_plot(model, test_data, sess)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
