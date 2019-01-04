import os
import math
import models
import argparse
import warnings
import base_models
import numpy as np
import tensorflow as tf
import matplotlib as mpl

from tqdm import tqdm
from visdom import Visdom

from matplotlib import pyplot as plt
from matplotlib import gridspec as grid

import includes.visualization as visualization
from includes.utils import load_data, generate_regression_variable

mpl.rc_file_defaults()

tf.logging.set_verbosity(tf.logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)

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
parser.add_argument("--pretrain_epochs_prior", type=int, default=200,
                    help="Number of epochs for pretraining the gmm model")

parser.add_argument("--init_lr", type=float, default=0.002,
                    help="Initial learning rate for training")
parser.add_argument("--decay_rate", type=float, default=0.9,
                    help="Decay rate for exponentially decaying learning rate (< 1.0)")
parser.add_argument("--decay_epochs", type=int, default=25,
                    help="Number of epochs between exponentially decay of learning rate")

parser.add_argument("--pretrain", action="store_true", default=False,
                    help="Whether to pretrain the model or not")

parser.add_argument("--pretrain_vae_lr", type=float, default=0.0005,
                    help="Initial learning rate for pretraining the vae")
parser.add_argument("--pretrain_decay_rate", type=float, default=0.9,
                    help="Decay rate for exponentially decaying learning rate (< 1.0) for pretraining")
parser.add_argument("--pretrain_decay_epochs", type=int, default=25,
                    help="Number of epochs between exponentially decay of learning rate for pretraining")

parser.add_argument("--pretrain_prior_lr", type=float, default=0.0005,
                    help="Initial learning rate for pretraining the prior")

parser.add_argument("--kl_annealing", action="store_true", default=False,
                    help="Whether to anneal the KL term while training or not")
parser.add_argument("--anneal_step", type=float, default=0.1,
                    help="Step size for annealing")
parser.add_argument("--anneal_epochs", type=int, default=1000,
                    help="Number of epochs before annealing the KL term")

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

    pretrain_vae_lr = argv.pretrain_vae_lr
    pretrain_prior_lr = argv.pretrain_prior_lr

    moe = model_str[-3:] == "moe"

    n_epochs = argv.n_epochs

    visdom = argv.visdom

    pretrain = args.pretrain
    pretrain_epochs_vae = argv.pretrain_epochs_vae
    pretrain_epochs_prior = argv.pretrain_epochs_prior

    pretrain_decay_rate = argv.pretrain_decay_rate
    pretrain_decay_epochs = argv.pretrain_decay_epochs

    kl_annealing = args.kl_annealing
    anneal_step = args.anneal_step
    anneal_epochs = args.anneal_epochs

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
                activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer, featLearn=argv.featLearn
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
                model_name, dataset.input_type, dataset.input_dim, dataset.input_shape, latent_dim,
                n_clusters, activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer
            ).build_graph()
        elif model_str == "vade":
            model = base_models.VaDE(
                model_name, dataset.input_type, dataset.input_dim, dataset.input_shape, latent_dim,
                n_clusters, activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer
            ).build_graph()

        train_data = np.concatenate(
            [dataset.train_data, dataset.test_data], axis=0
        )
        train_classes = np.concatenate(
            [dataset.train_classes, dataset.test_classes], axis=0
        )

        test_data = (dataset.test_data, dataset.test_classes)
        train_data = (train_data, train_classes)

    test_data = Dataset(test_data, batch_size=100)
    train_data = Dataset(train_data, batch_size=100)

    model.define_train_step(
        init_lr, train_data.epoch_len * decay_epochs, decay_rate
    )

    if pretrain:
        if model_str in ["dvmoe", "vademoe"]:
            model.define_pretrain_step(
                pretrain_vae_lr, train_data.epoch_len *
                pretrain_decay_epochs, pretrain_decay_rate
            )
        elif model_str in ["dmvae", "vade"]:
            model.define_pretrain_step(
                pretrain_vae_lr, pretrain_prior_lr
            )

    model.path = "saved-models/%s/%s" % (dataset.datagroup, model.name)
    for path in [model.path + "/" + x for x in ["model", "vae", "prior"]]:
        if not os.path.exists(path):
            os.makedirs(path)

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)

    if pretrain:
        if model_str in ["dvmoe", "vademoe"]:
            model.pretrain(
                sess, train_data, pretrain_epochs_vae
            )
        elif model_str in ["dmvae", "vade"]:
            model.pretrain(
                sess, train_data, pretrain_epochs_vae, pretrain_epochs_prior
            )

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    saver = tf.train.Saver(var_list)
    ckpt_path = model.path + "/model/parameters.ckpt"

    try:
        saver.restore(sess, ckpt_path)
    except:
        print("Could not load trained model")

    if visdom:
        viz = Visdom()
        options = dict(
            ytickmin=90,
            ytickmax=100,
            xlabel="Epochs",
            ylabel="Accuracy",
            title="Accuracy vs Time",
        )

    with tqdm(range(n_epochs)) as bar:
        bar.set_postfix({
            "max_acc": "%.4f" % 0.0,
            "test_acc": "%.4f" % 0.0,
            "train_acc": "%.4f" % 0.0,
            "loss": "%.4f" % float("inf"),
            "clustering_acc": "%.4f" % 0.0
        })

        max_acc = 0.0

        anneal_term = 0.0 if kl_annealing else 1.0

        for epoch in bar:
            if kl_annealing and (epoch + 1) % anneal_epochs == 0:
                anneal_term = min(anneal_term + anneal_step, 1.0)

            if epoch % save_epochs == 0:
                if moe:
                    test_acc, _ = model.get_accuracy(sess, test_data)
                    train_acc, clustering_acc = model.get_accuracy(
                        sess, train_data
                    )

                    if test_acc > max_acc:
                        max_acc = test_acc
                        saver.save(sess, ckpt_path)
                else:
                    train_acc = model.get_accuracy(sess, train_data)
                    test_acc = model.get_accuracy(sess, test_data)

                    clustering_acc = train_acc

                    if clustering_acc > max_acc:
                        max_acc = clustering_acc
                        saver.save(sess, ckpt_path)

            if plotting and epoch % plot_epochs == 0:
                if dataset.sample_plot is not None:
                    dataset.sample_plot(model, sess)

                if dataset.regeneration_plot is not None:
                    dataset.regeneration_plot(model, test_data, sess)

            if visdom:
                if epoch % 100 == 0:
                    win = viz.line(
                        X=np.arange(epoch, epoch + 0.1), Y=np.arange(0, .1)
                    )

                test_acc_, train_acc_ = test_acc, train_acc
                if epoch > 0:
                    viz.line(
                        X=np.linspace(epoch - 1, epoch, 50),
                        Y=np.linspace(train_acc_, train_acc, 50),
                        name="train", update="append", win=win, opts=options
                    )
                    viz.line(
                        X=np.linspace(epoch - 1, epoch, 50),
                        Y=np.linspace(test_acc_, test_acc, 50),
                        name="test", update="append", win=win, opts=options
                    )

            loss = model.train_op(sess, train_data, anneal_term)
            if math.isnan(loss):
                model.debug(sess, train_data)

            bar.set_postfix({
                "loss": "%.4f" % loss,
                "max_acc": "%.4f" % max_acc,
                "test_acc": "%.4f" % test_acc,
                "train_acc": "%.4f" % train_acc,
                "clustering_acc": "%.4f" % clustering_acc
            })

    if plotting:
        dataset.sample_plot(model, sess)
        dataset.regeneration_plot(model, test_data, sess)

    fl = open("logs/%s.txt" % model_name, "a+")
    fl.write("\n%s\n------\n" % str(argv))
    fl.write("Max Accuracy        %.4f\n============" % max_acc)


if __name__ == "__main__":
    args = parser.parse_args()

    if False:
        from pprint import PrettyPrinter

        pp = PrettyPrinter(indent=2)
        pp.pprint(vars(args))

    main(args)
