## Mixture of Experts using Discrete VAE

All files related to training the models reside in the code directory. To train the DMVAE model from scratch on MNIST dataset, simply run:

```python train.py```

This will automatically train the model and save the relevant reconstruction and generation plots. The parameters like model, dataset, etc can be controlled via command line arguments. To get a full list of all supported arguments, run:

```python train.py --helpshort```

Our code makes use of the following libraries:

* Tensorflow
* Numpy
* Sklearn
* Matplotlib
* tqdm
