# Computational Statistics (MVA) - Project

Experiments based on the paper
> Mattei, Pierre-Alexandre, et Jes Frellsen. Leveraging the Exact Likelihood of Deep Latent Variable Models. arXiv:1802.04826, arXiv, 28 juin 2018. arXiv.org, https://doi.org/10.48550/arXiv.1802.04826.

## Installation

In an environment with Python 3.8: `pip install -e .`

## Use

You can find different configurations in `cs/configs`.

To train the models, you can use the command `csrun`, like below:
- `csrun --config-name freyfaces ++loss.ksi=0`
- `csrun --config-name freyfaces ++loss.ksi=0.0625`
- `csrun --config-name freyfaces ++loss.early_stopping=2500`
- `csrun --config-name mnist`
- `csrun --config-name fashion_mnist`

To run imputation experiments, you can run the command `csimput`, like below (specifying the folder containing the trained model weights):
- `csimput --repertory outputs\fashion_mnist_elbo_unconstrained_bernouilli_2024-01-06\12-48-07`

To test the model (generate new samples), you can run:
- `cstest --repertory outputs\fashion_mnist_elbo_unconstrained_bernouilli_2024-01-06\12-48-07`
- `cstest --repertory outputs\freyfaces_elbo_0_gaussian_2024-01-06\15-45-02 --average --n_samples 25`

To compute the upper bound with a GMM, you can run: `python find_bound_gmm.py`.

## Others

Tensorflow 2 implementation: https://colab.research.google.com/drive/1bm_IPyApRag3rYJnQot4M8G5JV9gPuzv
