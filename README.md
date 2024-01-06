# comp_stats

`csrun --config-name freyfaces ++loss.ksi=0`
`csrun --config-name freyfaces ++loss.ksi=0.0625`
`csrun --config-name freyfaces ++loss.early_stopping=2500`

`csrun --config-name mnist`
`csrun --config-name fashion_mnist`

`cstest --repertory outputs\fashion_mnist_elbo_unconstrained_bernouilli_2024-01-06\12-48-07`
`csimput --repertory outputs\fashion_mnist_elbo_unconstrained_bernouilli_2024-01-06\12-48-07`
`cstest --repertory outputs\freyfaces_elbo_0_gaussian_2024-01-06\15-45-02 --average --n_samples 25`
`python find_bound_gmm.py`


https://colab.research.google.com/drive/1bm_IPyApRag3rYJnQot4M8G5JV9gPuzv
https://colab.research.google.com/drive/1EnzToNcidgM3HjdgQL4bgXRYnDtknvph#scrollTo=aW2fHCn2wds-
