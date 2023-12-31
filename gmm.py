from sklearn.mixture import GaussianMixture

from sklearn.model_selection import GridSearchCV

from torchvision.datasets import MNIST
import numpy as np

dataset = MNIST("mnist", download=True)
data = dataset.test_data.float().reshape(-1, 28*28)[:10000]
data = (data > 255/2).float()

# gmm = GaussianMixture(n_components=10, covariance_type='spherical', max_iter=100000, init_params="k-means++", verbose=2)
# gmm.fit(data)
# print(gmm.score(data))

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)

def gmm_likelihood_score(estimator, X):
  return np.mean(estimator.score_samples(X))

# [CV 4/5] END covariance_type=tied, n_components=20;, score=566.937 total time= 9.9min
# [CV 2/5] END covariance_type=spherical, n_components=100;, score=60.932 total time=  28.5s
param_grid = {
    "covariance_type": ["diag", "full", "spherical", "tied"],
    "n_components": [10,20,100] #list(range(8,12)),
}
grid_search = GridSearchCV(
    GaussianMixture(max_iter=10000, init_params="k-means++", verbose=2), param_grid=param_grid, scoring=gmm_likelihood_score, verbose=3,
    n_jobs=3
)
grid_search.fit(data)

print(grid_search.best_params_)
print(grid_search.best_score_)

import pandas as pd

df = pd.DataFrame(grid_search.cv_results_)[
    ["param_n_components", "param_covariance_type", "mean_test_score"]
]
df["mean_test_score"] = -df["mean_test_score"]
df = df.rename(
    columns={
        "param_n_components": "Number of components",
        "param_covariance_type": "Type of covariance",
        "mean_test_score": "Score",
    }
)
df.sort_values(by="Score").head()

np.mean(gmm_likelihood_score(grid_search.best_estimator_, data))

# {'covariance_type': 'spherical', 'n_components': 100}
# 59.567905902567645
