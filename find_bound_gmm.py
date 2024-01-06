from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
import numpy as np
import scipy.io as sio
import pandas as pd

file_folder = "./cs/data/frey_rawface.mat"
matfile = sio.loadmat(file_folder)
faces = matfile['ff'].T.reshape(-1, 28*20).astype(np.float32)
train_size = int(0.9 * len(faces))
test_size = len(faces) - train_size

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)

def gmm_likelihood_score(estimator, X):
  return np.mean(estimator.score_samples(X))

param_grid = {
    "n_components": list(range(1,100)), # len(train_size)
    "covariance_type": ["spherical", "tied", "diag", "full"],
}

grid_search = GridSearchCV(
    GaussianMixture(), param_grid=param_grid, scoring=gmm_likelihood_score, verbose=3
)
grid_search.fit(faces[train_size:])

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
print(df.sort_values(by="Score").head())
print("Validation")
print(np.mean(gmm_likelihood_score(grid_search.best_estimator_, faces[train_size:])))
