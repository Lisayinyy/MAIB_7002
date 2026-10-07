"""Model factories. Each returns an unfitted estimator with a scikit-learn fit/predict interface.

Add a new model by adding a factory here, then tune it with ``evaluate.cross_validate`` and
score it with ``evaluate.fit_predict_test``. Nothing else needs to change.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C


class FeatureWeights(BaseEstimator, TransformerMixin):
    """Multiply each (already standardized) feature by a weight.

    A weight of 2 makes that feature count twice as much when kNN measures how similar
    two rows are. Features not listed keep weight 1.
    """

    def __init__(self, weights=None, features=C.FEATURES):
        self.weights = weights
        self.features = features

    def fit(self, X, y=None):
        self.weights_ = np.array([(self.weights or {}).get(c, 1.0) for c in self.features])
        return self

    def transform(self, X):
        return np.asarray(X) * self.weights_


def make_knn(k=25, weights=None, features=C.FEATURES, **kwargs):
    """Standard scaling, optional feature weights, Euclidean k-nearest-neighbours regression."""
    return make_pipeline(StandardScaler(), FeatureWeights(weights, features), KNeighborsRegressor(n_neighbors=k, **kwargs))


def make_rf(max_depth=10, min_leaf=3, n_trees=300, max_features=0.5, random_state=0, **kwargs):
    """Random forest with squared-error splits. ``max_depth`` may be an int, None or "No limit"."""
    depth = None if max_depth in (None, "No limit") else int(max_depth)
    return RandomForestRegressor(n_estimators=n_trees, max_depth=depth, min_samples_leaf=min_leaf,
                                 max_features=max_features, n_jobs=-1, random_state=random_state, **kwargs)


def tree_depth(model):
    """Depth of the deepest tree in a fitted forest, for ``cross_validate(describe=...)``."""
    return {"actual_depth": max(tree.get_depth() for tree in model.estimators_)}


def feature_importance(model, features=C.FEATURES):
    import pandas as pd
    return pd.Series(model.feature_importances_, features).sort_values(ascending=False)


def make_catboost(depth=6, learning_rate=0.05, iterations=500, loss="RMSE", random_state=0, **kwargs):
    """Gradient-boosted trees (CatBoost). Starting point for tuning, not a locked design.

    ``loss`` is the training objective ("RMSE" matches the forest's squared-error splits;
    "MAE" matches the main metric directly).
    """
    from catboost import CatBoostRegressor

    return CatBoostRegressor(depth=depth, learning_rate=learning_rate, iterations=iterations,
                             loss_function=loss, random_seed=random_state, verbose=0,
                             allow_writing_files=False, thread_count=-1, **kwargs)


def make_ensemble(members=None, weights=None):
    """Average the predictions of several models. Default: the locked forest and CatBoost, equal weights."""
    from sklearn.ensemble import VotingRegressor

    members = members or [("rf", make_rf()), ("catboost", make_catboost())]
    return VotingRegressor(members, weights=weights)
