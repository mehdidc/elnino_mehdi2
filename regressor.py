from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.pipeline import make_pipeline


class Regressor(BaseEstimator):
    def __init__(self, **kwargs):
        self.clf = make_pipeline(
            GradientBoostingRegressor(n_estimators=302,
                                      max_features=122,
                                      max_depth=5,
                                      learning_rate=0.039185610530855515)
        )

    def fit(self, X, y):
        self.clf.fit(X, y.ravel())

    def predict(self, X):
        return self.clf.predict(X)
