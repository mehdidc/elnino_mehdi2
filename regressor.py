from sklearn.base import BaseEstimator
from pyearth import Earth
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = make_pipeline(
                StandardScaler(),
                PCA(n_components=200),
                Earth(max_terms=20, max_degree=10)
        )

    def fit(self, X, y):
        self.clf.fit(X, y.ravel())

    def predict(self, X):
        return self.clf.predict(X)
