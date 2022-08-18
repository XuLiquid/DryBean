import numpy as np
class PCA(object):
    def __init__(self):
        self.W = []
        self.means = []

    def fit_and_transform(self, X, d):
        means = np.mean(X, 0)
        self.means = means
        X = X - means
        covM = np.dot(X.T, X)
        eigval, eigvec = np.linalg.eig(covM)
        indexes = np.argsort(eigval)[-d:]
        W = eigvec[:, indexes]
        self.W = W
        return np.dot(X, W)

    def transform(self, X):
        X = X - self.means
        return np.dot(X, self.W)
