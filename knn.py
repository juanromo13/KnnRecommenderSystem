'''K-nearest Neighbor'''
import numpy as np
from sklearn.neighbors import NearestNeighbors


def train_model(X: np.ndarray, K: int) -> NearestNeighbors:
    """Train knn model

    Args:
        X (np.ndarray): (n, d) array holding the data
        K (int): number of neighbors

    Returns:
        NearestNeighbors: model
    """
    neigh = NearestNeighbors(n_neighbors=K)
    neigh.fit(X)
    return neigh


def fill_matrix(X: np.ndarray, K: int) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X (np.ndarray): (n, d) array of incomplete data (incomplete entries =0)
        K (int): number of neighbors

    Returns:
        np.ndarray: a (n, d) array with completed data
    """
    n, _ = X.shape
    X_pred = X.copy()

    for u in range(n):
        indZero = np.argwhere(X[u] == 0).reshape(-1)
        for i in indZero:
            mask = X[:, i] != 0
            X_mask = X[mask, :]
            model = train_model(X_mask, K)
            sim, indexes = model.kneighbors([X[u]], n_neighbors=K)

            sim = np.reshape(sim, -1)
            indexes = np.reshape(indexes, -1)

            X_pred[u, i] = (sim * X_mask[indexes, i]).sum() / sim.sum()

    return X_pred
