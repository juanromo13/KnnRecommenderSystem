import numpy as np
import knn
import common

from sklearn.metrics import r2_score

"""
X = | 1 2 0 1 3 | user 1 no califico la pelicula 2
    | 5 0 0 2 4 | user 2 no califico la pelicula 1 y 2
    | 1 2 5 0 1 | user 3 no califico la pelicula 3

K = numero de usuarios similares 
"""

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 3

X_pred = knn.fill_matrix(X, K)
# print(X_pred)
print(f'R2 score: {r2_score(X_gold, X_pred)}, RMSE: {common.rmse(X_gold, X_pred)}')
