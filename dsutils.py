"""
Decoding similarity helper functions.
"""

import torch
import warnings
import numpy as np
import numpy.typing as npt
from typing import Tuple
from sklearn.model_selection import KFold
from tqdm import tqdm  # optional for progress bar
from itertools import product
from torch import nn

def cross_val_score_custom(model_class, X, Z, param_grid, loss_fn, cv=5, kernel='linear'):
    
    param_combos = list(product(*param_grid.values()))

    best_score = float('inf')
    best_params = None
    
    # Dictionary to store all parameters tried and their coefficients
    all_params_coefs = {}

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    for params in tqdm(param_combos):
        param_dict = dict(zip(param_grid.keys(), params))
        losses = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            Z_train, Z_val = Z[train_idx], Z[val_idx]

            # Initialize and fit the model
            probe = model_class(**param_dict, kernel = kernel, center_columns=True, fit_intercept=False)
            probe.fit(X_train, Z_train)

            Z_pred = probe.predict(X_val)
            loss = loss_fn(Z_val, Z_pred, params)
            losses.append(loss)

            all_params_coefs[tuple(params)] = probe.coef_

        avg_loss = np.mean(losses)
        if avg_loss < best_score:
            best_score = avg_loss
            best_params = param_dict

    return best_params, best_score, X_train, Z_train, all_params_coefs

def mse_loss(y_true, y_pred, params):
    M = y_true.shape[0]
    return np.mean((y_true - y_pred) ** 2)

def inner_product_loss(y_true, y_pred, params):
    M = y_true.shape[0]
    a = params[0]
    return -np.trace(2* y_pred.T @ y_true - a* y_pred.T @ y_pred) #- y_true.T @ y_true)

def rbf_kernel(X, Y=None, gamma=1.0, center=False):
    """
    Evaluate the RBF (Gaussian) kernel between two sets of vectors.

    Parameters:
    ----------
    X : ndarray of shape (n_samples_X, n_features)
    Y : ndarray of shape (n_samples_Y, n_features), optional
        If None, computes the kernel between X and itself.
    gamma : float
        Kernel coefficient (1 / (2 * sigma^2)).

    Returns:
    -------
    K : ndarray of shape (n_samples_X, n_samples_Y)
        RBF kernel matrix.
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y) if Y is not None else X
    M = X.shape[0]
    Mp = Y.shape[0]

    # Squared Euclidean distance between each pair
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    dist_sq = X_norm + Y_norm - 2 * np.dot(X, Y.T)

    # RBF kernel matrix
    K = np.exp(-gamma * dist_sq)

    if center:
        C = np.eye(M) - (1/M)*np.ones((M,M))
        Cp = np.eye(Mp) - (1/Mp)*np.ones((Mp,Mp))
        K = C@K@Cp
        return K
    else:  
        return K

def linear_kernel(X, Y=None, center=False):
    """
    Evaluate the linear kernel between two sets of vectors.

    Parameters:
    ----------
    X : ndarray of shape (n_samples_X, n_features)
    Y : ndarray of shape (n_samples_Y, n_features), optional
        If None, computes the kernel between X and itself.
    Returns:
    -------
    K : ndarray of shape (n_samples_X, n_samples_Y)
        linear kernel matrix.
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y) if Y is not None else X
    M = X.shape[0]
    Mp = Y.shape[0]
    N = X.shape[1]

    # linear kernel matrix
    K = X@(Y.T)
    if center:
        C = np.eye(M) - (1/M)*np.ones((M,M))
        Cp = np.eye(Mp) - (1/Mp)*np.ones((Mp,Mp))
        K = C@K@Cp
        return K
    else:  
        return K

class genKernelRegression:
    
    def __init__(self,  center_columns=True, kernel = 'linear',  a=0, b=1, gamma = 1.0, fit_intercept=False):
        self.center_columns = center_columns
        self.a = a
        self.b = b
        self.gamma = gamma
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.kernel = kernel

    def _add_intercept(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X, Z):
        X = np.asarray(X)
        Z = np.asarray(Z)

        if self.center_columns:
            X = X - np.mean(X, axis=0)

        if self.fit_intercept:
            X = self._add_intercept(X)

        n_features = X.shape[1]
        # I = np.eye(n_features)
        Im = np.eye(X.shape[0])
        
        if self.fit_intercept:
            I[0, 0] = 0  # do not regularize intercept

        # Closed-form kernel regression solution:

        if self.kernel == 'rbf':
            KX = rbf_kernel(X, gamma = self.gamma, center=True)
        elif self.kernel == 'linear':
            KX = linear_kernel(X,center=True)
        else: 
            KX = linear_kernel(X,center=True)
            
        try:
            self.weights_ = np.linalg.solve(self.a*KX + self.b *Im, Im) @ Z
        except np.linalg.LinAlgError:
            print(f"Skipping hyperparams {self.a}, {self.b}, {self.gamma}: matrix is singular.")
            self.weights_ = np.full(Z.shape, np.nan)

        if self.fit_intercept:
            self.intercept_ = self.weights_[0, 0]
            self.coef_ = self.weights_[1:, 0]
        else:
            self.intercept_ = 0.0
            self.coef_ = self.weights_

        self.Xtrain = X
        
        return self

    def predict(self, X):
        X = np.asarray(X)
        M = X.shape[0]

        if self.kernel == 'rbf':
            KXXtrain = rbf_kernel(X,self.Xtrain, gamma = self.gamma, center=True)
        elif self.kernel == 'linear':
            KXXtrain = linear_kernel(X,self.Xtrain, center=True)
        else: 
            KXXtrain = linear_kernel(X,self.Xtrain, center=True)

        if self.fit_intercept:
            return self.intercept_ + KXXtrain @ self.coef_
        else:
            return KXXtrain @ self.coef_

    def score(self, X, Z):
        """R² score"""
        Z_pred = self.predict(X)
        ss_res = np.sum((Z - Z_pred) ** 2)
        ss_tot = np.sum((Z - np.mean(Z,axis=0)) ** 2)
        return 1 - ss_res / ss_tot

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class SoftMaxModule(nn.Module):
    def __init__(self):
        super(SoftMaxModule, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, x):
        return self.softmax(x)

def bespoke_cov_matrix(z):
    """
    Computes the covariance matrix for a custom set of tasks.  Assumes tasks are equally weighted.  

    Parameters
    ----------
    z : list of tasks.  Each element of list should be an array of (number of samples x 1).  These are the desired readouts for every input sample.  

    Returns
    -------
    Cz : ndarray
        M x M empirical covariance matrix.
    """

    n = len(z)

    Cz = 0
    for i in range(n):
        Cz = Cz + z[i]@z[i].T
    Cz = Cz/(n)

    return Cz

def random_partitions_cov_matrix(M, n):
    """
    Computes the empirical task covariance matrix for a set of n random binary partitions of the input samples.  Readouts are sampled from Rademacher distribution for each input sample. 

    Parameters
    ----------
    M : int (number of input samples)
    n : int (number of random tasks to generate)

    Returns
    -------
    Cz : ndarray
        M x M empirical covariance matrix.
    """
    z = []
    for i in range(n):
        zrand = np.random.randint(0,2,(M,1)) 
        zrand = 2*zrand - 1
        z.append(zrand)

    Cz = 0
    for i in range(n):
        Cz = Cz + z[i]@z[i].T
    Cz = Cz/(n)

    return Cz

def gaussian_partitions_cov_matrix(M, n):
    """
    Computes the empirical task covariance matrix if the desired readouts are a set of n samples from the M-dimensional standard normal distribution. 

    Parameters
    ----------
    M : int (number of input samples)
    n : int (number of random tasks to generate)

    Returns
    -------
    Cz : ndarray
        M x M empirical covariance matrix.
    """
    z = []
    for i in range(n):
        zrand = np.random.normal(0,1,(M,1)) 
        z.append(zrand)

    Cz = 0
    for i in range(n):
        Cz = Cz + z[i]@z[i].T
    Cz = Cz/(n)

    return Cz

def update_gaussian_cov_matrix(Cz, n, delta_n):
    """ Update a Gaussian covariance matrix generated by gaussian_partitions_cov_matrix(M, n) with delta_n more samples.  
    """
    M = np.shape(Cz)[0]
    new_cov_contrib = delta_n*gaussian_partitions_cov_matrix(M,delta_n)
    newCz = (1/(n + delta_n))*(n*Cz + new_cov_contrib)
    return newCz

def update_random_cov_matrix(Cz, n, delta_n):
    """ Update a random binary partitions covariance matrix generated by random_partitions_cov_matrix(M, n) with delta_n more samples.  
    """
    M = np.shape(Cz)[0]
    new_cov_contrib = delta_n*random_partitions_cov_matrix(M,delta_n)
    newCz = (1/(n + delta_n))*(n*Cz + new_cov_contrib)
    return newCz

class PartitionsCovMatrix:
    """
    Make task covariance matrices to evaluate average decoding similarity.  
    """
    def __init__(self, M, n_initial, method = 'binary'):
        self.M = M
        self.n = n_initial
        self.method = method
        self.matrix = None

    def initialize_cov_matrix(self):
        if self.method == 'binary':
            Cz = random_partitions_cov_matrix(self.M, self.n)
        elif self.method == 'gaussian':
            Cz = gaussian_partitions_cov_matrix(self.M, self.n)
        else:
            raise ValueError(
                "method must be either 'binary' or 'gaussian'.")
        self.matrix = Cz
        
        return None

    def update_cov_matrix(self, add_n):
        if self.method == 'binary':
            Cz = update_random_cov_matrix(self.matrix, self.n, add_n)
        elif self.method == 'gaussian':
            Cz = update_gaussian_cov_matrix(self.matrix, self.n, add_n)
        else:
            raise ValueError(
                "method must be either 'binary' or 'gaussian'.")
        
        self.n = self.n + add_n
        self.matrix = Cz
        
        return None

def whiten(
    X: npt.NDArray, 
    alpha: float, 
    preserve_variance: bool = True, 
    eigval_tol=1e-7
    ) -> Tuple[npt.NDArray, npt.NDArray]:
    """Return regularized whitening transform for a matrix X.

    Parameters
    ----------
    X : ndarray
        Matrix with shape `(m, n)` holding `m` observations
        in `n`-dimensional feature space. Columns of `X` are
        expected to be mean-centered so that `X.T @ X` is
        the covariance matrix.
    alpha : float
        Regularization parameter, `0 <= alpha <= 1`. When
        `alpha == 0`, the data matrix is fully whitened.
        When `alpha == 1` the data matrix is not transformed
        (`Z == eye(X.shape[1])`).
    preserve_variance : bool
        If True, rescale the (partial) whitening matrix so
        that the total variance, trace(X.T @ X), is preserved.
    eigval_tol : float
        Eigenvalues of covariance matrix are clipped to this
        minimum value.

    Returns
    -------
    X_whitened : ndarray
        Transformed data matrix.
    Z : ndarray
        Matrix implementing the whitening transformation.
        `X_whitened = X @ Z`.
    """

    # Return early if regularization is maximal (no whitening).
    if alpha > (1 - eigval_tol):
        return X, np.eye(X.shape[1])

    # Compute eigendecomposition of covariance matrix
    lam, V = np.linalg.eigh(X.T @ X)
    lam = np.maximum(lam, eigval_tol)

    # Compute diagonal of (partial) whitening matrix.
    # 
    # When (alpha == 1), then (d == ones).
    # When (alpha == 0), then (d == 1 / sqrt(lam)).
    d = alpha + (1 - alpha) * lam ** (-1 / 2)

    # Rescale the whitening matrix.
    if preserve_variance:

        # Compute the variance of the transformed data.
        #
        # When (alpha == 1), then new_var = sum(lam)
        # When (alpha == 0), then new_var = len(lam)
        new_var = np.sum(
            (alpha ** 2) * lam
            + 2 * alpha * (1 - alpha) * (lam ** 0.5)
            + ((1 - alpha) ** 2) * np.ones_like(lam)
        )

        # Now re-scale d so that the variance of (X @ Z)
        # will equal the original variance of X.
        d *= np.sqrt(np.sum(lam) / new_var)

    # Form (partial) whitening matrix.
    Z = (V * d[None, :]) @ V.T

    # An alternative regularization strategy would be:
    #
    # lam, V = np.linalg.eigh(X.T @ X)
    # d = lam ** (-(1 - alpha) / 2)
    # Z = (V * d[None, :]) @ V.T

    # Returned (partially) whitened data and whitening matrix.
    return X @ Z, Z