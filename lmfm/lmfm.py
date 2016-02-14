from __future__ import absolute_import
import numpy as np
from lmfm_fast import FM as _FM
from scipy.sparse import dok_matrix
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y, assert_all_finite, check_array
__author__ = "mathewlee11"


class LMFM(BaseEstimator, RegressorMixin):
    """
    Factorization machine fitted with Alternating Least Squares

    Parameters
    ----------

    n_topics : int
        The dimensionality of the factorized 2-way interactions
    n_iters : int
        Number of iterations
    test_size: double
        Percent of the training set to use for validation
        Defaults to 0.01
    k0 : bool
        Use bias. Defaults to True.
    k1 : bool
        Use 1-way interactions (learn feature weights).
        Defaults to true.
    w0 : double
        Initial global bias.
        Defaults to 0.0
    lambda_w0 : double
        regularization hyperparameter for w0
        defaults to 0.0
    init_stdev : double, optional
        Standard deviation for initialization of 2-way factors.
        Defaults to 0.01.
    learning_rate_schedule : string, optional
        The learning rate:
            constant: eta = eta0
            optimal: eta = 1.0/(t+t0) [default]
            invscaling: eta = eta0 / pow(t, power_t)

    shuffle: bool
        Whether or not to shuffle training dataset before learning

    task : string
        regression: Labels are real values.
        classification: Labels are either positive or negative.
    seed : int
        The seed of the pseudo random number generator
    verbose : bool
        Whether or not to print current iteration, training error


    """
    def __init__(self,
                 n_topics=10,
                 n_iter=3,
                 k0=True,
                 k1=True,
                 lambda_w=0.0,
                 lambda_v=0.0,
                 min_target=None,
                 max_target=None,
                 shuffle=False,
                 init_stdev=0.01,
                 seed=42,
                 verbose=True
                 ):

        self.n_topics = n_topics
        self.n_iter = n_iter
        self.k0 = k0
        self.k1 = k1
        self.lambda_w = lambda_w
        self.lambda_v = lambda_v
        self.min_target = min_target
        self.max_target = max_target
        self.shuffle = shuffle
        self.init_stdev = init_stdev
        self.seed = seed
        self.verbose = verbose

    def fit(self, X, y):
        """Fit factorization machine using Alternating Least Squares

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data

        y : numpy array of shape [n_samples]
            Target values

        Returns
        -------
        self : returns an instance of self.
        """

        if type(X) == dok_matrix:
            raise TypeError('dok_matrix not supported')
        check_X_y(X, y, accept_sparse=['csr', 'csc'])
        check_array(X)

        if self.min_target is None:
            min_target = -np.inf
        else:
            min_target = self.min_target

        if self.max_target is None:
            max_target = np.inf
        else:
            max_target = self.max_target

        verbose = int(self.verbose)
        shuffle = int(self.shuffle)
        k0 = int(self.k0)
        k1 = int(self.k1)
        self.fm = _FM(self.n_topics, self.n_iter, k0, k1,
                      self.lambda_w, self.lambda_v, min_target,
                      max_target, shuffle, self.init_stdev,
                      self.seed, verbose)
        self.fm.fit(X, y)
        return self

    @property
    def V(self):
        return np.asarray(self.fm.V)

    @property
    def w(self):
        return np.asarray(self.fm.w)

    def predict(self, X):
        """Predict using the factorization machine

        Parameters
        ----------
        X : sparse matrix, shape = [n_samples, n_features]


        Returns
        -------

        array, shape = [n_samples]
           Predicted target values per element in X.
        """
        assert_all_finite(X)
        return self.fm.predict(X)
