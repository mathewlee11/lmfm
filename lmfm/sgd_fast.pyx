# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.sparse.csr import csr_matrix
from libc.math cimport exp, sqrt, log
from libc.stdlib cimport malloc, free
cimport numpy as np
cimport cython

cdef class FMClassifier(object):

    cdef:
        int n_factors, n_iter, k0, k1, shuffle, seed, verbose, n_samples
        int n_features
        double lambda_w, lambda_v, init_stdev, learning_rate, w0
        double [:] w
        double [:] sum_
        double [:] sum_sqr_
        double [:] data
        double [:,:] v
        double [:, :] X
        int [:] y
        int [:] indptr
        int [:] indices



    def __init__(self, int n_factors=10, int n_iter=3, int k0=1, int k1=1,
                 int shuffle=0, double init_stdev=0.01, int seed=-1,
                 int verbose=1, double learning_rate=0.001):

        self.n_factors = n_factors
        self.n_iter = n_iter
        self.k0 = k0
        self.k1 = k1
        self.shuffle = shuffle
        self.init_stdev = init_stdev
        self.seed = seed
        self.verbose = verbose
        self.learning_rate = learning_rate

    cdef _predict_instance(self, int sample_ind):

        cdef double[:] sum_ = self.sum_
        cdef double[:] sum_sqr_ = self.sum_sqr_
        cdef double pred = self.w0
        cdef int num_nonzero = self.indptr[sample_ind+1] - self.indptr[sample_ind]
        cdef int feat_ind, i, f
        cdef double d
        cdef int [:] indices = self.indices
        cdef int [:] indptr = self.indptr
        cdef double [:] w = self.w
        cdef double [:] data = self.data
        cdef double [:, :] v = self.v
        cdef int n_factors = self.n_factors
        cdef int k1 = self.k1


        for i in range(n_factors):
            sum_[i] = 0.0
            sum_sqr_[i] = 0.0

        if k1:
            for i in range(num_nonzero):
                feat_ind = indices[indptr[sample_ind] + i]
                pred += w[feat_ind] * data[indptr[sample_ind] + i]

        for f in range(n_factors):
            sum_[f] = 0
            sum_sqr_[f] = 0
            for i in range(num_nonzero):
                feat_ind = indices[indptr[sample_ind] + i]
                d = v[f, feat_ind] * data[indptr[sample_ind] + i]
                sum_[f] += d
                sum_sqr_[f] += d*d
            pred += 0.5 * (sum_[f] * sum_[f] - sum_sqr_[f])
        self.sum_ = sum_
        return pred

    cdef _sgd_update(self):
        cdef double pred, err, gradv
        cdef int num_nonzero, feat_ind, i, f, j
        cdef double[:] w = self.w
        cdef double[:] data = self.data
        cdef double[:, :] v = self.v
        cdef int[:] indptr = self.indptr
        cdef int[:] indices = self.indices
        cdef double[:] sum_ = self.sum_
        cdef int[:] y = self.y
        cdef int k0 = self.k0
        cdef int k1 = self.k1
        cdef int n_factors = self.n_factors
        cdef double learning_rate = self.learning_rate
        cdef int n_samples = self.n_samples
        cdef double w0 = self.w0


        for i in range(n_samples):

            num_nonzero = indptr[i+1] - indptr[i]
            pred = self._predict_instance(i)
            err = y[i] * ((1.0/(1.0 + exp(-y[i]*pred))) - 1.0)

            if k0:
                w0 -= learning_rate * err

            if k1:
                for j in range(num_nonzero):
                    feat_ind = indices[indptr[i] + j]
                    w[feat_ind] -= learning_rate * err * data[indptr[i] + j]

            for f in range(n_factors):
                for j in range(num_nonzero):
                    feat_ind = indices[indptr[i] + j]
                    gradv = err * (data[indptr[i] + j]) * (sum_[f] - v[f, feat_ind] * data[indptr[i] + j])
                    v[f, feat_ind] -= learning_rate * gradv
        self.w0 = w0

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.v = np.random.normal(scale=self.init_stdev, size=(self.n_factors,
                                                               X.shape[1]))
        self.X = X
        y = y.copy().astype(np.int32)
        y[y == 0] = -1
        self.y = y
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.sum_ = np.zeros(self.n_factors)
        self.sum_sqr_ = np.zeros(self.n_factors)
        self.w0 = 0.0
        # Roll out the sparse matrix for super sweet optimization
        if type(X) != csr_matrix:
            X = csr_matrix(X)
        self.indptr = X.indptr
        self.indices = X.indices
        self.data = X.data
        for i in range(self.n_iter):
            self._sgd_update()

    def predict(self, X):
        if type(X) != csr_matrix:
            X = csr_matrix(X)
        self.indptr = X.indptr
        self.indices = X.indices
        self.data = X.data
        y = np.array([self._predict_instance(i) for i in range(X.shape[0])])
        y = (1.0 / (1.0 + np.exp(-y)))
        return y.round().astype(int)
