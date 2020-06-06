"""Base class for metric learning."""

import copy
import gc
import timeit

# Numpy
import numpy as np
from numpy import linalg as LA
from numpy.core.umath_tests import inner1d
from numpy.random import RandomState

# Scipy
from scipy import spatial

# Sklearn
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score

# Mine
import LDA as LDA

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.simplefilter('ignore', UndefinedMetricWarning)
settings = np.seterr(invalid='raise')


class MetricBase(object):

    def __init__(self, maxcache=1.E9, seed=None, verbose=False, **kwargs):

        self._verbose = verbose

        # Base model and data parameters
        self.L = None
        self.X = None
        self.labels = None
        self.d = None
        self.D = None

        # Seed for random generator
        self._SEED = seed

        # Matrix product caching
        self._maxcache = maxcache
        self._Xij_prds = {}
        self._Xij_vals = {}
        self._N_Xij_entries = 0
        self._Xij_max_entries = 0
        self._active_consts = None

        # Penalization functions
        self._penfuns_map = {
            'logdet': {'fun': self.logdet_pen, 'd_fun': self.d_logdet_pen},
            'l1': {'fun': self.l1_pen, 'd_fun': self.d_l1_pen},
            'l2': {'fun': self.l2_pen, 'd_fun': self.d_l2_pen},
            'l1l2': {'fun': self.l1l2_pen, 'd_fun': self.d_l1l2_pen},
            'l2sqrd': {'fun': self.l2sqrd_pen, 'd_fun': self.d_l2sqrd_pen},
            'entropy': {'fun': self.entropy_pen, 'd_fun': self.d_entropy_pen},
            'frob': {'fun': self.frob_pen, 'd_fun': self.d_frob_pen},
            'elastic': {'fun': self.elastic_pen, 'd_fun': self.d_elastic_pen},
            'meancenter': {'fun': self.mean_center_loss, 'd_fun': self.d_mean_center_loss},
            'mean_loss': {'fun': self.mean_loss, 'd_fun': self.d_mean_loss},
            'trace': {'fun': self.trace_pen, 'd_fun': self.d_trace_pen},
            'trace_squared': {'fun': self.trace_sqrd_pen, 'd_fun': self.d_trace_sqrd_pen},
            }

        # Assign Penalization Function(s)
        self._penfun_list = []
        self._penfun_weights = {}
        self._penfuns = {}
        self._df_penfuns = {}

        # If weights for a pen function is in the kwargs we add that pen function to the loss fun
        for key, value in self._penfuns_map.items():
            if key in kwargs:
                if self._verbose:
                    print('Using regularization', key)
                self._penfun_list.append(key)
                self._penfuns[key] = value['fun']
                self._df_penfuns[key] = value['d_fun']
                self._penfun_weights[key] = kwargs[key]

    # L INITIALIZATION FUNCTION

    def _init_trans_mat(self):
        # Check input
        if any([x is None for x in [self.X, self.labels, self.d]]):
            raise ValueError('X, labels and subdim not set!')

        num_pts = self.X.shape[0]
        D = self.X.shape[1]
        subdim = self.d

        # Setup random state
        prng = RandomState()
        if self._SEED is not None:
            prng = RandomState(self._SEED)
            if self._verbose:
                print("Setting random seed to", self._SEED)

        if self._init_method == "PCA":
            if num_pts < self.d:
                raise ValueError('num_pts < subdim')
            if self.d > D:
                raise ValueError('subdim > inputdim')

            pca = PCA(n_components=subdim, whiten=False)
            pca.fit(self.X)
            L = pca.components_.T + 1E-6

        elif self._init_method == "LDA":
            if self.d > D:
                raise ValueError('subdim > inputdim')

            lda_obj = LDA.LDA(self.X, self.labels)
            lda_obj.compute(dim=self.d)
            L = lda_obj.getTransform()
            L = L * (1. / LA.norm(L, ord=1, axis=1)).reshape(-1, 1)
        elif self._init_method == "randbeng":
            # L = 1. * bound * prng.rand(D, self.d) - bound
            L = np.random.normal(0, np.sqrt(2) / np.sqrt(self.D + self.d), (self.D, self.d))
        elif self._init_method == "randbest":
            # Do some random generation of matrices pick the one with lowest # of constraints
            if self._verbose:
                print('Doing random pre-gen L')
            t0 = timeit.default_timer()
            best_L = prng.rand(D, self.d)
            L = best_L
            self.loss_fun(best_L)
            # nconsts = self._count_active_constraints()
            bound = np.sqrt(6. / (D + self.d))
            best_N_consts = 1E10
            for i in range(0, 10):
                L = 1. * bound * prng.rand(D, self.d) - bound
                # L = 1E-5*prng.rand(D,self.d)
                # L = L * (1./LA.norm(L,ord=1,axiss=1)).reshape(-1,1)
                self.loss_fun(L)
                consts = self._count_active_constraints()
                if consts < best_N_consts:
                    best_N_consts = consts
                    best_L = copy.copy(L)
            L = copy.copy(best_L)
            if self._verbose:
                print("Pre-gen of L done. Took:", "%3.3f" %
                      (timeit.default_timer() - t0), end=", ")
                print("# active const", best_N_consts, end=", ")

        elif self._init_method == "rand":
            # method_str = print('Doing random pre-gen Lapa')
            bound = np.sqrt(6. / (D + self.d))
            L = 1. * bound * prng.rand(D, self.d) - bound

        return L


    def _upsample_data(self):
        from imblearn.over_sampling import SMOTE
        if self._verbose:
            print("Checking if we need to uppsample. Number pts:", len(self.labels), end=" ")
        ratio = 0.75
        kmin = np.bincount(self.labels).min()
        if kmin > 5:
            smote = SMOTE(ratio=ratio, verbose=self._verbose, kind='regular')
        else:
            smote = SMOTE(ratio=ratio, k=kmin - 1, verbose=self._verbose, kind='regular')
        X, self.labels = smote.fit_sample(self.X, self.labels)
        self.labels = self.labels.astype(int)
        if self._verbose:
            print("Upsampled. Number pts:", len(self.labels))


    ############### CACHING MEACHNISM FUNCTIONS ###############
    # Stores point differences and matrix products for fast retrieval
    # i.e. no need to recompute matrix products.


    def _get_Xij_diff(self, i, j):
        if i > j:
            j, i = i, j
        if i not in self._Xij_vals:
            self._Xij_vals[i] = {}
        if j not in self._Xij_vals[i]:
            self._Xij_vals[i][j] = (self.X[i, :] - self.X[j, :]).reshape(1, -1)

    def _cpt_Xij_diff(self, i, j):
        if not isinstance(j, (np.ndarray, list)):
            self._get_Xij_diff(i, j)
            if i > j:
                return -self._Xij_vals[j][i]
            else:
                return self._Xij_vals[i][j]
        else:
            return self.X[i, :] - self.X[j, :]

    def _cpt_Xij_matprd(self, i, j):
        if i > j:
            j, i = i, j
        # We use * instead of dot since we have (N,1)x(1,N)
        if self._N_Xij_entries < self._Xij_max_entries:
            if i not in self._Xij_prds:
                self._Xij_prds[i] = {}
            if j not in self._Xij_prds[i]:  # (1 * D).T * 1 x D
                self._Xij_prds[i][j] = self._cpt_Xij_diff(i, j).T * self._cpt_Xij_diff(i, j)
                # print(np.allclose(self._Xij_prds[i][j],self._cpt_Xij_diff(i,j).T.dot(self._cpt_Xij_diff(i,j))))
                self._N_Xij_entries += 1
            return self._Xij_prds[i][j]
        else:
            if i in self._Xij_prds and j in self._Xij_prds[i]:
                return self._Xij_prds[i][j]
            else:
                return self._cpt_Xij_diff(i, j).T * self._cpt_Xij_diff(i, j)

    def _prune_Xij_prds(self):
        for idx in range(0, self.num_pts):
            for jdx in self.NNmat[idx, :]:
                for ldx in self._non_label_idxs[self.labels[idx]]:
                    if self._active_consts[idx][jdx][ldx] == 0:
                        if idx in self._Xij_prds and ldx in self._Xij_prds[idx]:
                            del self._Xij_prds[idx][ldx]
                            self._N_Xij_entries -= 1
        gc.collect()

    def _rm_Xij_dics(self):
        keys = [key for key in self._Xij_prds]
        for key in keys:
            del self._Xij_prds[key]

        keys = [key for key in self._Xij_vals]
        for key in keys:
            del self._Xij_vals[key]

        del self._Xij_prds
        del self._Xij_vals

        gc.collect()

    def _init_cache(self):
        self._Xij_max_entries = np.rint(self._maxcache / (8. * self.D**2))
        if self._verbose:
            print("Max cached matrix products:",
                  self._Xij_max_entries.astype(int))


    def _count_active_constraints(self):
        Nactive = 0
        for idx, labeli in enumerate(self.labels):
            for jdx in self.NNmat[idx, :]:
                for ldx in self._non_label_idxs[labeli]:
                    Nactive += self._active_consts[idx][jdx][ldx]
        return Nactive

    # HINGE LOSS FUNCTIONS

    def smooth_hinge_vec(self, x):
        val = np.zeros((np.size(x)))
        idx = np.flatnonzero((x > 0.) * (x < 1.))
        if len(idx) > 0:
            val[idx] = 0.5 * (x[idx]**2)

        idx = np.flatnonzero(x >= 1.)
        if len(idx) > 0:
            val[idx] = x[idx] - .5

        return val

    @staticmethod
    def df_smooth_hinge_vec(x):
        val = np.zeros((np.size(x)))
        idx = np.flatnonzero((x > 0.) * (x < 1.))
        if len(idx) > 0:
            val[idx] = x[idx]

        idx = np.flatnonzero(x >= 1.)
        if len(idx) > 0:
            val[idx] = 1.

        return val

    def smooth_hinge(self, x):
        if x <= 0.:
            return 0.
        elif x < 1.:
            return 0.5 * (x)**2
        else:
            return x - 0.5

    def df_smooth_hinge(self, x):
        if x <= 0.:
            return 0.
        elif x < 1.:
            return x
        else:
            return 1.

    def hinge_vec(self, x):
        val = np.zeros((np.size(x)))
        idx = np.flatnonzero(x > 0.)
        val[idx] = x[idx]
        return val

    def quadloss_vec(self, x, factor=1.):
        val = np.zeros((np.size(x)))
        try:
            idx = np.flatnonzero(x >= 0.)
            if len(idx) > 0:
                val[idx] = (0.5 / factor) * x[idx]**2
            return val
        except ValueError:
            print('val error', val)

    def df_quadloss_vec(self, x, factor=1.):
        val = np.zeros((np.size(x)))
        idx = np.flatnonzero(x >= 0.)
        if len(idx) > 0:
            val[idx] = (1. / factor) * x[idx]
        return val


    def _init_ew(self, Npts):
        self.ew = np.ones(Npts) / Npts

    # NEAREST NEIGHBOURS FUNCTIONS
    def cpt_knn_score(self, xTr, labels_tr, xTe, labels_te, k=3, score_fun=None):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(xTr, labels_tr)
        if score_fun == 'f1':
            labels_pred = knn.predict(xTe)
            return f1_score(labels_te, labels_pred)
        elif score_fun == 'r2':
            labels_pred = knn.predict(xTe)
            return r2_score(labels_te, labels_pred)
        else:
            return knn.score(xTe, labels_te)

    def _cpt_nn_mat(self, X):
        nns_dists = spatial.distance.pdist(X)
        nns_dists = spatial.distance.squareform(nns_dists, 'tomatrix')
        NNmat = np.argsort(nns_dists)
        return NNmat[:, 0:self.k + 1]


    def leave_one_out_average_score(self, X, labels, transform_x=True):
        if transform_x:
            X = self.transform(X)

        num_pts = X.shape[0]
        idxs = np.arange(0, num_pts)
        scores = 0.
        for te_idx in idxs:
            tr_idx = np.flatnonzero(te_idx != idxs)
            score = self.cpt_knn_score(X[tr_idx, :], labels[tr_idx], X[te_idx, :].reshape(
                1, -1), labels[te_idx].reshape(1, -1), k=self.k)
            scores += score
        return scores / num_pts, np.ones(num_pts)


    def compute_validation_error(self):
        Xtrnew = self.transform(self.X)
        Xtenew = self.transform(self.Xte)
        validation_score = self.cpt_knn_score(Xtrnew, self.labels, Xtenew, self.yTe, k=self.k)
        return validation_score

    def get_metric(self):
        return self.L.dot(self.L.T)

    def get_linear_transform(self):
        return self.L.T

    def transform(self, X=None):
        if X is None:
            return self.X.dot(self.L)  # NxD * Dxd -> Nxd
        else:
            return X.dot(self.L)  # NxD * Dxd -> Nxd

    # LOSS FUNCTIONS

    def mean_loss_(self, L):
        # Compute means
        xBar = np.zeros((len(self.unique_labels), self.D))
        for label, idx in enumerate(self.unique_labels):
            lidx = np.where(label == self.labels)[0]
            xBar[idx, :] = np.mean(self.X[lidx, :], axis=0)
        xBarSum = 0.
        # Compute distance between all means
        for idx in range(0, len(self.unique_labels)):
            for jdx in range(idx + 1, len(self.unique_labels)):
                Lij = (xBar[idx, :] - xBar[jdx, :]).dot(self.L)
                xBarSum += Lij.dot(Lij.T)
        return -xBarSum

    def d_mean_loss_(self, L):
        # Compute meanscov_diff
        xBar = np.zeros((len(self.unique_labels), self.D))
        for label, idx in enumerate(self.unique_labels):
            lidx = np.where(label == self.labels)[0]
            xBar[idx, :] = np.mean(self.X[lidx, :], axis=0)
        # xBarSum = 0.
        # Compute distance between all means
        d_xBar = np.zeros((self.L.shape))
        for idx in range(0, len(self.unique_labels)):
            for jdx in range(idx + 1, len(self.unique_labels)):
                Xm = (xBar[idx, :] - xBar[jdx, :]).reshape(-1, 1)
                d_xBar += 2. * (Xm.dot(Xm.T)).dot(L)

        return -d_xBar

    def mean_loss(self, L):
        # Compute means
        xBar = np.zeros((len(self.unique_labels), self.D))
        for label, idx in enumerate(self.unique_labels):
            xBar[idx, :] = np.mean(self.X[self.labelIdxs[label], :], axis=0)
        xBarSum = 0.

        # Compute distance between all means
        for idx, label in enumerate(self.unique_labels):
            for jdx in self._non_label_idxs[label]:
                Lij = (xBar[idx, :] - self.X[jdx, :]).dot(self.L)
                xBarSum += self.ratios[idx] * inner1d(Lij, Lij)   # Lij.T.dot(Lij)
        return xBarSum

    def d_mean_loss(self, L):
        # Compute means
        xBar = np.zeros((len(self.unique_labels), self.D))
        for label, idx in enumerate(self.unique_labels):
            xBar[idx, :] = np.mean(self.X[self.labelIdxs[label], :], axis=0)
        # xBarSum = 0.

        # Compute distance between all means
        d_xBar = np.zeros((self.L.shape))
        for idx, label in enumerate(self.unique_labels):
            for jdx in self._non_label_idxs[label]:
                Xm = (xBar[idx, :] - self.X[jdx, :]).reshape(-1, 1)
                d_xBar += self.ratios[idx] * 2. * (Xm.dot(Xm.T)).dot(L)
        return d_xBar

    def mean_center_loss(self, L):
        xBarSum = 0.
        # Compute intra class distance to all means
        for idx, label in enumerate(self.unique_labels):
            mLabel = np.mean(self.X[self.labelIdxs[label], :], axis=0)
            for jdx in self.labelIdxs[label]:
                Lij = (mLabel - self.X[jdx, :]).dot(L)
                xBarSum += self.ratios[idx] * inner1d(Lij, Lij)  # Lij.T.dot(Lij)
        return xBarSum

    def d_mean_center_loss(self, L):
        # Compute intra class distance to the centroid
        d_xBar = np.zeros((self.L.shape))
        for idx, label in enumerate(self.unique_labels):
            mLabel = np.mean(self.X[self.labelIdxs[label], :], axis=0)
            for jdx in self.labelIdxs[label]:
                Xm = (mLabel - self.X[jdx, :]).reshape(-1, 1)
                d_xBar += self.ratios[idx] * 2. * (Xm.dot(Xm.T)).dot(L)
        return d_xBar

    # Frobenius loss between covariance matrix of training and test data

    def cov_diff(self, L):
        xTrL = self.X.dot(L)
        xTeL = self.Xte.dot(L)
        cov_diff = np.cov(xTrL, rowvar=False) - np.cov(xTeL, rowvar=False)
        return np.trace(cov_diff)
        # return np.trace(cov_diff.T.dot(cov_diff))
        # return np.linalg.norm(cov_diff,ord='fro')**2

    def d_cov_diff(self, L, dL=None):
        if self.dCovProdSum is None:
            self.dCovProdSum = np.cov(self.X, rowvar=False) - \
                np.cov(self.Xte, rowvar=False)
        cvdf = np.cov(self.X, rowvar=False) - np.cov(self.Xte, rowvar=False)
        # print("Derivate mean",np.mean(3.92*cvdf.dot(L.dot(L.T)).dot(cvdf).dot(L)))
        return 2. * cvdf.dot(self.L)
        # return 3.92*cvdf.dot(L.dot(L.T)).dot(cvdf).dot(L)
        # return cvdf.dot(L.dot(L.T)).dot(cvdf).dot(L)
        # 3.92 should be 4. but due to numerical errors 4 over estimates the derivative.
        # return 2.*cvdf.dot(L)

    def cov_diff_NNs(self, L):
        xTrL = self.X.dot(L)
        xTeL = self.Xte.dot(L)

        covTr = np.zeros((self.d, self.d))
        covTe = np.zeros((self.d, self.d))

        nbrs = NearestNeighbors(n_neighbors=self.k + 1, algorithm='ball_tree').fit(xTrL)
        distances, NNs = nbrs.kneighbors(xTrL)
        for idx in range(NNs.shape[0]):
            covTr += np.cov(xTrL[NNs[idx, 1:self.k + 1]], rowvar=False)

        nbrs = NearestNeighbors(n_neighbors=self.k + 1, algorithm='ball_tree').fit(xTeL)
        distances, NNs = nbrs.kneighbors(xTeL)
        for idx in range(NNs.shape[0]):
            covTe += np.cov(xTeL[NNs[idx, 1:self.k + 1]], rowvar=False)

        cov_diff = covTr - covTe
        return np.trace(cov_diff.T * cov_diff)
        # return np.linalg.norm(cov_diff,ord='fro')**2

    def d_cov_diff_NNs(self, L, dL=None):
        # if True:#self.dCovProdSum is None:
        xTrL = self.X.dot(L)
        xTeL = self.Xte.dot(L)
        covTr = np.zeros((self.D, self.D))
        covTe = np.zeros((self.D, self.D))

        nbrs = NearestNeighbors(n_neighbors=self.k + 1, algorithm='ball_tree').fit(xTrL)
        distances, NNs = nbrs.kneighbors(xTrL)
        for idx in range(NNs.shape[0]):
            covTr += np.cov(self.X[NNs[idx, 1:self.k + 1]], rowvar=False)

        nbrs = NearestNeighbors(n_neighbors=self.k + 1, algorithm='ball_tree').fit(xTeL)
        distances, NNs = nbrs.kneighbors(xTeL)
        for idx in range(NNs.shape[0]):
            covTe += np.cov(self.Xte[NNs[idx, 1:self.k + 1]], rowvar=False)

        cvdf = covTr - covTe

        return 3.92 * cvdf.dot(L.dot(L.T)).dot(cvdf).dot(L)

    def kl_divergence(self, L):
        xTrL = self.X.dot(L)
        xTeL = self.Xte.dot(L)
        kl_divergence_sum = 0.
        for idx in range(self.X.shape[0]):
            p = self.krnsum(xTrL, xTrL[idx, :])
            q = self.krnsum(xTeL, xTrL[idx, :])
            # print(p,q)
            kl_divergence_sum += p * np.log(1E-30 + (p / (q + 1E-12)))
        for jdx in range(self.Xte.shape[0]):
            p = self.krnsum(xTrL, xTeL[jdx, :])
            q = self.krnsum(xTeL, xTeL[jdx, :])
            # print(p,q)
            kl_divergence_sum += p * np.log(1E-30 + (p / (q + 1E-12)))
        # print(kl_divergence_sum)
        return kl_divergence_sum

    def d_kl_divergence(self, L):
        # Compute p and q
        return 0.

    def krnsum(self, X, x):
        xbar = x - X
        sgm = 1.E3
        dim = 3.
        xx = -(0.5 / sgm) * np.sum(xbar * xbar, axis=1)
        nrmTerm = (3.14 * sgm)**(dim / 2.)
        # detL = np.linalg.det(self.L.T.dot(self.L))**2
        # print(np.prod( np.exp( -0.5 * xx ), axis=1))

        return (1. / X.shape[0]) * np.sum(np.exp(xx) / nrmTerm)

    # METRIC TRANSFORM PENALIZATION FUNCTIONS

    def no_pen(self, *args):
        return 0.

    def d_no_pen(self, *args):
        return 0.

    # Log determinant penalization

    def logdet_pen(self, *args):
        (sign, logdet) = np.linalg.slogdet(self.L.T.dot(self.L))
        return 0.5 * logdet

    def d_logdet_pen(self, *args):
        return np.linalg.pinv(self.L).T

    # Entropy
    def entropy_pen(self, *args):
        lNorm = LA.norm(self.L, ord=2, axis=1)
        p = lNorm / np.sum(lNorm)
        return -np.sum(p * np.log(p + 1E-12))

    def d_entropy_pen(self, *args):
        pass
        # lNorm = LA.norm(self.LL, ord=2, axis=1)
        # lNormSum = np.sum(lNorm)
        # p = lNorm / lNormSum
        # normdL = self.L / lNorm.reshape(-1, 1)

    # Trace Norm

    def trace_pen(self, *args):
        return np.trace(self.L.T.dot(self.L))

    def d_trace_pen(self, *args):
        return 2. * self.L

    # Trace sqrd Norm
    def trace_sqrd_pen(self, *args):
        print(np.trace(self.L.T.dot(self.L))**2)
        return np.trace(self.L.T.dot(self.L))**2

    def d_trace_sqrd_pen(self, *args):
        return 4. * self.L * np.trace(self.L.T.dot(self.L))

    # Column Norm penalization
    # l1
    def l1_pen(self, *args):
        # print(np.sum(np.linalg.norm(self.L,ord=1,axis=0)))
        return np.sum(np.linalg.norm(self.L, ord=1, axis=0))

    def d_l1_pen(self, *args):
        Lsign = (self.L > 1E-6).astype(int) - (self.L < -1E-6).astype(int)
        dLplus = (args['dL'] > self.lmbda[0]).astype(int) * (Lsign == 0)
        dLneg = -(args['dL'] < self.lmbda[0]).astype(int) * (Lsign == 0)
        # print("Zeroelementsum",np.sum((Lsign==0)))
        # print(( Lsign + dLplus + dLneg ).astype(float))
        return (Lsign + dLplus + dLneg).astype(float)

    # l2

    def l2_pen(self, *args):
        # M = np.sum(np.sqrt(np.diag(L.T.dot(L))))
        # axis=0 -> column space, axis=1 -> row space
        # print "Error Penal Ratio",Ertot/(0.5*np.sum(np.linalg.norm(L,ord=2,axis=0)))
        return np.sum(np.linalg.norm(self.L, ord=2, axis=0))

    def d_l2_pen(self, *args):
        vnorm = np.linalg.norm(self.L, ord=2, axis=0).reshape(1, -1)
        # print("normshape",vnorm.shape,"self.L.shape",self.L.shape)
        return (self.L / vnorm)

    def l2sqrd_pen(self, *args):
        return 0.5 * np.sum(self.L * self.L)

    def d_l2sqrd_pen(self, *args):
        return self.L

    # Frobenius Norm

    def frob_pen(self, L, *args):
        return np.linalg.norm(L, ord='fro')

    def d_frob_pen(self, L, *args):
        return (1. / np.linalg.norm(L, ord='fro')) * L

    def l1l2_pen(self, L, *args):
        return np.sum(np.linalg.norm(L, ord=2, axis=1))

    def d_l1l2_pen(self, L, dL=None, *args):
        return L / (1.E-12 + np.linalg.norm(L, ord=2, axis=1).reshape(-1, 1))

    # # Elastic OB

    def elastic_pen(self, L, *args):
        # return ( 0.25 * self.l1_pen(L) + 0.75 * self.l2_pen(L) )
        return self.l2_pen(L)
        # colNorml1 = np.linalg.norm(L,ord=1,axis=0)
        # colNorml2 = np.linalg.norm(L,ord=2,axis=0)
        # return ( 0.75 * np.sum(colNorml2) + 0.25 * np.linalg.norm(colNorml2,ord=2,axis=0) )
        # return ( np.linalg.norm(colNorml2,ord=2,axis=0) )

    def d_elastic_pen(self, L, dL, *args):
        # return ( 0.25 * self.d_l1_pen(L,dL) + 0.75 * self.d_l2_pen(L) )
        return self.d_l2_pen(L)
        # Lsign = (L>1E-6).astype(int) - (L<-1E-6).astype(int)
        # dLplus = (dL > self.lmbda).astype(int) * (Lsign==0)
        # dLneg = -(dL < self.lmbda).astype(int) * (Lsign==0)
        # vnorm = np.linalg.norm(L,ord=2,axis=0).reshape(1,-1)
        # colNorml2 = np.linalg.norm(L,ord=2,axis=0)
        # colSumNorm = np.linalg.norm(colNorml2,ord=2,axis=0)
        # return ( 0.75 * ( Lsign + dLplus + dLneg ).astype(float) + 0.25 * L/colSumNorm )
        # return ( 0.75 * L/colNorml2 + 0.25 * L/colSumNorm )
        # return ( L/colSumNorm )

    def proj_pen(self, L):
        M = L.dot(L.T)
        loss = np.linalg.norm(np.eye(M.shape[0]) - M, ord='fro')
        # print(" ")
        # print(loss)

        return loss

    def d_proj_pen(self, L):
        M = L.dot(L.T)
        # A = np.eye(M.shape[0]) - M * M.T
        # return -(1. / np.linalg.norm(A, ord='fro')) * ( M.dot(L)
        T_0 = np.eye(M.shape[0]) - M
        t_1 = np.linalg.norm(T_0, ord='fro')
        gradient = -((2 / t_1) * np.dot(T_0, L))
        return gradient

    def sigmoid(self, z):
        y = 1. / (1. + np.exp(-z))
        return y

    # def logit_loss(self, L):
    #     X_trans = self.X.dot(L)  # Nxd
    #     sig_val = self.sigmoid(X_trans * self.theta)
    #     loss = (1 / self.Npts) * np.sum(-self.labels * np.log(sig_val)
    #             - (1 - self.labels) * np.log(1 - sig_val))
    #     return loss
    #
    # def d_theta_logit_loss(self, L):
    #     X_trans = self.X.dot(L)
    #     sig_val = self.sigmoid(X_trans * self.theta)
    #     gradient = (1 / self.Npts) * \
    #         (my_sigmoid(self.theta * self.X) - self.labels) * X_trans
    #
    # def dL_logit_loss(self, L):
    #     X_trans = self.X.dot(L)
    #     sig_val = self.sigmoid(X_trans * self.theta)
    #     gradient = (1 / self.Npts) * (my_sigmoid(self.theta * self.X)
    #                                   - self.labels) * X_trans * self.theta
