"""Implementation of the Large Margin Nearest Neighbor algorithm."""

import copy
import timeit

# Numpy
import numpy as np
# settings = np.seterr(invalid='raise')

# Sklearn
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

# Mine
from metricbase import MetricBase


class LMNN(MetricBase):
    def __init__(self, k=3, dimension=0, max_iter=1000, regularization=0.5, verbose=False, balance=None,
                 init_method='rand', L=None, stepsize=5.E-5, stepsize_min=1E-8, maxcache=3.E9,
                 NNmat=None, seed=None, convergence_tol=1E-3, use_validation=False, nn_weights=None,
                 diag=False, **kwargs):

        super(LMNN, self).__init__(maxcache, seed, verbose, **kwargs)

        self.k = k
        self.reg_e2 = regularization  # Weight for second error term
        self.reg_e1 = 1. - regularization
        self.d = dimension

        if nn_weights is None:
            self._nn_weights = np.ones(self.k)
        else:
            self._nn_weights = nn_weights

        self._stepsize = stepsize
        self._stepsize_min = stepsize_min

        self._balance = balance
        self._init_method = init_method
        self._diag = diag

        self.NNmat = NNmat

        # Convergence criterias
        self.convergence_tol = convergence_tol
        self.max_iter = max_iter
        self.validation = use_validation
        self.NNmat = NNmat

        # Assign inputs
        self.num_pts = 0
        self.unique_labels = None
        self.dL1 = None
        self.active = None
        self.trIdx = None
        self.teIdx = None

        self._label_idxs = {}
        self._non_label_idxs = {}
        self._active_const = {}
        # self._deactivated = {}
        # self._deactivated = {}
        self._activate = []
        self._deactivate = []


    def _process_inputs(self, X, labels, Xte=None, yTe=None):
        # ASSIGN DATA VARIABLES, STATS AND CONTAINERS
        self.X = X
        self.num_pts = self.X.shape[0]
        self.Xte = Xte
        self.yTe = yTe
        self.D = self.X.shape[1]
        self.labels = labels.astype(int)
        self.unique_labels, label_inds = np.unique(self.labels, return_inverse=True)
        self._active_consts = [0] * self.num_pts

        self._init_cache()

        # CHECK THAT DATA AND PARAMS ARE CORRECT
        assert_fail_str = 'The number of labels {:d} do not equal the number of data points {:d}'
        assert len(self.labels) == self.num_pts, (assert_fail_str.format(
            len(labels), self.num_pts))

        # Enough instances in each class to do k-NN
        NminClass = np.bincount(self.labels).min()
        assert_fail_str = 'Not enough class labels for specified k (smallest class has {:d} instances.)'
        assert NminClass > self.k, (assert_fail_str.format(NminClass))

        if self.validation:
            if self.Xte is None:
                raise ValueError('Using validation Xte needs to be set.')
            if self.yTe is None:
                raise ValueError('Using validation yTe needs to be set.')

        # Set projection dim
        if self.d == 0:
            self.d = self.D
        if self.d > self.D:
            print("WARNING: Enlarging the projection output dimension!", self.D, self.d)

        # Set data imbalance strategy
        self.ratios = np.ones(len(self.unique_labels))
        if self._balance == 'ratios':
            self.ratios = self.num_pts / np.bincount(self.labels)
        elif self._balance == 'upsample':
            self._upsample_data()

        # LABEL INDICES LISTS
        for idx, label in enumerate(self.unique_labels):
            self._label_idxs[label] = np.flatnonzero(self.labels == label)
            self._non_label_idxs[label] = np.flatnonzero(self.labels != label)

        self.NNs = len(self._nn_weights)
        if self.NNmat is not None:
            if self.NNmat.shape[0] != self.num_pts and self.NNmat.shape[1] != self.NNs:
                raise RuntimeError(
                    "Shape of NNmat should be N-by-k but is" + str(self.NNmat.shape))
        else:
            self.NNmat = np.zeros((self.num_pts, self.NNs), dtype=int)
            for clsIdxs in self._label_idxs.values():
                nbrs = NearestNeighbors(n_neighbors=self.NNs + 1,
                                        algorithm='ball_tree').fit(self.X[clsIdxs, :])
                distances, NNs = nbrs.kneighbors(self.X[clsIdxs, :])
                for idx, val in enumerate(clsIdxs):
                    self.NNmat[val, :] = clsIdxs[NNs[idx, 1:self.NNs + 1]]

        # IMPOSTOR MATRIX Set init value that all ldxs are inactive
        for idx, label in enumerate(self.labels):
            self._active_consts[idx] = {}
            for jdx in self.NNmat[idx, :]:
                self._active_consts[idx][jdx] = {}
                for ldx in self._non_label_idxs[label]:
                    self._active_consts[idx][jdx][ldx] = 0

        # Precompute part of the first error term in the derivative since it is the same all the time
        self.dL = np.zeros((self.D, self.D))
        for idx in range(0, self.num_pts):
            for nn_idx, jdx in enumerate(self.NNmat[idx, :]):
                self.dL += self.reg_e1 * self._nn_weights[nn_idx] * \
                    self.ratios[self.labels[idx]] * self._cpt_Xij_matprd(idx, jdx)

        self.dL1 = copy.deepcopy(self.dL)

    @staticmethod
    def cross_val_fun(X, y, train_index, test_index, factor, **args):
        x_tr, x_te = X[train_index], X[test_index]
        y_tr, y_te = y[train_index], y[test_index]

        params = copy.deepcopy(args)
        params['seed'] = factor

        # Preprocess
        if params['standardize'] == 1:
            scaler = preprocessing.StandardScaler().fit(x_tr)
            x_tr = scaler.transform(x_tr)
            x_te = scaler.transform(x_te)

        # Fit
        metric = LMNN(**params)
        metric.fit(x_tr, y_tr, x_te, y_te)
        x_tr = metric.transform(x_tr)
        x_te = metric.transform(x_te)

        # Evaluate
        score = metric.cpt_knn_score(x_tr, y_tr, x_te, y_te, k=params['k'])
        f1score = 0.
        if np.size(np.unique(y_te)) == 2:
            f1score = metric.cpt_knn_score(
                x_tr, y_tr, x_te, y_te, k=params['k'], score_fun='f1')

        return {'score': score, 'f1score': f1score, 'L': metric.L, 'loss': metric.loss_values[-1]}

    def _calibrate_stepsize(self):
        """Set step size to max without overshooting loss decrease."""

        loss = self.loss_fun(self.L, chg_active=True)
        self._update_deriv_loss()
        G_pen = np.zeros(self.L.shape)
        for idx, name in enumerate(self._penfun_list):
            G_pen += self._penfun_weights[name] * self._df_penfuns[name](self.L)

        G = 2 * self.dL.dot(self.L) + G_pen

        # Do for as long as loss keeps decreasing
        for i in range(0, 1000):
            L_new = self.L - self._stepsize * G
            delta_loss = loss - self.loss_fun(L=L_new, chg_active=False)

            if delta_loss > 0:
                self._stepsize *= 1.1
            else:
                self._stepsize /= 1.1
                break

    def fit(self, X, labels, Xte=None, yTe=None):
        # Set and precompute constant terms of gradient
        self._process_inputs(X, labels, Xte=Xte, yTe=yTe)
        self.L = self._init_trans_mat()

        self._calibrate_stepsize()
        self.loss_values = [self.loss_fun(self.L, chg_active=True)]

        self.Ls = [self.L.copy()]
        self.validation_scores = [0.]
        ticker = 0

        if self._verbose:
            print("Done preprocessing starting GD.")

        start = timeit.default_timer()

        for ii in range(1, self.max_iter + 2):

            # Gradient descent update is L = L - learn_rate  * dL
            # Dimensions: (dxD) = (dxD) - learn_rate * (dxD)
            # Compute gradient
            self._update_deriv_loss()
            G = self.dL.dot(self.L)

            # Add penalty derivatives
            G_pen = np.zeros(self.L.shape)
            for idx, name in enumerate(self._penfun_list):
                G_pen += self._penfun_weights[name] * self._df_penfuns[name](self.L)

            # Update
            if self._diag:
                M = self.L.dot(self.L.T)
                M = M - self._stepsize * (G + G_pen)
                # w, v = LA.eigh(M)
                # Extract diagnonal of L and turn into L format
                # self.L = np.sqrt(np.diag(w).clip(min=0))
                self.L = np.sqrt(np.diag(np.diag(M)).clip(min=0))
            else:
                self.L = self.L - self._stepsize * (2 * G + G_pen)
                # M = self.L.dot(self.L.T)
                # w, v = LA.eigh(M)
                # self.L = v.dot(np.diag(np.sqrt(w.clip(min=0))))

            # Do pruning of matrix stored products every x iteration
            if ii % 11 == 0:
                self._prune_Xij_prds()

            # self.L = self.L * (self.L > 1E-20 )
            # if ii % 3 == 0:
            #     self.loss_values.append(self.loss_fun(self.L, chg_active=True))
            # else:
            loss = self.loss_fun(self.L, chg_active=True)
            self.loss_values.append(loss)
            self.Ls.append(self.L)

            if self.validation:
                self.validation_scores.append(self.compute_validation_error())
            else:
                self.validation_scores.append(-1.)

            if self.loss_values[ii] > self.loss_values[ii - 1]:
                self._stepsize *= 0.5
            else:
                self._stepsize *= 1.05

            # CHECK TERMINATION CONDITIONS
            terminate_bool, termination_str = self._test_for_termination(ii)
            if terminate_bool:
                if self._verbose:
                    print(termination_str)
                break
            #
            # Stepsize reset if steps are reduced too fast
            if self._stepsize < self._stepsize_min and ticker < 10:
                ticker += 1
                self._stepsize = 1E-6

            # PRINT ITERATION INFORMATION
            if self._verbose and ii % 1 == 0:
                # if iter%10 == 0 and iter!=0:
                edlstr = "\r"

                if ii % 1 == 0:
                    Nacs = self._count_active_constraints()
                    loss_diff = self.loss_values[ii] - self.loss_values[ii - 1]
                    loss_diff_prcnt = 100 * (loss_diff / self.loss_values[ii - 1])
                    nonZeroCols = np.sum(np.linalg.norm(self.L, ord=2, axis=1) > 1.E-3)
                    running_time = timeit.default_timer() - start
                    time_per_iter = running_time / float(ii + 1)

                    info_str = ("\033[92mIter: {a:d}\033[0m # acs: {b:d}, Step size: {c:.3g}, "
                                "Er chng: {d:.3g}, Er chng pcnt: {e:3.3g}, Er: {f:3.3g}, "
                                "L-cols: {g:d}/{h:d}, Time: {i:3.3g} per iter {j:3.3g}     ")
                    print(info_str.format(a=ii, b=Nacs, c=self._stepsize, d=loss_diff,
                                          e=loss_diff_prcnt, f=self.loss_values[ii], g=nonZeroCols, h=self.D, i=running_time,
                                          j=time_per_iter), end=edlstr)

                if ii % 100 == 0 and ii != 0:
                    print('')

        # PRINT END INFORMATION
        if self._verbose:

            print(' ')
            # print("Validation Accuracy:",self.computeValidationError(self.L))
            lou_kNN, w = self.leave_one_out_average_score(self.X, self.labels, False)
            lou_LMCA, w = self.leave_one_out_average_score(self.X, self.labels)
            print(("LOU Error: kNN {:.3f}, LMCA: {:.3f}").format(
                lou_kNN, lou_LMCA))

            running_time = timeit.default_timer() - start
            time_per_iter = running_time / float(ii + 1)
            info_str = "Converged in {:d} iterations and time {:.3f} per iter {:.3f} and Error: {:.3f}"
            print(info_str.format(ii, running_time,
                                  time_per_iter, self.loss_values[ii]))

            nonZeroCols = np.sum(np.linalg.norm(self.L.T, ord=2, axis=0) > 1.E-3)
            print("L Nonzero cols {:d}/{:d}".format(nonZeroCols, self.D))
            print(' ')

        # Clean up memory at once
        self._rm_Xij_dics()

    def _test_for_termination(self, ii=3):
        """Test for min step size, convergence of the loss function, max iterations."""
        # Convergence tolerance
        if ii > 10 and np.all(np.abs((100 * (self.loss_values[-1] - np.array(self.loss_values[-4:-1])) / self.loss_values[-1])) <= self.convergence_tol):
            # self._stepsize *= 1E-3
            if ii > 10 and np.all(np.abs((100 * (self.loss_values[-1] - np.array(self.loss_values[-10:-1])) / self.loss_values[-1])) <= self.convergence_tol):
                termination_str = "\n\033[92mStopping\033[0m since loss change prcnt is below threshold value."
                return True, termination_str

        # Step size
        if self._stepsize < self._stepsize_min:
            termination_str = "\n\033[92mStopping\033[0m since step size smaller than min step size."
            return True, termination_str

        # Max iterations
        if ii > self.max_iter:
            termination_str = "\n\033[92mStopping\033[0m since maximum number of iterations reached before convergence."
            return True, termination_str

        # Validation set goes negative
        if self.validation and ii > 100:
            if np.all(self.validation_scores[-5:] - np.array(self.validation_scores[-6:-1]) <= 0.):
                termination_str = "\n\033[92mStopping\033[0m since validation scores have a negative derivative for 10 steps."
                return True, termination_str

        # Passed all, so continue
        return False, "Passed termination tests."

    def loss_fun(self, L, chg_active=True):
        er1 = 0.
        er2 = 0.

        # For each point...
        for idx, label in enumerate(self.labels):

            # ...and its k nearest neighbors
            for nn_idx, jdx in enumerate(self.NNmat[idx, :]):

                if self._nn_weights[nn_idx] < 1.E-12:
                    continue

                # LOSS TERM 1
                Lij = self._cpt_Xij_diff(idx, jdx).dot(L)
                xLijNorm = Lij.dot(Lij.T)
                er1 += self._nn_weights[nn_idx] * self.ratios[label] * xLijNorm

                # LOSS TERM 2
                Lil = self._cpt_Xij_diff(idx, self._non_label_idxs[label]).dot(L)
                impostor_dists = np.ravel(xLijNorm - np.einsum('ij, ji->i', Lil, Lil.T) + 1.)
                er2 += self._nn_weights[nn_idx] * self.ratios[label] * \
                    np.sum(self.hinge_vec(impostor_dists))

                # Check which constraints changed
                if chg_active:
                    # to active
                    activated_ldxs = np.flatnonzero(impostor_dists > 0.)
                    activated = [(idx, jdx, self._non_label_idxs[label][ldx], nn_idx)
                                 for ldx in activated_ldxs if
                                 self._active_consts[idx][jdx][self._non_label_idxs[label][ldx]] == 0]
                    self._activate.extend(activated)

                    # to inactive
                    deactivated_ldxs = np.flatnonzero(impostor_dists < 0.)
                    deactivated = [(idx, jdx, self._non_label_idxs[label][ldx], nn_idx)
                                   for ldx in deactivated_ldxs if self._active_consts[idx][jdx][self._non_label_idxs[label][ldx]] == 1]
                    self._deactivate.extend(deactivated)

                    # Toggle changed matrix
                    for i, j, l, nn_idx_chg in activated:
                        self._active_consts[i][j][l] = 1
                    for i, j, l, nn_idx_chg in deactivated:
                        self._active_consts[i][j][l] = 0

        er = self.reg_e1 * er1 + self.reg_e2 * er2

        for idx, name in enumerate(self._penfun_list):
            er += self._penfun_weights[name] * self._penfuns[name](L)

        return np.asscalar(er)

    def _update_deriv_loss(self):
        for idx, jdx, ldx, nn_idx in self._deactivate:
            if self._nn_weights[nn_idx] > 1.E-12:
                self.dL -= self._nn_weights[nn_idx] * self.ratios[self.labels[idx]] * self.reg_e2 * \
                    self._impostor_prd_diff(idx, jdx, ldx)

        for idx, jdx, ldx, nn_idx in self._activate:
            if self._nn_weights[nn_idx] > 1.E-12:
                self.dL += self._nn_weights[nn_idx] * self.ratios[self.labels[idx]] * self.reg_e2 * \
                    self._impostor_prd_diff(idx, jdx, ldx)

        self._deactivate[:] = []
        self._activate[:] = []

        if self._count_active_constraints() == 0:
            self.dL = copy.deepcopy(self.dL1)


    def _impostor_prd_diff(self, idx, jdx, ldx):
        return self._cpt_Xij_matprd(idx, jdx) - self._cpt_Xij_matprd(idx, ldx)
