import copy
import timeit

# Numpy
import numpy as np

# Sklearn
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

# Mine
from metricbase import MetricBase


class LMCA(MetricBase):
    def __init__(self, k=3, dimension=0, max_iter=1000, regularization=0.5, verbose=False, balance=None, 
                 init_method='rand', L=None, stepsize=5.E-5, stepsize_min=1E-20, maxcache=3.E9, 
                 NNmat=None, seed=None, convergence_tol=1E-1, use_validation=False, nn_weights=None,
                 **kwargs):

        super(LMCA, self).__init__(maxcache, seed, verbose, **kwargs)

        self.k = k
        self.reg = regularization  # Weight for second error term
        self.d = dimension

        if nn_weights is None:
            self._nn_weights = np.ones(self.k)
        else:
            self._nn_weights = nn_weights

        self._stepsize = stepsize
        self._stepsize_min = stepsize_min

        self._balance = balance
        self._init_method = init_method

        self.NNmat = NNmat

        # Convergence criterias
        self.convergence_tol = convergence_tol
        self.max_iter = max_iter
        self.validation = use_validation

        self.num_pts = 0
        self.unique_labels = None
        self.dL1 = None
        self._label_idxs = []
        self._non_label_idxs = []  # List of list of indices not belonging to one data class
        self._active_consts = {}

        self.trIdx = None
        self.teIdx = None

        # Penalization Functions are assigned in the base

    def _process_inputs(self, X, labels, Xte=None, yTe=None):
        # ASSIGN DATA VARIABLES AND CONTAINERS
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

        # ENOUGH INSTANCES IN EACH CLASS TO DO K-NN
        NminClass = np.bincount(self.labels).min()
        assert_fail_str = 'Not enough class labels for specified k (smallest class has {:d} instances.)'
        assert NminClass > self.k, (assert_fail_str.format(NminClass))

        if self.validation:
            if self.Xte is None:
                raise ValueError('Using validation Xte needs to be set.')
            if self.yTe is None:
                raise ValueError('Using validation yTe needs to be set.')

        # SET PROJECTION DIM
        if self.d == 0:
            self.d = self.D
        if self.d > self.D:
            print("WARNING: Enlarging the projection output dimension!", self.D, self.d)

        # SET DATA IMBALANCE STRATEGY
        self.ratios = np.ones(len(self.unique_labels))
        if self._balance == 'ratios':
            self.ratios = self.num_pts / np.bincount(self.labels)
        elif self._balance == 'upsample':
            self._upsample_data()

        # LABEL INDICES LISTS
        for idx, label in enumerate(self.unique_labels):
            self._label_idxs.append(np.flatnonzero(self.labels == label))
            self._non_label_idxs.append(np.flatnonzero(self.labels != label))

        # COMPUTE NN MATRIX
        self.kNNs = len(self._nn_weights)
        if self.NNmat is not None:
            if self.NNmat.shape[0] != self.num_pts and self.NNmat.shape[1] != self.k:
                raise RuntimeError(
                    "Shape of NNmat should be N-by-k but is" + str(self.NNmat.shape))
        else:
            # No NN Mat so set to NN
            self.NNmat = np.zeros((self.num_pts, self.kNNs), dtype=int)
            for cls_idxs in self._label_idxs:
                nbrs = NearestNeighbors(n_neighbors=self.kNNs + 1,
                                        algorithm='ball_tree').fit(self.X[cls_idxs, :])
                distances, NNs = nbrs.kneighbors(self.X[cls_idxs, :])
                self.NNmat[cls_idxs, :] = cls_idxs[NNs[:, 1:self.kNNs + 1]]

        # IMPOSTOR MATRIX Set init value that all ldxs are inactive
        for idx, label in enumerate(self.labels):
            self._active_consts[idx] = {}
            for jdx in self.NNmat[idx, :]:
                self._active_consts[idx][jdx] = {}
                for ldx in self._non_label_idxs[label]:
                    self._active_consts[idx][jdx][ldx] = 0

        # PRECOMPUTE PART OF THE FIRST ERROR TERM IN THE DERIVATIVE (since it is the same all the time)
        self.dL1 = np.zeros((self.D, self.D))
        for idx in range(0, self.num_pts):
            for nn_idx, jdx in enumerate(self.NNmat[idx, :]):
                    self.dL1 += self._nn_weights[nn_idx] * self.ratios[self.labels[idx]] * self._cpt_Xij_matprd(idx, jdx)

    def _cpt_avg_active_constraints(self):
        Nactive = 0
        Nacforpt = 0
        NptsWithConstraints = 0
        for idx, labeli in enumerate(self.labels):
            Nacforpt = 0
            for jdx in self.NNmat[idx, :]:
                for ldx in self._non_label_idxs[labeli]:
                    Nacforpt += np.sum(self._active_consts[idx][jdx][ldx])
            if Nacforpt > 0:
                Nactive += Nacforpt
                NptsWithConstraints += 1
                self._active_consts[idx] = Nacforpt
            else:
                self._active_consts[idx] = 0
        if NptsWithConstraints > 0:
            return float(Nactive) / float(NptsWithConstraints)
        else:
            return 0.

    def _get_top_active_constraints(self, Npts=10):
        # srtdCnstrs = sorted(range(len(self._active_consts)),key=lambda x:self._active_consts[x],reverse=True)
        # return srtdCnstrs[0:Npts]
        srtdCnstrs = [(i[0], i[1]) for i in sorted(
            enumerate(self._active_consts), key=lambda x:x[1])]
        idxs = []
        for i in srtdCnstrs:
            if i[1] > 0:
                idxs.append(i[0])
            if len(idxs) > Npts:
                return idxs
        return idxs

    def calibrate_stepsize(self):
        '''Sets step size to max with out overshooting loss decrease.'''
        dL = self.d_loss_fun()
        loss = self.loss_fun(L=self.L)

        # Do for as long as loss keeps decreasing
        for i in range(0, 1000):
            L_new = self.L - self._stepsize * dL
            delta_loss = loss - self.loss_fun(L=L_new)

            if delta_loss > 0:
                self._stepsize *= 1.1
            else:
                self._stepsize /= 1.1   # Retrace last increase
                break

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
        metric = LMCA(**params)        
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

    def setNNwithTransform(self):
        # Transform the data to compute a better approximation of the NNs
        Xtrans = self.transform()
        # Reset the NN matrix
        self.NNmat = -np.ones((self.num_pts, self.k))
        # Compute NNs per class
        for label in self.unique_labels:
            clsIdxs = np.where(self.labels == label)[0]
            nbrs = NearestNeighbors(
                n_neighbors=self.k + 1, algorithm='ball_tree').fit(Xtrans[clsIdxs, :])
            distances, NNs = nbrs.kneighbors(Xtrans[clsIdxs, :])
            for idx, val in enumerate(clsIdxs):
                nnList = clsIdxs[NNs[idx, 1:self.k + 1]].tolist()
                for i in range(0, self.k):
                    # Only switch if neighbor is not defined
                    if self.NNmat[val, i] == -1:
                        while nnList:
                            nn = nnList.pop(0)
                            # Check that neighbor does not already exist as a neighbor
                            if not np.any(nn == self.NNmat[val, i]):
                                break
                        self.NNmat[val, i] = nn
        self.NNmat = self.NNmat.astype(int)

    def fit(self, X, labels, Xte=None, yTe=None):
        # Precompute neighbors, constant terms of gradient, etc.
        self._process_inputs(X, labels, Xte=Xte, yTe=yTe)

        # Generate a good starting point for the L matrix
        self.L = self._init_trans_mat()
        # Loss. First run sets active false to find all active and inactive constraints
        self.loss_values = [self.loss_fun(self.L, active=False, chg_active=True)]
        self.Ls = [self.L]
        self.validation_scores = [0.]
        ticker = 0

        self._prune_Xij_prds()
        self.calibrate_stepsize()

        if self._verbose:
            print("Done preprocessing starting GD.")
        start = timeit.default_timer()

        for ii in range(1, self.max_iter + 2):

            # start2 = timeit.default_timer()
            # GRADIENT DESCENT UPDATE
            # Update is L = L - learn_rate  * dL
            # Dimensions: (dxD) = (dxD) - learn_rate * (dxD)
            self.L = self.L - self._stepsize * self.d_loss_fun()
            # print("Time",timeit.default_timer() - start2)
            # print(' ')
            self.Ls.append(self.L)
            # if ii % 10 == 0:
            #     self.loss_values.append(self.loss_fun(
            #         self.L, active=False, chg_active=True))
            # else:
            #     self.loss_values.append(self.loss_fun(
            #         self.L, active=False, chg_active=False))
                
            self.loss_values.append(self.loss_fun(
                    self.L, active=False, chg_active=True))

            if self.validation:
                self.validation_scores.append(self.compute_validation_error())
            else:
                self.validation_scores.append(-1.)

            if self.loss_values[ii] > self.loss_values[ii - 1]:
                self._stepsize *= 0.5
            else:
                self._stepsize *= 1.01

            # CHECK TERMINATION CONDITIONS
            terminate_bool, termination_str = self._test_for_termination(ii)
            if terminate_bool:
                if self._verbose:
                    print(termination_str)
                break

            # Stepsize reset if steps are reduced too fast
            if self._stepsize < self._stepsize_min and ticker < 10:
                ticker += 1
                self._stepsize = 1E-6

            # Do pruning of matrix stored products every x iteration
            if ii % 5 == 0:
                self._prune_Xij_prds()

            # PRINT ITERATION INFORMATION
            if self._verbose and ii % 1 == 0:
                # if iter%10 == 0 and iter!=0:
                edlstr = "\r"

                if ii % 1 == 0:
                    Nacs = self._count_active_constraints()
                    loss_diff = self.loss_values[ii] - self.loss_values[ii - 1]
                    loss_diff_prcnt = 100 * (loss_diff / self.loss_values[ii - 1])
                    nonZeroCols = np.sum(np.linalg.norm(
                        self.L, ord=2, axis=1) > 1.E-3)
                    running_time = timeit.default_timer() - start
                    time_per_iter = running_time / float(ii + 1)

                    info_str = ("\033[92mIter: {a:d}\033[0m # acs: {b:d}, Step size: {c:.3g}, "
                                "Er chng: {d:.3g}, Er chng pcnt: {e:3.3g}, Er: {f:3.3g}, ValEr: {f2:3.3g}, "
                                "L-cols: {g:d}/{h:d}, Time: {i:3.3g} per iter {j:3.3g}     ")
                    print(info_str.format(a=ii, b=Nacs, c=self._stepsize, d=loss_diff,
                                          e=loss_diff_prcnt, f=self.loss_values[ii], f2=self.validation_scores[
                                              ii], g=nonZeroCols, h=self.D, i=running_time,
                                          j=time_per_iter), end=edlstr)

                if ii % 100 == 0 and ii != 0:
                    print('')

        # Go back to when the validation accuracy was best
        if self.validation and ii > 500:
            for i in np.arange(ii, 10, -1):
                if self.validation_scores[i] > self.validation_scores[i - 1]:
                    self.L = self.Ls[i]
                    break

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

            nonZeroCols = np.sum(np.linalg.norm(
                self.L.T, ord=2, axis=0) > 1.E-3)
            print("L Nonzero cols {:d}/{:d}".format(nonZeroCols, self.D))
            print(' ')

        # DELETE STORED MATRIX PRODUCT VALUES GARBAGE COLLECTION TAKES TOO MUCH TIME...
        self._rm_Xij_dics()

    def _test_for_termination(self, ii=3):
        '''Checks for min step size, convergence of the loss function, max iterations'''

        # Convergence tolerance
        if ii > 10 and np.all(np.abs((100 * (self.loss_values[-1] - np.array(self.loss_values[-4:-1])) / self.loss_values[-1:])) <= self.convergence_tol):
            # self._stepsize *= 1E-3
            if ii > 10 and np.all(np.abs((100 * (self.loss_values[-1] - np.array(self.loss_values[-10:-1])) / self.loss_values[-1:])) <= self.convergence_tol):
                termination_str_nolosschange = "\n\033[92mStopping\033[0m since loss change prcnt is below threshold value."
                return True, termination_str_nolosschange

        # Step size
        if (self._stepsize < self._stepsize_min):
            termination_str_minstepsize = "\n\033[92mStopping\033[0m since step size smaller than min step size."
            return True, termination_str_minstepsize

        # Max iterations
        if ii > self.max_iter:
            termination_str_maxiter = "\n\033[92mStopping\033[0m since maximum number of iterations reached before convergence."
            return True, termination_str_maxiter

        # Validation set goes negative
        if self.validation and ii > 100:
            if np.all(self.validation_scores[-5:] - np.array(self.validation_scores[-6:-1]) <= 0.):
                # print(' ')
                # print(' ')
                # print(' ')
                # print(self.validation_scores[-5:])
                # print(np.array(self.validation_scores[-6:-1]))
                # plt.plot(np.arange(0,ii+1), self.validation_scores)
                # plt.show()
                # print(' ')
                termination_str_validation = "\n\033[92mStopping\033[0m since validation scores have a negative derivative for 10 steps."
                # print(termination_str_validation)
                return True, termination_str_validation

        # if self.leave_one_out_average_score(self.X,self.labels,self.L) >= 1.:# and ii > 40:
        #     if self._verbose:
        #         print("\033[92mStopping\033[0m since LOE is 1.")
        #     return True

        # if self._count_active_constraints() == 0:
        #     # Update constraints.
        #     self.loss_fun(self.L,active=False,chg_active=True)
        #     # Do 2nd check
        #     if self._count_active_constraints() == 0:
        #         if self._verbose:
        #             print("\n\033[92mStopping\033[0m since the 2nd term constraints are fulfilled.")
        #         return True

        # Passed all, so continue
        return False, "Passed termination tests."

    def loss_fun(self, L, active=False, chg_active=False, verbose=False):
        er1 = 0
        er2 = 0

        # For each point...
        for idx, label in enumerate(self.labels):

            # ...and its k nearest neighbors
            for nn_idx, jdx in enumerate(self.NNmat[idx, :]):
                
                # if self._nn_weights[nn_idx] < 1.E-12:
                    # continue
                
                # LOSS TERM 1 1xD * D*d
                Lij = self._cpt_Xij_diff(idx, jdx).dot(L)
                xLijNorm = Lij.dot(Lij.T)
                er1 += self._nn_weights[nn_idx] * self.ratios[label] * xLijNorm

                # LOSS TERM 2
                # For all points not in class=label
                Lil = self._cpt_Xij_diff(idx, self._non_label_idxs[label]).dot(L)
                impostor_dists = np.ravel(xLijNorm - np.einsum('ij, ji->i', Lil, Lil.T) + 1.)
                er2 += self._nn_weights[nn_idx] * self.ratios[label] * np.sum(self.smooth_hinge_vec(impostor_dists))

                # Deactivate constraints not active only for final loss
                if chg_active:

                    for ldx in np.flatnonzero(impostor_dists > 0.):
                        self._active_consts[idx][jdx][self._non_label_idxs[label][ldx]] = 1

                    for ldx in np.flatnonzero(impostor_dists < 0.):
                        self._active_consts[idx][jdx][self._non_label_idxs[label][ldx]] = 0

        er = er1 + self.reg * er2

        # Regularization
        for idx, name in enumerate(self._penfun_list):
            er += self._penfun_weights[name] * self._penfuns[name](L)

        if self._verbose and verbose:
            info_str = "Er1: {:.3f}, Er2: {:.3f}, Er1/Er2: {:.3f}"
            print(info_str.format(er1, self.reg * er2, (er1 / (self.reg * er2))))

        return np.asscalar(er)

    def d_loss_fun(self, active=False):
        dL1 = np.zeros((self.D, self.D))
        dL2 = np.zeros((self.D, self.D))

        # For each point...
        for idx, label in enumerate(self.labels):

            # ...and its k nearest neighbors
            for nn_idx, jdx in enumerate(self.NNmat[idx, :]):

                if self._nn_weights[nn_idx] < 1.E-20:
                    continue

                XijProd = self._cpt_Xij_matprd(idx, jdx)
                
                # LOSS TERM 1 Precomputed
                # dL1 += self._nn_weights[nn_idx] * self.ratios[label] * XijProd

                # LOSS TERM 2
                if active:
                    # Check only for active impostors to speed up computations
                    ldxActive = []
                    for ldx in self._non_label_idxs[label]:
                        if self._active_consts[idx][jdx][ldx] == 1:
                            ldxActive.append(ldx)
                    if len(ldxActive) == 0:
                        continue

                    Lil = self._cpt_Xij_diff(idx, ldxActive).dot(self.L)
                else:
                    ldxActive = self._non_label_idxs[label]
                    Lil = self._cpt_Xij_diff(idx, ldxActive).dot(self.L)

                Lij = self._cpt_Xij_diff(idx, jdx).dot(self.L)
                hingevec = self.df_smooth_hinge_vec(np.ravel(Lij.dot(Lij.T) - np.einsum('ij, ji->i', Lil, Lil.T) + 1.))
                # hingevec = self.df_smooth_hinge_vec(inner1d(Lij, Lij) - inner1d(Lil, Lil) + 1.)
                # hingevec = self.df_quadloss_vec(
                #     inner1d(Lij, Lij) - inner1d(Lil, Lil) + 1.)

                # ...and its impostors
                if len(ldxActive) > 0. and np.any(hingevec > 0.):
                    # XijProd = self._cpt_Xij_matprd(idx, jdx)
                    for (lidx, ldx) in enumerate(ldxActive):
                        if hingevec[lidx] > 0.:
                            # XilProd = self._cpt_Xij_diff(idx,ldx).dot(self._cpt_Xij_diff(idx,ldx).T)
                            dL2 += self._nn_weights[nn_idx] * self.ratios[label] * \
                                (XijProd - self._cpt_Xij_matprd(idx, ldx)) * \
                                hingevec[lidx]

        dL = 2. * (self.dL1 + self.reg * dL2).dot(self.L)
        # Regularization functions acting on L
        for idx, name in enumerate(self._penfun_list):
            dL += self._penfun_weights[name] * self._df_penfuns[name](self.L)

        return dL


