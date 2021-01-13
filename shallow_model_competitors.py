import numpy as np
from time import time
import scipy.linalg as spl
from utils import *

class ADMM_lin(object):
    def __init__(self, rankings, X):
        '''
        n: number of items
        p: number of features
        M: number of rankings
        :param rankings: (c_l, A_l): 1...M
        :param X: n*p, feature matrix
        :param method_pi_tilde_init: for ilsr_feat, initialize with prev_weights or orthogonal projection
        '''
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.M = len(rankings)
        self.X = X.astype(float)
        self.X_ls = np.linalg.solve(np.dot(self.X.T, self.X) + np.eye(self.p, dtype=float) * rtol, self.X.T)
        self.X_tilde = np.concatenate((self.X, np.ones((self.n, 1))), axis=1)
        self.X_tilde_ls = np.linalg.solve(np.dot(self.X_tilde.T, self.X_tilde)
                                          + np.eye(self.p+1, dtype=float) * rtol, self.X_tilde.T)
        self.rankings = rankings

    def fit_lin(self, rho, weights=None, u=None, gamma=1):
        '''
        :param rho: penalty parameter
        :param beta: parameter vector at each iteration, px1
        :param b: bias at each iteration, scalar
        :param weights: scores at each iteration, nx1
        :param u: scaled dual variable at each iteration, nx1
        :param gamma: scaling on the dual variable update
        '''
        start = time()
        ## beta_tilde = (beta; b) update
        # beta_tilde = spl.lstsq(self.X_tilde, weights - u)[0]  # uses svd
        beta_tilde = np.dot(self.X_tilde_ls, weights - u)
        x_beta_b = np.dot(self.X_tilde, beta_tilde)
        ## pi update
        weights = self.ilsrx_lin(rho=rho, weights=weights, x_beta_b=x_beta_b, u=u)
        ## dual update
        u += gamma * (x_beta_b - weights)
        end = time()
        return weights, x_beta_b, u, (end - start)

    def ilsrx_lin(self, rho, weights, x_beta_b, u):
        """modified spectral ranking algorithm for partial ranking data. Remove the inner loop for top-1 ranking.
        n: number of items
        rho: penalty parameter
        sigmas = rho * (weights - Xbeta - b - u) is the additional term compared to ILSR
        """
        ilsr_conv = False
        iter = 0
        while not ilsr_conv:
            sigmas = rho * (weights - x_beta_b - u)
            pi_sigmas = weights * sigmas
            #######################
            # indices of states for which sigmas < 0
            ind_minus = np.where(sigmas < 0)[0]
            # indices of states for which sigmas >= 0
            ind_plus = np.where(sigmas >= 0)[0]
            # sum of pi_sigmas over states for which sigmas >= 0
            scaled_sigmas_plus = 2 * sigmas[ind_plus] / (np.sum(pi_sigmas[ind_minus]) - np.sum(pi_sigmas[ind_plus]))
            # fill up the transition matrix
            chain = np.zeros((self.n, self.n), dtype=float)
            # increase the outgoing rate from ind_plus to ind_minus
            for ind_minus_cur in ind_minus:
                chain[ind_plus, ind_minus_cur] = pi_sigmas[ind_minus_cur] * scaled_sigmas_plus
            for ranking in self.rankings:
                sum_weights = sum(weights[x] for x in ranking) + epsilon
                for i, winner in enumerate(ranking):
                    val = 1.0 / sum_weights
                    for loser in ranking[i + 1:]:
                        chain[loser, winner] += val
                    sum_weights -= weights[winner]
            # each row sums up to 0
            chain -= np.diag(chain.sum(axis=1))
            weights_prev = np.copy(weights)
            weights = statdist(chain, v_init=weights)
            # Check convergence
            iter += 1
            ilsr_conv = np.linalg.norm(weights_prev - weights) < rtol * np.linalg.norm(weights) or iter >= n_iter
        return weights


class ADMM_kl_lin(object):
    def __init__(self, rankings, X):
        '''
        n: number of items
        p: number of features
        M: number of rankings
        :param rankings: (c_l, A_l): 1...M
        :param X: n*p, feature matrix
        :param method_pi_tilde_init: for ilsr_feat, initialize with prev_weights or orthogonal projection
        '''
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.M = len(rankings)
        self.X = X.astype(float)
        self.X_tilde = np.concatenate((self.X, np.ones((self.n, 1))), axis=1)
        self.rankings = rankings

    def fit_lin(self, rho, weights=None, u=None, gamma=1):
        '''
        :param rho: penalty parameter
        :param beta: parameter vector at each iteration, px1
        :param b: bias at each iteration, scalar
        :param weights: scores at each iteration, nx1
        :param u: scaled dual variable at each iteration, nx1
        :param gamma: scaling on the dual variable update
        '''
        # Define variables
        beta = cp.Variable(self.p)
        b = cp.Variable(1)
        params = cp.vstack([beta, b])
        # Define objective
        objective = u.T * (self.X_tilde * params) - weights.T * cp.log(self.X_tilde * params)
        # Optimize
        prob = cp.Problem(cp.Minimize(objective))
        resolution = rtol
        converged = False
        while not converged:
            start = time()
            # Optimize
            prob.solve(solver='SCS', eps=resolution)
            converged = params.value is not None
            # If cannot be solved, reduce the accuracy requirement
            resolution *= 2
        x_beta_b = np.dot(self.X_tilde, np.squeeze(np.array(params.value)))
        ## pi update
        weights = self.ilsrx_lin(rho=rho, weights=weights, x_beta_b=x_beta_b, u=u)
        ## dual update
        u += gamma * (x_beta_b - weights)
        end = time()
        return weights, x_beta_b, u, (end - start)

    def ilsrx_lin(self, rho, weights, x_beta_b, u):
        """modified spectral ranking algorithm for partial ranking data. Remove the inner loop for top-1 ranking.
        n: number of items
        rho: penalty parameter
        sigmas is the additional term compared to ILSR
        """
        ilsr_conv = False
        iter = 0
        while not ilsr_conv:
            sigmas = rho * (1 + np.log(np.divide(weights, x_beta_b + epsilon) + epsilon) - u)
            pi_sigmas = weights * sigmas
            #######################
            # indices of states for which sigmas < 0
            ind_minus = np.where(sigmas < 0)[0]
            # indices of states for which sigmas >= 0
            ind_plus = np.where(sigmas >= 0)[0]
            # sum of pi_sigmas over states for which sigmas >= 0
            scaled_sigmas_plus = 2 * sigmas[ind_plus] / (np.sum(pi_sigmas[ind_minus]) - np.sum(pi_sigmas[ind_plus]))
            # fill up the transition matrix
            chain = np.zeros((self.n, self.n), dtype=float)
            # increase the outgoing rate from ind_plus to ind_minus
            for ind_minus_cur in ind_minus:
                chain[ind_plus, ind_minus_cur] = pi_sigmas[ind_minus_cur] * scaled_sigmas_plus
            for ranking in self.rankings:
                sum_weights = sum(weights[x] for x in ranking) + epsilon
                for i, winner in enumerate(ranking):
                    val = 1.0 / sum_weights
                    for loser in ranking[i + 1:]:
                        chain[loser, winner] += val
                    sum_weights -= weights[winner]
            # each row sums up to 0
            chain -= np.diag(chain.sum(axis=1))
            weights_prev = np.copy(weights)
            weights = statdist(chain, v_init=weights)
            # Check convergence
            iter += 1
            ilsr_conv = np.linalg.norm(weights_prev - weights) < rtol * np.linalg.norm(weights) or iter >= n_iter
        return weights
