import numpy as np
from time import time
import scipy.linalg as spl
import cvxpy as cp
from utils import *
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from time import time
from googlenet_functional import *


def custom_loss_kl_pi_pitilde(weights, y_pred):
    return u_global * y_pred + K.binary_crossentropy(weights, y_pred)


class deep_KL_ADMM(object):
    def __init__(self, rankings, imgs_lst, train_samp, rankings_val, reg_param=0.0002, learning_rate=1e-4,
                init_last_layer_beta=None, init_last_layer_b=None,
                n=100, save_model_name='deep_admm.h5', basenet="googlenet", no_of_layers=2):
        '''
        :param rankings: [(i_1,i_2, ...), (i_1,i_2, ...), ...]
        :param imgs_lst: (n, 3, 224, 224) /  (n, p)
        :param reg_param: regularization for network layers
        :param n: number of samples
        '''
        self.train_samp = train_samp
        self.rankings = rankings
        self.rankings_val = rankings_val
        self.imgs_lst = imgs_lst
        self.input_shape = imgs_lst.shape[1:]
        self.reg_param = reg_param
        self.lr = learning_rate
        self.batch_size = batch_size
        self.save_model_name = save_model_name
        global u_global
        u_global = np.zeros(len(train_samp), dtype=float)
        self.weight_net = self.create_base_network(basenet=basenet, no_of_layers=no_of_layers,
                                                   init_last_layer_weights=[init_last_layer_beta, init_last_layer_b])
        self.n = n

    def create_base_network(self, basenet="googlenet", no_of_layers=2, init_last_layer_weights=None):
        input = Input(shape=self.input_shape)
        # get features from base network
        if basenet == "googlenet":
            comp_out, _ = create_googlenet(input, input, reg_param=self.reg_param)
            weight_net = Model(input, comp_out)
            weight_net.load_weights(GOOGLENET_INIT_WEIGHTS_PATH, by_name=True)
            if init_last_layer_weights[0] is not None:
                weight_net.get_layer('comp').set_weights(init_last_layer_weights)
        else:
            comp_out, _ = create_fc_basenet(input, input, reg_param=self.reg_param, no_of_layers=no_of_layers)
            weight_net = Model(input, comp_out)
        weight_net.compile(loss=custom_loss_kl_pi_pitilde, optimizer=Adam(self.lr))
        weight_net.summary()
        return weight_net

    def fit_deep(self, rho, weights=None, x_beta_b=None, u=None, gamma=1, max_inner_iter=n_iter):
        '''
        :param rho: penalty parameter
        :param x_beta_b: base network prediction of scores
        :param weights: scores at each iteration, nx1
        :param u: scaled dual variable at each iteration, nx1
        :param gamma: scaling on the dual variable update
        '''
        #if np.all(weights == 1.0 * np.ones(self.n, dtype=float) / self.n):
        start = time()
        ## pi update: no matter what the initial weights are, should come first.
        weights = self.ilsrx(rho=rho, weights=weights, x_beta_b=x_beta_b, u=u)
        end = time()
        ilrsx_time = (end - start)
        ## beta_tilde = (beta; b) update UNTIL CONVERGENCE
        params_conv = False
        nn_time = 0
        iter = 0
        val_kendall = [kendall_tau_test(x_beta_b, self.rankings_val)]
        obj = [objective(x_beta_b, self.rankings)]
        while not params_conv:
            current_input = self.imgs_lst[self.train_samp]
            current_output = weights[self.train_samp]
            start = time()
            history = self.weight_net.fit(current_input, current_output, batch_size=self.batch_size, epochs=1)
            end = time()
            x_beta_b = np.squeeze(self.weight_net.predict(self.imgs_lst))  # predict new scores
            nn_time += (end - start)  # log time
            iter += 1  # log number of epochs
            val_kendall.append(kendall_tau_test(x_beta_b, self.rankings_val))  # current validation performance
            obj.append(objective(x_beta_b, self.rankings))
            if iter <= avg_window:
                avg_val_kendall = np.mean(val_kendall[:-1])
                avg_obj = np.mean(obj[:-1])
            else:
                avg_val_kendall = np.mean(val_kendall[-avg_window - 1:-1])
                avg_obj = np.mean(obj[-avg_window - 1:-1])
            params_conv = (np.abs(avg_val_kendall - val_kendall[-1]) < rtol * avg_val_kendall \
                            and np.abs(avg_obj - obj[-1]) < rtol * avg_obj) \
                            or iter >= max_inner_iter  # check conv.
        print("\n ********* Deep KL ADMM no of epochs for inner convergence", iter)
        ## dual update
        start = time()
        u += gamma * (x_beta_b - weights)
        end = time()
        global u_global
        u_global = np.copy(u[self.train_samp])
        u_time = (end - start)
        self.weight_net.save(self.save_model_name)
        return weights, x_beta_b, u, ilrsx_time, nn_time, u_time, history.history['loss'][-1]

    def ilsrx(self, rho, weights, x_beta_b, u):
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
