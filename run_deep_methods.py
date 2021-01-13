from utils import *
from deep_kl_admm import *
from deep_l2_admm import *
from siamese_network import *
from shallow_model_competitors import *
import pickle
import csv
from math import ceil
import argparse
from os.path import exists
from os import listdir, remove, mkdir
import numpy as np
from scipy.sparse import load_npz
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
from itertools import product

def init_all_methods(dir, val_fold, reg_param, learning_rate, basenet, no_of_layers):
    save_name = str(val_fold)
    train_samp = np.load('../data/' + dir + 'data/' + save_name + '_train_samp.npy')  # training sample indices
    X = np.load('../data/' + dir + 'data/' + save_name + '_features.npy', allow_pickle=True)  # extracted features
    imgs_lst = np.load('../data/' + dir + 'data/' + save_name + '_imgs_lst.npy')  # original images
    rankings_train = np.load('../data/' + dir + 'data/' + save_name + '_train.npy')
    rankings_val = np.load('../data/' + dir + 'data/' + save_name + '_val.npy')
    rankings_test = np.load('../data/' + dir + 'data/rankings_test.npy')
    n = imgs_lst.shape[0]  # number of samples
    if basenet == "googlenet":
        # siamese network data
        comp_imgs_train_lst_left = np.load('../data/' + dir + 'data/' + save_name + '_comp_train_imgs_lst_left.npy')
        comp_imgs_train_lst_right = np.load('../data/' + dir + 'data/' + save_name + '_comp_train_imgs_lst_right.npy')
        comp_imgs_train_lst_pair = [comp_imgs_train_lst_left, comp_imgs_train_lst_right]
        comp_train_labels = np.load('../data/' + dir + 'data/' + save_name + '_comp_train_labels.npy')
    else:
        # multiway siamese network data
        rank_imgs_lst = list(np.load('../data/' + dir + 'data/' + save_name + '_rank_imgs_lst.npy'))
        true_order_labels = np.load('../data/' + dir + 'data/' + save_name + '_true_order_labels.npy')

    # Load initial parameters
    with open('../data/' + dir + 'data/' + save_name + '_init_params.pickle', "rb") as pickle_in:
        all_init_params = pickle.load(pickle_in)
    [(beta_init, b_init, time_beta_b_init), (_, _), (u_init, time_u_init)] = all_init_params
    save_name += '_no_of_layers_' + str(no_of_layers) + '_lambda_' + str(reg_param) + '_lr_' + str(learning_rate)
    
    # deep KL admm parameters
    deep_admm_log_dict = dict()
    model_name = '../models/' + dir + 'model/deep_admm_' + save_name + '.h5'
    deep_admm_object = deep_KL_ADMM(rankings_train, imgs_lst, train_samp, rankings_val,
                            reg_param=reg_param, learning_rate=learning_rate,
                            init_last_layer_beta=None, init_last_layer_b=None,
                            n=n, save_model_name=model_name, basenet=basenet, no_of_layers=no_of_layers)
    deep_admm_log_dict['pi_deep_admm'] = 1.0 * np.ones(n, dtype=float) / n
    deep_admm_log_dict['time_deep_admm'] = [time_u_init]
    deep_admm_log_dict['time_cont_deep_admm'] = [time_u_init]
    deep_admm_log_dict['deep_admm_conv'] = False
    deep_admm_log_dict['pi_tilde_deep_admm'] = np.copy(deep_admm_log_dict['pi_deep_admm'])
    deep_admm_log_dict['u_deep_admm'] = np.copy(u_init)
    deep_admm_log_dict['iter_deep_admm'] = 0
    deep_admm_log_dict['prim_feas_deep_admm'] = [epsilon]  # violation of equality constraint
    deep_admm_log_dict['dual_feas_deep_admm'] = [epsilon]  # (pi_tilde^(k+1)-pi_tilde^k)
    deep_admm_log_dict['obj_pi_tilde_deep_admm'] = [objective(deep_admm_log_dict['pi_tilde_deep_admm'], rankings_train)]
    deep_admm_log_dict['obj_pi_deep_admm'] = [objective(deep_admm_log_dict['pi_deep_admm'], rankings_train)]
    deep_admm_log_dict['negative_loss_deep_admm'] = [objective(deep_admm_log_dict['pi_tilde_deep_admm'], rankings_train)]
    deep_admm_log_dict['val_acc_pi_tilde_deep_admm'] = [top1_test_accuracy(deep_admm_log_dict['pi_tilde_deep_admm'], rankings_val)]
    deep_admm_log_dict['val_acc_pi_deep_admm'] = [top1_test_accuracy(deep_admm_log_dict['pi_deep_admm'], rankings_val)]
    deep_admm_log_dict['val_kendall_pi_tilde_deep_admm'] = [kendall_tau_test(deep_admm_log_dict['pi_tilde_deep_admm'], rankings_val)]
    deep_admm_log_dict['val_kendall_pi_deep_admm'] = [kendall_tau_test(deep_admm_log_dict['pi_deep_admm'], rankings_val)]
    deep_admm_log_dict['test_acc_pi_tilde_deep_admm'] = [top1_test_accuracy(deep_admm_log_dict['pi_tilde_deep_admm'], rankings_test)]
    deep_admm_log_dict['test_acc_pi_deep_admm'] = [top1_test_accuracy(deep_admm_log_dict['pi_deep_admm'], rankings_test)]
    deep_admm_log_dict['test_kendall_pi_tilde_deep_admm'] = [kendall_tau_test(deep_admm_log_dict['pi_tilde_deep_admm'], rankings_test)]
    deep_admm_log_dict['test_kendall_pi_deep_admm'] = [kendall_tau_test(deep_admm_log_dict['pi_deep_admm'], rankings_test)]
    deep_admm_log_dict['rho_deep_admm'] = [1]

    # siamese network parameters
    siamese_log_dict = dict()
    model_name = '../models/' + dir + 'model/siamese_' + save_name + '.h5'
    if basenet == "googlenet":
        siamese_object = comparison_siamese(comp_train_labels, comp_imgs_train_lst_pair, imgs_lst,
                             reg_param=reg_param, learning_rate=learning_rate,
                             init_last_layer_beta=None, save_model_name=model_name,
                             basenet=basenet, no_of_layers=no_of_layers)
    else:
        siamese_object = multiway_siamese(true_order_labels, rank_imgs_lst, imgs_lst,
                                        reg_param=reg_param, learning_rate=learning_rate,
                                        save_model_name=model_name, no_of_layers=no_of_layers)
    siamese_log_dict['pi_tilde_siamese'] = 1.0 * np.ones(n, dtype=float) / n
    siamese_log_dict['time_siamese'] = [epsilon]
    siamese_log_dict['time_cont_siamese'] = [epsilon]
    siamese_log_dict['siamese_conv'] = False
    siamese_log_dict['iter_siamese'] = 0
    siamese_log_dict['obj_siamese'] = [objective(np.exp(siamese_log_dict['pi_tilde_siamese']), rankings_train)]
    siamese_log_dict['val_acc_siamese'] = [top1_test_accuracy(siamese_log_dict['pi_tilde_siamese'], rankings_val)]
    siamese_log_dict['val_kendall_siamese'] = [kendall_tau_test(siamese_log_dict['pi_tilde_siamese'], rankings_val)]
    siamese_log_dict['test_acc_siamese'] = [top1_test_accuracy(siamese_log_dict['pi_tilde_siamese'], rankings_test)]
    siamese_log_dict['test_kendall_siamese'] = [kendall_tau_test(siamese_log_dict['pi_tilde_siamese'], rankings_test)]

    # linear admm parameters
    lin_admm_log_dict = dict()
    lin_admm_object = ADMM_lin(rankings_train, X)
    lin_admm_log_dict['lin_admm_conv'] = False
    lin_admm_log_dict['pi_lin_admm'] = np.squeeze(np.dot(X, beta_init) + b_init)
    if np.min(lin_admm_log_dict['pi_lin_admm']) < 0:
        lin_admm_log_dict['pi_lin_admm'] += np.abs(np.min(lin_admm_log_dict['pi_lin_admm'])) + rtol
    lin_admm_log_dict['pi_tilde_lin_admm'] = np.copy(lin_admm_log_dict['pi_lin_admm'])
    lin_admm_log_dict['u_lin_admm'] = np.copy(u_init)
    lin_admm_log_dict['time_lin_admm'] = [time_beta_b_init + time_u_init]
    lin_admm_log_dict['time_cont_lin_admm'] = [time_beta_b_init + time_u_init]
    lin_admm_log_dict['iter_lin_admm'] = 0
    lin_admm_log_dict['obj_lin_admm'] = [objective(lin_admm_log_dict['pi_lin_admm'], rankings_train)]
    lin_admm_log_dict['val_acc_lin_admm'] = [top1_test_accuracy(lin_admm_log_dict['pi_lin_admm'], rankings_val)]
    lin_admm_log_dict['val_kendall_lin_admm'] = [kendall_tau_test(lin_admm_log_dict['pi_lin_admm'], rankings_val)]
    lin_admm_log_dict['test_acc_lin_admm'] = [top1_test_accuracy(lin_admm_log_dict['pi_lin_admm'], rankings_test)]
    lin_admm_log_dict['test_kendall_lin_admm'] = [kendall_tau_test(lin_admm_log_dict['pi_lin_admm'], rankings_test)]

    # linear kl admm parameters
    lin_kl_admm_log_dict = dict()
    lin_kl_admm_object = ADMM_kl_lin(rankings_train, X)
    lin_kl_admm_log_dict['lin_kl_admm_conv'] = False
    lin_kl_admm_log_dict['pi_lin_kl_admm'] = np.copy(lin_admm_log_dict['pi_lin_admm'])
    lin_kl_admm_log_dict['pi_tilde_lin_kl_admm'] = np.copy(lin_kl_admm_log_dict['pi_lin_kl_admm'])
    lin_kl_admm_log_dict['u_lin_kl_admm'] = np.copy(u_init)
    lin_kl_admm_log_dict['time_lin_kl_admm'] = [time_beta_b_init + time_u_init]
    lin_kl_admm_log_dict['time_cont_lin_kl_admm'] = [time_beta_b_init + time_u_init]
    lin_kl_admm_log_dict['iter_lin_kl_admm'] = 0
    lin_kl_admm_log_dict['obj_lin_kl_admm'] = [objective(lin_kl_admm_log_dict['pi_lin_kl_admm'], rankings_train)]
    lin_kl_admm_log_dict['val_acc_lin_kl_admm'] = [top1_test_accuracy(lin_kl_admm_log_dict['pi_lin_kl_admm'], rankings_val)]
    lin_kl_admm_log_dict['val_kendall_lin_kl_admm'] = [kendall_tau_test(lin_kl_admm_log_dict['pi_lin_kl_admm'], rankings_val)]
    lin_kl_admm_log_dict['test_acc_lin_kl_admm'] = [top1_test_accuracy(lin_kl_admm_log_dict['pi_lin_kl_admm'], rankings_test)]
    lin_kl_admm_log_dict['test_kendall_lin_kl_admm'] = [kendall_tau_test(lin_kl_admm_log_dict['pi_lin_kl_admm'], rankings_test)]

    # deep L2 admm parameters
    deep_l2_admm_log_dict = dict()
    model_name = '../models/' + dir + 'model/deep_l2_admm_' + save_name + '.h5'
    deep_l2_admm_object = deep_l2_ADMM(rankings_train, imgs_lst, train_samp, rankings_val,
                                    reg_param=reg_param, learning_rate=learning_rate,
                                    init_last_layer_beta=None, init_last_layer_b=None,
                                    n=n, save_model_name=model_name, basenet=basenet, no_of_layers=no_of_layers)
    deep_l2_admm_log_dict['pi_deep_l2_admm'] = 1.0 * np.ones(n, dtype=float) / n
    deep_l2_admm_log_dict['time_deep_l2_admm'] = [time_u_init]
    deep_l2_admm_log_dict['time_cont_deep_l2_admm'] = [time_u_init]
    deep_l2_admm_log_dict['deep_l2_admm_conv'] = False
    deep_l2_admm_log_dict['pi_tilde_deep_l2_admm'] = np.copy(deep_l2_admm_log_dict['pi_deep_l2_admm'])
    deep_l2_admm_log_dict['u_deep_l2_admm'] = np.copy(u_init)
    deep_l2_admm_log_dict['iter_deep_l2_admm'] = 0
    deep_l2_admm_log_dict['prim_feas_deep_l2_admm'] = [epsilon]  # violation of equality constraint
    deep_l2_admm_log_dict['dual_feas_deep_l2_admm'] = [epsilon]  # (pi_tilde^(k+1)-pi_tilde^k)
    deep_l2_admm_log_dict['obj_pi_tilde_deep_l2_admm'] = [objective(deep_l2_admm_log_dict['pi_tilde_deep_l2_admm'], rankings_train)]
    deep_l2_admm_log_dict['obj_pi_deep_l2_admm'] = [objective(deep_l2_admm_log_dict['pi_deep_l2_admm'], rankings_train)]
    deep_l2_admm_log_dict['val_acc_pi_tilde_deep_l2_admm'] = [top1_test_accuracy(deep_l2_admm_log_dict['pi_tilde_deep_l2_admm'], rankings_val)]
    deep_l2_admm_log_dict['val_acc_pi_deep_l2_admm'] = [top1_test_accuracy(deep_l2_admm_log_dict['pi_deep_l2_admm'], rankings_val)]
    deep_l2_admm_log_dict['val_kendall_pi_tilde_deep_l2_admm'] = [kendall_tau_test(deep_l2_admm_log_dict['pi_tilde_deep_l2_admm'], rankings_val)]
    deep_l2_admm_log_dict['val_kendall_pi_deep_l2_admm'] = [kendall_tau_test(deep_l2_admm_log_dict['pi_deep_l2_admm'], rankings_val)]
    deep_l2_admm_log_dict['test_acc_pi_tilde_deep_l2_admm'] = [top1_test_accuracy(deep_l2_admm_log_dict['pi_tilde_deep_l2_admm'], rankings_test)]
    deep_l2_admm_log_dict['test_acc_pi_deep_l2_admm'] = [top1_test_accuracy(deep_l2_admm_log_dict['pi_deep_l2_admm'], rankings_test)]
    deep_l2_admm_log_dict['test_kendall_pi_tilde_deep_l2_admm'] = [kendall_tau_test(deep_l2_admm_log_dict['pi_tilde_deep_l2_admm'], rankings_test)]
    deep_l2_admm_log_dict['test_kendall_pi_deep_l2_admm'] = [kendall_tau_test(deep_l2_admm_log_dict['pi_deep_l2_admm'], rankings_test)]
    print('INITIALIZATION OVER!')

    if not exists('../results/' + dir + 'fig'):
        mkdir('../results/' + dir + 'fig')
    if not exists('../models/' + dir + 'model'):
        mkdir('../models/' + dir + 'model')

    return deep_admm_log_dict, siamese_log_dict, lin_admm_log_dict, lin_kl_admm_log_dict, deep_l2_admm_log_dict, \
           deep_admm_object, siamese_object, lin_admm_object, lin_kl_admm_object, deep_l2_admm_object, \
           save_name, X, imgs_lst, train_samp, rankings_train, rankings_val, rankings_test


def run_save_all_methods(tasks, dir, val_fold, reg_param, learning_rate, basenet, no_of_layers,
                         max_inner_iter, rho_deep_admm, rho_deep_l2_admm, gamma, plot_results):
    '''
    Run all methods and save all logged results
    :param tasks: list of algorithms to be run
    :param dir: data directory
    :param val_fold: index of fold
    :param reg_param: regularization for network layers
    :param rho: penalty parameter of ADMM
    '''
    deep_admm_log_dict, siamese_log_dict, lin_admm_log_dict, lin_kl_admm_log_dict, deep_l2_admm_log_dict, \
    deep_admm_object, siamese_object, lin_admm_object, lin_kl_admm_object, deep_l2_admm_object, \
    save_name, X, imgs_lst, train_samp, rankings_train, rankings_val, rankings_test, \
                        = init_all_methods(dir, val_fold, reg_param, learning_rate, basenet, no_of_layers)
    for ind_iter in range(n_iter):
        # Diminishing gamma
        gamma /= (ind_iter + 1)
        # deep KL admm update
        if not deep_admm_log_dict['deep_admm_conv'] and 'deep_admm' in tasks:
            deep_admm_log_dict['pi_deep_admm_prev'] = deep_admm_log_dict['pi_deep_admm']  # spectral prediction
            deep_admm_log_dict['pi_tilde_deep_admm_prev'] = deep_admm_log_dict['pi_tilde_deep_admm']  # regressed
            # Adapt rho
            r = deep_admm_log_dict['prim_feas_deep_admm'][-1]
            s = deep_admm_log_dict['dual_feas_deep_admm'][-1]
            if r > mu * s:
                rho_deep_admm *= tau
            elif s > mu * r:
                rho_deep_admm /= tau
            deep_admm_log_dict['rho_deep_admm'].append(rho_deep_admm)
            # Update
            deep_admm_log_dict['pi_deep_admm'], deep_admm_log_dict['pi_tilde_deep_admm'], deep_admm_log_dict['u_deep_admm'], \
                        ilsrx_time_deep_admm_iter, nn_time_deep_admm_iter, u_time_deep_admm_iter, deep_admm_loss = \
                        deep_admm_object.fit_deep(rho_deep_admm, weights=deep_admm_log_dict['pi_deep_admm'],
                        x_beta_b=deep_admm_log_dict['pi_tilde_deep_admm'], u=deep_admm_log_dict['u_deep_admm'],
                        gamma=gamma, max_inner_iter=max_inner_iter)
            time_deep_admm_iter = ilsrx_time_deep_admm_iter + nn_time_deep_admm_iter + u_time_deep_admm_iter
            deep_admm_log_dict['time_deep_admm'].append(time_deep_admm_iter)
            deep_admm_log_dict['iter_deep_admm'] += 1
            deep_admm_log_dict['prim_feas_deep_admm'].append(np.linalg.norm(deep_admm_log_dict['pi_tilde_deep_admm'] -
                                                                         deep_admm_log_dict['pi_deep_admm']))
            deep_admm_log_dict['dual_feas_deep_admm'].append(np.linalg.norm(deep_admm_log_dict['pi_tilde_deep_admm_prev'] -
                                                                         deep_admm_log_dict['pi_tilde_deep_admm']))
            deep_admm_log_dict['obj_pi_tilde_deep_admm'].append(objective(deep_admm_log_dict['pi_tilde_deep_admm'], rankings_train))
            deep_admm_log_dict['obj_pi_deep_admm'].append(objective(deep_admm_log_dict['pi_deep_admm'], rankings_train))
            deep_admm_log_dict['negative_loss_deep_admm'].append(-deep_admm_loss)
            deep_admm_log_dict['val_acc_pi_tilde_deep_admm'].append(top1_test_accuracy(deep_admm_log_dict['pi_tilde_deep_admm'], rankings_val))
            deep_admm_log_dict['val_acc_pi_deep_admm'].append(top1_test_accuracy(deep_admm_log_dict['pi_deep_admm'], rankings_val))
            deep_admm_log_dict['val_kendall_pi_tilde_deep_admm'].append(kendall_tau_test(deep_admm_log_dict['pi_tilde_deep_admm'], rankings_val))
            deep_admm_log_dict['val_kendall_pi_deep_admm'].append(kendall_tau_test(deep_admm_log_dict['pi_deep_admm'], rankings_val))
            deep_admm_log_dict['test_acc_pi_tilde_deep_admm'].append(top1_test_accuracy(deep_admm_log_dict['pi_tilde_deep_admm'], rankings_test))
            deep_admm_log_dict['test_acc_pi_deep_admm'].append(top1_test_accuracy(deep_admm_log_dict['pi_deep_admm'], rankings_test))
            deep_admm_log_dict['test_kendall_pi_tilde_deep_admm'].append(kendall_tau_test(deep_admm_log_dict['pi_tilde_deep_admm'], rankings_test))
            deep_admm_log_dict['test_kendall_pi_deep_admm'].append(kendall_tau_test(deep_admm_log_dict['pi_deep_admm'], rankings_test))
            # Correct time scale
            deep_admm_log_dict['time_cont_deep_admm'].append(sum(deep_admm_log_dict['time_deep_admm']))
            # Check convergence
            if ind_iter > avg_window:
                avg_val_kendall = np.mean(deep_admm_log_dict['val_kendall_pi_tilde_deep_admm'][-avg_window-1:-1])
                deep_admm_log_dict['deep_admm_conv'] = np.abs(avg_val_kendall -
                                        deep_admm_log_dict['val_kendall_pi_tilde_deep_admm'][-1]) \
                                        < rtol * avg_val_kendall
        # siamese update
        if not siamese_log_dict['siamese_conv'] and 'siamese' in tasks:
            siamese_log_dict['pi_tilde_siamese_prev'] = siamese_log_dict['pi_tilde_siamese']  # regressed
            siamese_log_dict['pi_tilde_siamese'], time_siamese_iter = siamese_object.train_one_epoch()
            siamese_log_dict['time_siamese'].append(time_siamese_iter)
            siamese_log_dict['iter_siamese'] += 1
            siamese_log_dict['obj_siamese'].append(objective(np.exp(siamese_log_dict['pi_tilde_siamese']), rankings_train))
            siamese_log_dict['val_acc_siamese'].append(top1_test_accuracy(siamese_log_dict['pi_tilde_siamese'], rankings_val))
            siamese_log_dict['val_kendall_siamese'].append(kendall_tau_test(siamese_log_dict['pi_tilde_siamese'], rankings_val))
            siamese_log_dict['test_acc_siamese'].append(top1_test_accuracy(siamese_log_dict['pi_tilde_siamese'], rankings_test))
            siamese_log_dict['test_kendall_siamese'].append(kendall_tau_test(siamese_log_dict['pi_tilde_siamese'], rankings_test))
            # Correct time scale
            siamese_log_dict['time_cont_siamese'].append(sum(siamese_log_dict['time_siamese']))
            # Check convergence
            if ind_iter > avg_window:
                avg_val_kendall = np.mean(siamese_log_dict['val_kendall_siamese'][-avg_window - 1:-1])
                siamese_log_dict['siamese_conv'] = np.abs(avg_val_kendall - siamese_log_dict['val_kendall_siamese'][-1]) \
                                        < rtol * avg_val_kendall
        # linear admm update
        if not lin_admm_log_dict['lin_admm_conv'] and 'lin_admm' in tasks:
            lin_admm_log_dict['pi_lin_admm_prev'] = lin_admm_log_dict['pi_lin_admm']
            lin_admm_log_dict['pi_tilde_lin_admm_prev'] = lin_admm_log_dict['pi_tilde_lin_admm']
            lin_admm_log_dict['pi_lin_admm'], lin_admm_log_dict['pi_tilde_lin_admm'], lin_admm_log_dict['u_lin_admm'], \
                        time_lin_admm_iter = lin_admm_object.fit_lin(rho=1, weights=lin_admm_log_dict['pi_lin_admm'],
                                                                     u=lin_admm_log_dict['u_lin_admm'])
            lin_admm_log_dict['time_lin_admm'].append(time_lin_admm_iter)
            lin_admm_log_dict['obj_lin_admm'].append(objective(lin_admm_log_dict['pi_tilde_lin_admm'], rankings_train))
            lin_admm_log_dict['val_acc_lin_admm'].append(top1_test_accuracy(lin_admm_log_dict['pi_tilde_lin_admm'], rankings_val))
            lin_admm_log_dict['val_kendall_lin_admm'].append(kendall_tau_test(lin_admm_log_dict['pi_tilde_lin_admm'], rankings_val))
            lin_admm_log_dict['test_acc_lin_admm'].append(top1_test_accuracy(lin_admm_log_dict['pi_tilde_lin_admm'], rankings_test))
            lin_admm_log_dict['test_kendall_lin_admm'].append(kendall_tau_test(lin_admm_log_dict['pi_tilde_lin_admm'], rankings_test))
            lin_admm_log_dict['iter_lin_admm'] += 1
            # Correct time scale
            lin_admm_log_dict['time_cont_lin_admm'].append(sum(lin_admm_log_dict['time_lin_admm']))
            # Check convergence
            if ind_iter > avg_window:
                avg_val_kendall = np.mean(lin_admm_log_dict['val_kendall_lin_admm'][-avg_window - 1:-1])
                lin_admm_log_dict['lin_admm_conv'] = np.abs(avg_val_kendall - lin_admm_log_dict['val_kendall_lin_admm'][-1]) \
                                       < rtol * avg_val_kendall
        # linear kl admm update
        if not lin_kl_admm_log_dict['lin_kl_admm_conv'] and 'lin_kl_admm' in tasks:
            lin_kl_admm_log_dict['pi_lin_kl_admm_prev'] = lin_kl_admm_log_dict['pi_lin_kl_admm']
            lin_kl_admm_log_dict['pi_tilde_lin_kl_admm_prev'] = lin_kl_admm_log_dict['pi_tilde_lin_kl_admm']
            lin_kl_admm_log_dict['pi_lin_kl_admm'], lin_kl_admm_log_dict['pi_tilde_lin_kl_admm'], \
                        lin_kl_admm_log_dict['u_lin_kl_admm'], time_lin_kl_admm_iter = \
                        lin_kl_admm_object.fit_lin(rho=1, weights=lin_kl_admm_log_dict['pi_lin_kl_admm'],
                        u=lin_kl_admm_log_dict['u_lin_kl_admm'])
            lin_kl_admm_log_dict['time_lin_kl_admm'].append(time_lin_kl_admm_iter)
            lin_kl_admm_log_dict['obj_lin_kl_admm'].append(
                objective(lin_kl_admm_log_dict['pi_tilde_lin_kl_admm'], rankings_train))
            lin_kl_admm_log_dict['val_acc_lin_kl_admm'].append(
                top1_test_accuracy(lin_kl_admm_log_dict['pi_tilde_lin_kl_admm'], rankings_val))
            lin_kl_admm_log_dict['val_kendall_lin_kl_admm'].append(
                kendall_tau_test(lin_kl_admm_log_dict['pi_tilde_lin_kl_admm'], rankings_val))
            lin_kl_admm_log_dict['test_acc_lin_kl_admm'].append(
                top1_test_accuracy(lin_kl_admm_log_dict['pi_tilde_lin_kl_admm'], rankings_test))
            lin_kl_admm_log_dict['test_kendall_lin_kl_admm'].append(
                kendall_tau_test(lin_kl_admm_log_dict['pi_tilde_lin_kl_admm'], rankings_test))
            lin_kl_admm_log_dict['iter_lin_kl_admm'] += 1
            # Correct time scale
            lin_kl_admm_log_dict['time_cont_lin_kl_admm'].append(sum(lin_kl_admm_log_dict['time_lin_kl_admm']))
            # Check convergence
            if ind_iter > avg_window:
                avg_val_kendall = np.mean(lin_kl_admm_log_dict['val_kendall_lin_kl_admm'][-avg_window - 1:-1])
                lin_kl_admm_log_dict['lin_kl_admm_conv'] = np.abs(
                    avg_val_kendall - lin_kl_admm_log_dict['val_kendall_lin_kl_admm'][-1]) \
                                                     < rtol * avg_val_kendall
        # deep l2 admm update
        if not deep_l2_admm_log_dict['deep_l2_admm_conv'] and 'deep_l2_admm' in tasks:
            deep_l2_admm_log_dict['pi_deep_l2_admm_prev'] = deep_l2_admm_log_dict['pi_deep_l2_admm']  # spectral prediction
            deep_l2_admm_log_dict['pi_tilde_deep_l2_admm_prev'] = deep_l2_admm_log_dict['pi_tilde_deep_l2_admm']  # regressed
            # Adapt rho
            r = deep_l2_admm_log_dict['prim_feas_deep_l2_admm'][-1]
            s = deep_l2_admm_log_dict['dual_feas_deep_l2_admm'][-1]
            if r > mu * s:
                rho_deep_l2_admm *= tau
            elif s > mu * r:
                rho_deep_l2_admm /= tau
            # Update
            deep_l2_admm_log_dict['pi_deep_l2_admm'], deep_l2_admm_log_dict['pi_tilde_deep_l2_admm'], deep_l2_admm_log_dict[
                        'u_deep_l2_admm'], ilsrx_time_deep_l2_admm_iter, nn_time_deep_l2_admm_iter, u_time_deep_l2_admm_iter = \
                        deep_l2_admm_object.fit_deep(rho_deep_l2_admm, weights=deep_l2_admm_log_dict['pi_deep_l2_admm'],
                        x_beta_b=deep_l2_admm_log_dict['pi_tilde_deep_l2_admm'], u=deep_l2_admm_log_dict['u_deep_l2_admm'],
                        gamma=gamma, max_inner_iter=max_inner_iter)
            time_deep_l2_admm_iter = ilsrx_time_deep_l2_admm_iter + nn_time_deep_l2_admm_iter + u_time_deep_l2_admm_iter
            deep_l2_admm_log_dict['time_deep_l2_admm'].append(time_deep_l2_admm_iter)
            deep_l2_admm_log_dict['iter_deep_l2_admm'] += 1
            deep_l2_admm_log_dict['prim_feas_deep_l2_admm'].append(np.linalg.norm(
                        deep_l2_admm_log_dict['pi_tilde_deep_l2_admm'] - deep_l2_admm_log_dict['pi_deep_l2_admm']))
            deep_l2_admm_log_dict['dual_feas_deep_l2_admm'].append(np.linalg.norm(deep_l2_admm_log_dict['pi_tilde_deep_l2_admm_prev'] -
                        deep_l2_admm_log_dict['pi_tilde_deep_l2_admm']))
            deep_l2_admm_log_dict['obj_pi_tilde_deep_l2_admm'].append(objective(deep_l2_admm_log_dict['pi_tilde_deep_l2_admm'], rankings_train))
            deep_l2_admm_log_dict['obj_pi_deep_l2_admm'].append(objective(deep_l2_admm_log_dict['pi_deep_l2_admm'], rankings_train))
            deep_l2_admm_log_dict['val_acc_pi_tilde_deep_l2_admm'].append(top1_test_accuracy(deep_l2_admm_log_dict['pi_tilde_deep_l2_admm'], rankings_val))
            deep_l2_admm_log_dict['val_acc_pi_deep_l2_admm'].append(top1_test_accuracy(deep_l2_admm_log_dict['pi_deep_l2_admm'], rankings_val))
            deep_l2_admm_log_dict['val_kendall_pi_tilde_deep_l2_admm'].append(kendall_tau_test(deep_l2_admm_log_dict['pi_tilde_deep_l2_admm'], rankings_val))
            deep_l2_admm_log_dict['val_kendall_pi_deep_l2_admm'].append(kendall_tau_test(deep_l2_admm_log_dict['pi_deep_l2_admm'], rankings_val))
            deep_l2_admm_log_dict['test_acc_pi_tilde_deep_l2_admm'].append(top1_test_accuracy(deep_l2_admm_log_dict['pi_tilde_deep_l2_admm'], rankings_test))
            deep_l2_admm_log_dict['test_acc_pi_deep_l2_admm'].append(top1_test_accuracy(deep_l2_admm_log_dict['pi_deep_l2_admm'], rankings_test))
            deep_l2_admm_log_dict['test_kendall_pi_tilde_deep_l2_admm'].append(kendall_tau_test(deep_l2_admm_log_dict['pi_tilde_deep_l2_admm'], rankings_test))
            deep_l2_admm_log_dict['test_kendall_pi_deep_l2_admm'].append(kendall_tau_test(deep_l2_admm_log_dict['pi_deep_l2_admm'], rankings_test))
            # Correct time scale
            deep_l2_admm_log_dict['time_cont_deep_l2_admm'].append(sum(deep_l2_admm_log_dict['time_deep_l2_admm']))
            # Check convergence
            if ind_iter > avg_window:
                avg_val_kendall = np.mean(deep_l2_admm_log_dict['val_kendall_pi_tilde_deep_l2_admm'][-avg_window - 1:-1])
                deep_l2_admm_log_dict['deep_l2_admm_conv'] = np.abs(avg_val_kendall -
                                    deep_l2_admm_log_dict['val_kendall_pi_tilde_deep_l2_admm'][-1]) \
                                    < rtol * avg_val_kendall
        # Plot and save the results so far for current set of hyperparameters
        if ind_iter % avg_window == 0:
            if plot_results:
                plot_convergence(deep_admm_log_dict, siamese_log_dict, deep_l2_admm_log_dict, tasks, save_name, dir)
            if "deep_admm" in tasks:
                with open('../results/' + dir + 'fig/deep_admm_' + save_name + '.pickle', "wb") as pickle_out:
                    pickle.dump(deep_admm_log_dict, pickle_out)
            if "siamese" in tasks:
                with open('../results/' + dir + 'fig/siamese_' + save_name + '.pickle', "wb") as pickle_out:
                    pickle.dump(siamese_log_dict, pickle_out)
            if "lin_admm" in tasks:
                with open('../results/' + dir + 'fig/lin_admm_' + save_name + '.pickle', "wb") as pickle_out:
                    pickle.dump(lin_admm_log_dict, pickle_out)
            if "lin_kl_admm" in tasks:
                with open('../results/' + dir + 'fig/lin_kl_admm_' + save_name + '.pickle', "wb") as pickle_out:
                    pickle.dump(lin_kl_admm_log_dict, pickle_out)
            if "deep_l2_admm" in tasks:
                with open('../results/' + dir + 'fig/deep_l2_admm_' + save_name + '.pickle', "wb") as pickle_out:
                    pickle.dump(deep_l2_admm_log_dict, pickle_out)
        print("\n ********* Outer iteration", ind_iter)
    # Plot and save results for best models wrt validation performance
    metric_and_CI(tasks, dir, val_fold, no_of_layers, plot_results)


def plot_convergence(deep_admm_log_dict, siamese_log_dict, deep_l2_admm_log_dict, tasks, save_name, dir):
    linestyle = {"linewidth": 2, "markeredgewidth": 2}
    max_time = 2 * np.max(deep_admm_log_dict['time_cont_deep_admm'])
    # Rho
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Time (s)', fontsize=28)
    plt.grid()
    plt.plot(deep_admm_log_dict['time_cont_deep_admm'], deep_admm_log_dict['rho_deep_admm'])
    ax.tick_params(labelsize='large')
    plt.tight_layout()
    plt.savefig('../results/' + dir + 'fig/rho_' + save_name + '.pdf')
    plt.rcParams.update({'font.size': 18})
    plt.close()
    #################################################
    # Objective
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('Log Likelihood', fontsize=28)
    ax.set_xlabel('Time (s)', fontsize=28)
    plt.grid()
    legend_list = ['DSR-KL $\\tilde{\pi}$', 'DSR-KL $\pi$']
    plt.plot(deep_admm_log_dict['time_cont_deep_admm'], deep_admm_log_dict['obj_pi_tilde_deep_admm'], color='g', marker='o', **linestyle)
    plt.plot(deep_admm_log_dict['time_cont_deep_admm'], deep_admm_log_dict['obj_pi_deep_admm'], color='b', marker='>', **linestyle)
    if 'siamese' in tasks:
        plt.plot(siamese_log_dict['time_cont_siamese'], siamese_log_dict['obj_siamese'], color='k', marker='+', **linestyle)
        legend_list.append("Siamese")
        # Time limit for plotting
        plt.xlim(left=0, right=max_time)
    if 'deep_l2_admm' in tasks:
        plt.plot(deep_l2_admm_log_dict['time_cont_deep_l2_admm'], deep_l2_admm_log_dict['obj_pi_tilde_deep_l2_admm'], color='m', marker='<', **linestyle)
        legend_list.append("DSR-l2 $\\tilde{\pi}$")
    ax.tick_params(labelsize='large')
    ax.legend(legend_list, prop={'size': 14})
    plt.tight_layout()
    plt.savefig('../results/' + dir + 'fig/obj_' + save_name + '.pdf')
    plt.rcParams.update({'font.size': 18})
    plt.close()
    # Validation
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('Validation KT', fontsize=28)
    ax.set_xlabel('Time (s)', fontsize=28)
    plt.grid()
    legend_list = ['DSR-KL $\\tilde{\pi}$']
    plt.plot(deep_admm_log_dict['time_cont_deep_admm'], deep_admm_log_dict['val_kendall_pi_tilde_deep_admm'], color='g', marker='o', **linestyle)
    if "par" not in dir:
        plt.plot(deep_admm_log_dict['time_cont_deep_admm'], deep_admm_log_dict['val_kendall_pi_deep_admm'], color='b', marker='>', **linestyle)
        legend_list.append('DSR-KL $\pi$')
    if 'siamese' in tasks:
        plt.plot(siamese_log_dict['time_cont_siamese'], siamese_log_dict['val_kendall_siamese'], color='k', marker='+', **linestyle)
        # Time limit for plotting
        plt.xlim(left=0, right=max_time)
        legend_list.append("Siamese")
    if 'deep_l2_admm' in tasks:
        plt.plot(deep_l2_admm_log_dict['time_cont_deep_l2_admm'], deep_l2_admm_log_dict['val_kendall_pi_tilde_deep_l2_admm'], color='m', marker='<', **linestyle)
        legend_list.append("DSR-l2 $\\tilde{\pi}$")
    ax.tick_params(labelsize='large')
    ax.legend(legend_list, prop={'size': 14})
    plt.tight_layout()
    plt.savefig('../results/' + dir + 'fig/val_kendall_' + save_name + '.pdf')
    plt.rcParams.update({'font.size': 18})
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('Validation Acc.', fontsize=28)
    ax.set_xlabel('Time (s)', fontsize=28)
    plt.grid()
    plt.plot(deep_admm_log_dict['time_cont_deep_admm'], deep_admm_log_dict['val_acc_pi_tilde_deep_admm'], color='g', marker='o', **linestyle)
    if "par" not in dir:
        plt.plot(deep_admm_log_dict['time_cont_deep_admm'], deep_admm_log_dict['val_acc_pi_deep_admm'], color='b', marker='>', **linestyle)
    if 'siamese' in tasks:
        plt.plot(siamese_log_dict['time_cont_siamese'], siamese_log_dict['val_acc_siamese'], color='k', marker='+', **linestyle)
        # Time limit for plotting
        plt.xlim(left=0, right=max_time)
    if 'deep_l2_admm' in tasks:
        plt.plot(deep_l2_admm_log_dict['time_cont_deep_l2_admm'], deep_l2_admm_log_dict['val_acc_pi_tilde_deep_l2_admm'], color='m', marker='<', **linestyle)
    ax.tick_params(labelsize='large')
    ax.legend(legend_list, prop={'size': 14})
    plt.tight_layout()
    plt.savefig('../results/' + dir + 'fig/val_acc_' + save_name + '.pdf')
    plt.rcParams.update({'font.size': 18})
    plt.close()


def metric_and_CI(tasks, dir, val_fold, no_of_layers, plot_results):
    '''
    :param tasks: list of algorithms to be evaluated.
    :param dir: directory of log files
    '''
    # find all pickle files for the current fold and architecture
    results_dir = '../results/' + dir + 'fig/'
    final_name = str(val_fold) + '_no_of_layers_' + str(no_of_layers)
    all_files = listdir(results_dir)
    deep_admm_pickle_files = list(filter(lambda x: x[-7:] == '.pickle' and "deep_admm_" + final_name in x, all_files))
    siamese_pickle_files = list(filter(lambda x: x[-7:] == '.pickle' and "siamese_" + final_name in x, all_files))
    lin_admm_pickle_files = list(filter(lambda x: x[-7:] == '.pickle' and "lin_admm_" + final_name in x, all_files))
    lin_kl_admm_pickle_files = list(filter(lambda x: x[-7:] == '.pickle' and "lin_kl_admm_" + final_name in x, all_files))
    deep_l2_admm_pickle_files = list(filter(lambda x: x[-7:] == '.pickle' and "deep_l2_admm_" + final_name in x, all_files))
    # get validation results at convergence
    val_results_dict = dict()
    val_results_dict['val_acc_pi_tilde_deep_admm'] = []
    val_results_dict['val_acc_siamese'] = []
    val_results_dict['val_acc_pi_tilde_deep_l2_admm'] = []
    val_results_dict['val_kendall_pi_tilde_deep_admm'] = []
    val_results_dict['val_kendall_siamese'] = []
    val_results_dict['val_kendall_pi_tilde_deep_l2_admm'] = []
    for pickle_file in deep_admm_pickle_files:
        with open(results_dir + pickle_file, mode='rb') as pickle_in:
            log_dict = pickle.load(pickle_in)
        val_results_dict['val_acc_pi_tilde_deep_admm'].append(log_dict['val_acc_pi_tilde_deep_admm'][-1])
        val_results_dict['val_kendall_pi_tilde_deep_admm'].append(log_dict['val_kendall_pi_tilde_deep_admm'][-1])
    for pickle_file in siamese_pickle_files:
        with open(results_dir + pickle_file, mode='rb') as pickle_in:
            log_dict = pickle.load(pickle_in)
        val_results_dict['val_acc_siamese'].append(log_dict['val_acc_siamese'][-1])
        val_results_dict['val_kendall_siamese'].append(log_dict['val_kendall_siamese'][-1])
    for pickle_file in deep_l2_admm_pickle_files:
        with open(results_dir + pickle_file, mode='rb') as pickle_in:
            log_dict = pickle.load(pickle_in)
        val_results_dict['val_acc_pi_tilde_deep_l2_admm'].append(log_dict['val_acc_pi_tilde_deep_l2_admm'][-1])
        val_results_dict['val_kendall_pi_tilde_deep_l2_admm'].append(log_dict['val_kendall_pi_tilde_deep_l2_admm'][-1])
    # find best validation accuracy for each method and save corresponding results
    save_final_name = "best_acc_models_" + final_name
    with open(results_dir + save_final_name + '.csv', "w") as infile:
        w = csv.writer(infile)
        if 'deep_admm' in tasks:
            best_pickle_file = deep_admm_pickle_files[np.argmax(val_results_dict['val_acc_pi_tilde_deep_admm'])]
            with open(results_dir + best_pickle_file, mode='rb') as pickle_in:
                deep_admm_log_dict = pickle.load(pickle_in)
            w.writerow(['Best deep admm model: ' + best_pickle_file])
            w.writerow(['time_cont_deep_admm: ' + str(deep_admm_log_dict['time_cont_deep_admm'][-1])])
            w.writerow(['iter_deep_admm: ' + str(deep_admm_log_dict['iter_deep_admm'])])
            w.writerow(['val_acc_pi_tilde_deep_admm: ' + str(deep_admm_log_dict['val_acc_pi_tilde_deep_admm'][-1])])
            w.writerow(['val_kendall_pi_tilde_deep_admm: ' + str(deep_admm_log_dict['val_kendall_pi_tilde_deep_admm'][-1])])
            w.writerow(['test_acc_pi_tilde_deep_admm: ' + str(deep_admm_log_dict['test_acc_pi_tilde_deep_admm'][-1])])
            w.writerow(['test_kendall_pi_tilde_deep_admm: ' + str(deep_admm_log_dict['test_kendall_pi_tilde_deep_admm'][-1])])
            w.writerow(['obj_pi_tilde_deep_admm: ' + str(deep_admm_log_dict['obj_pi_tilde_deep_admm'][-1])])
            w.writerow([""])
        else:
            deep_admm_log_dict = None
        if 'siamese' in tasks:
            best_pickle_file = siamese_pickle_files[np.argmax(val_results_dict['val_acc_siamese'])]
            with open(results_dir + best_pickle_file, mode='rb') as pickle_in:
                siamese_log_dict = pickle.load(pickle_in)
            w.writerow(['Best siamese model: ' + best_pickle_file])
            w.writerow(['time_cont_siamese: ' + str(siamese_log_dict['time_cont_siamese'][-1])])
            w.writerow(['iter_siamese: ' + str(siamese_log_dict['iter_siamese'])])
            w.writerow(['val_acc_siamese: ' + str(siamese_log_dict['val_acc_siamese'][-1])])
            w.writerow(['val_kendall_siamese: ' + str(siamese_log_dict['val_kendall_siamese'][-1])])
            w.writerow(['test_acc_siamese: ' + str(siamese_log_dict['test_acc_siamese'][-1])])
            w.writerow(['test_kendall_siamese: ' + str(siamese_log_dict['test_kendall_siamese'][-1])])
            w.writerow(['obj_siamese: ' + str(siamese_log_dict['obj_siamese'][-1])])
            w.writerow([""])
        else:
            siamese_log_dict = None
        if 'lin_admm' in tasks:
            with open(results_dir + lin_admm_pickle_files[0], mode='rb') as pickle_in:
                lin_admm_log_dict = pickle.load(pickle_in)
            w.writerow(['time_cont_lin_admm: ' + str(lin_admm_log_dict['time_cont_lin_admm'][-1])])
            w.writerow(['iter_lin_admm: ' + str(lin_admm_log_dict['iter_lin_admm'])])
            w.writerow(['test_acc_lin_admm: ' + str(lin_admm_log_dict['test_acc_lin_admm'][-1])])
            w.writerow(['test_kendall_lin_admm: ' + str(lin_admm_log_dict['test_kendall_lin_admm'][-1])])
            w.writerow(['obj_lin_admm: ' + str(lin_admm_log_dict['obj_lin_admm'][-1])])
            w.writerow([""])
        else:
            lin_admm_log_dict = None
        if 'lin_kl_admm' in tasks:
            with open(results_dir + lin_kl_admm_pickle_files[0], mode='rb') as pickle_in:
                lin_kl_admm_log_dict = pickle.load(pickle_in)
            w.writerow(['time_cont_lin_kl_admm: ' + str(lin_kl_admm_log_dict['time_cont_lin_kl_admm'][-1])])
            w.writerow(['iter_lin_kl_admm: ' + str(lin_kl_admm_log_dict['iter_lin_kl_admm'])])
            w.writerow(['test_acc_lin_kl_admm: ' + str(lin_kl_admm_log_dict['test_acc_lin_kl_admm'][-1])])
            w.writerow(['test_kendall_lin_kl_admm: ' + str(lin_kl_admm_log_dict['test_kendall_lin_kl_admm'][-1])])
            w.writerow(['obj_lin_kl_admm: ' + str(lin_kl_admm_log_dict['obj_lin_kl_admm'][-1])])
            w.writerow([""])
        else:
            lin_kl_admm_log_dict = None
        if 'deep_l2_admm' in tasks:
            best_pickle_file = deep_l2_admm_pickle_files[np.argmax(val_results_dict['val_acc_pi_tilde_deep_l2_admm'])]
            with open(results_dir + best_pickle_file, mode='rb') as pickle_in:
                deep_l2_admm_log_dict = pickle.load(pickle_in)
            w.writerow(['Best deep admm model: ' + best_pickle_file])
            w.writerow(['time_cont_deep_l2_admm: ' + str(deep_l2_admm_log_dict['time_cont_deep_l2_admm'][-1])])
            w.writerow(['iter_deep_l2_admm: ' + str(deep_l2_admm_log_dict['iter_deep_l2_admm'])])
            w.writerow(['val_acc_pi_tilde_deep_l2_admm: ' + str(deep_l2_admm_log_dict['val_acc_pi_tilde_deep_l2_admm'][-1])])
            w.writerow(['val_kendall_pi_tilde_deep_l2_admm: ' + str(deep_l2_admm_log_dict['val_kendall_pi_tilde_deep_l2_admm'][-1])])
            w.writerow(['test_acc_pi_tilde_deep_l2_admm: ' + str(deep_l2_admm_log_dict['test_acc_pi_tilde_deep_l2_admm'][-1])])
            w.writerow(['test_kendall_pi_tilde_deep_l2_admm: ' + str(deep_l2_admm_log_dict['test_kendall_pi_tilde_deep_l2_admm'][-1])])
            w.writerow(['obj_pi_tilde_deep_l2_admm: ' + str(deep_l2_admm_log_dict['obj_pi_tilde_deep_l2_admm'][-1])])
        else:
            deep_l2_admm_log_dict = None
    # plot convergence for best model for each method
    if plot_results:
        plot_convergence(deep_admm_log_dict, siamese_log_dict, deep_l2_admm_log_dict, tasks, save_final_name, dir)
    # find best validation KT for each method and save corresponding results
    save_final_name = "best_kendall_models_" + final_name
    with open(results_dir + save_final_name + '.csv', "w") as infile:
        w = csv.writer(infile)
        if 'deep_admm' in tasks:
            best_pickle_file = deep_admm_pickle_files[np.argmax(val_results_dict['val_kendall_pi_tilde_deep_admm'])]
            with open(results_dir + best_pickle_file, mode='rb') as pickle_in:
                deep_admm_log_dict = pickle.load(pickle_in)
            w.writerow(['Best deep admm model: ' + best_pickle_file])
            w.writerow(['time_cont_deep_admm: ' + str(deep_admm_log_dict['time_cont_deep_admm'][-1])])
            w.writerow(['iter_deep_admm: ' + str(deep_admm_log_dict['iter_deep_admm'])])
            w.writerow(['val_acc_pi_tilde_deep_admm: ' + str(deep_admm_log_dict['val_acc_pi_tilde_deep_admm'][-1])])
            w.writerow(['val_kendall_pi_tilde_deep_admm: ' + str(deep_admm_log_dict['val_kendall_pi_tilde_deep_admm'][-1])])
            w.writerow(['test_acc_pi_tilde_deep_admm: ' + str(deep_admm_log_dict['test_acc_pi_tilde_deep_admm'][-1])])
            w.writerow(['test_kendall_pi_tilde_deep_admm: ' + str(deep_admm_log_dict['test_kendall_pi_tilde_deep_admm'][-1])])
            w.writerow(['obj_pi_tilde_deep_admm: ' + str(deep_admm_log_dict['obj_pi_tilde_deep_admm'][-1])])
            w.writerow([""])
        else:
            deep_admm_log_dict = None
        if 'siamese' in tasks:
            best_pickle_file = siamese_pickle_files[np.argmax(val_results_dict['val_kendall_siamese'])]
            with open(results_dir + best_pickle_file, mode='rb') as pickle_in:
                siamese_log_dict = pickle.load(pickle_in)
            w.writerow(['Best siamese model: ' + best_pickle_file])
            w.writerow(['time_cont_siamese: ' + str(siamese_log_dict['time_cont_siamese'][-1])])
            w.writerow(['iter_siamese: ' + str(siamese_log_dict['iter_siamese'])])
            w.writerow(['val_acc_siamese: ' + str(siamese_log_dict['val_acc_siamese'][-1])])
            w.writerow(['val_kendall_siamese: ' + str(siamese_log_dict['val_kendall_siamese'][-1])])
            w.writerow(['test_acc_siamese: ' + str(siamese_log_dict['test_acc_siamese'][-1])])
            w.writerow(['test_kendall_siamese: ' + str(siamese_log_dict['test_kendall_siamese'][-1])])
            w.writerow(['obj_siamese: ' + str(siamese_log_dict['obj_siamese'][-1])])
            w.writerow([""])
        else:
            siamese_log_dict = None
        if 'lin_admm' in tasks:
            with open(results_dir + lin_admm_pickle_files[0], mode='rb') as pickle_in:
                lin_admm_log_dict = pickle.load(pickle_in)
            w.writerow(['time_cont_lin_admm: ' + str(lin_admm_log_dict['time_cont_lin_admm'][-1])])
            w.writerow(['iter_lin_admm: ' + str(lin_admm_log_dict['iter_lin_admm'])])
            w.writerow(['test_acc_lin_admm: ' + str(lin_admm_log_dict['test_acc_lin_admm'][-1])])
            w.writerow(['test_kendall_lin_admm: ' + str(lin_admm_log_dict['test_kendall_lin_admm'][-1])])
            w.writerow(['obj_lin_admm: ' + str(lin_admm_log_dict['obj_lin_admm'][-1])])
            w.writerow([""])
        else:
            lin_admm_log_dict = None
        if 'lin_kl_admm' in tasks:
            with open(results_dir + lin_kl_admm_pickle_files[0], mode='rb') as pickle_in:
                lin_kl_admm_log_dict = pickle.load(pickle_in)
            w.writerow(['time_cont_lin_kl_admm: ' + str(lin_kl_admm_log_dict['time_cont_lin_kl_admm'][-1])])
            w.writerow(['iter_lin_kl_admm: ' + str(lin_kl_admm_log_dict['iter_lin_kl_admm'])])
            w.writerow(['test_acc_lin_kl_admm: ' + str(lin_kl_admm_log_dict['test_acc_lin_kl_admm'][-1])])
            w.writerow(['test_kendall_lin_kl_admm: ' + str(lin_kl_admm_log_dict['test_kendall_lin_kl_admm'][-1])])
            w.writerow(['obj_lin_kl_admm: ' + str(lin_kl_admm_log_dict['obj_lin_kl_admm'][-1])])
            w.writerow([""])
        else:
            lin_kl_admm_log_dict = None
        if 'deep_l2_admm' in tasks:
            best_pickle_file = deep_l2_admm_pickle_files[np.argmax(val_results_dict['val_kendall_pi_tilde_deep_l2_admm'])]
            with open(results_dir + best_pickle_file, mode='rb') as pickle_in:
                deep_l2_admm_log_dict = pickle.load(pickle_in)
            w.writerow(['Best deep admm model: ' + best_pickle_file])
            w.writerow(['time_cont_deep_l2_admm: ' + str(deep_l2_admm_log_dict['time_cont_deep_l2_admm'][-1])])
            w.writerow(['iter_deep_l2_admm: ' + str(deep_l2_admm_log_dict['iter_deep_l2_admm'])])
            w.writerow(['val_acc_pi_tilde_deep_l2_admm: ' + str(deep_l2_admm_log_dict['val_acc_pi_tilde_deep_l2_admm'][-1])])
            w.writerow(['val_kendall_pi_tilde_deep_l2_admm: ' + str(deep_l2_admm_log_dict['val_kendall_pi_tilde_deep_l2_admm'][-1])])
            w.writerow(['test_acc_pi_tilde_deep_l2_admm: ' + str(deep_l2_admm_log_dict['test_acc_pi_tilde_deep_l2_admm'][-1])])
            w.writerow(['test_kendall_pi_tilde_deep_l2_admm: ' + str(deep_l2_admm_log_dict['test_kendall_pi_tilde_deep_l2_admm'][-1])])
            w.writerow(['obj_pi_tilde_deep_l2_admm: ' + str(deep_l2_admm_log_dict['obj_pi_tilde_deep_l2_admm'][-1])])
        else:
            deep_l2_admm_log_dict = None
    # plot convergence for best model for each method
    if plot_results:
        plot_convergence(deep_admm_log_dict, siamese_log_dict, deep_l2_admm_log_dict, tasks, save_final_name, dir)


def metric_and_CI_avg(tasks, dir):
    '''
    :param tasks: list of algorithms to be evaluated.
    :param dir: directory of log files
    '''
    results_dir = '../results/' + dir + 'fig/'
    all_files = listdir(results_dir)
    val_results_dict = dict()
    lamda_list = [0.0002, 0.002, 0.02]
    lr_list = [1e-3, 1e-4, 1e-5]
    hyperparam_combinations = list(product(lamda_list, lr_list))
    # get validation results at convergence
    val_results_dict['val_kendall_pi_tilde_deep_admm'] = np.zeros((n_fold, len(hyperparam_combinations)))
    val_results_dict['val_kendall_siamese'] = np.zeros((n_fold, len(hyperparam_combinations)))
    val_results_dict['val_kendall_pi_tilde_deep_l2_admm'] = np.zeros((n_fold, len(hyperparam_combinations)))
    # for each val_fold on axis 0, log convergence result for each set of hyperparameters on axis 1
    for val_fold in range(n_fold):
        for hyperparam_index, (lamda, lr) in enumerate(hyperparam_combinations):
            save_name = str(val_fold) + '_lambda_' + str(lamda) + '_lr_' + str(lr) + ".pickle"
            # if exists log the result. then take the fold mean over only the nonzero ones
            pickle_file = "deep_admm_" + save_name
            if pickle_file in all_files:
                with open(results_dir + pickle_file, mode='rb') as pickle_in:
                    log_dict = pickle.load(pickle_in)
                val_results_dict['val_kendall_pi_tilde_deep_admm'][val_fold, hyperparam_index] = log_dict['val_kendall_pi_tilde_deep_admm'][-1]
            pickle_file = "siamese_" + save_name
            if pickle_file in all_files:
                with open(results_dir + pickle_file, mode='rb') as pickle_in:
                    log_dict = pickle.load(pickle_in)
                val_results_dict['val_kendall_siamese'][val_fold, hyperparam_index] = log_dict['val_kendall_siamese'][-1]
            pickle_file = "deep_l2_admm_" + save_name
            if pickle_file in all_files:
                with open(results_dir + pickle_file, mode='rb') as pickle_in:
                    log_dict = pickle.load(pickle_in)
                val_results_dict['val_kendall_pi_tilde_deep_l2_admm'][val_fold, hyperparam_index] = log_dict['val_kendall_pi_tilde_deep_l2_admm'][-1]
    # average validation results over folds for each hyperparameter setting
    for key, lst_all_params in val_results_dict.items():
        lst_avg = []
        for hyperparam_index in range(len(hyperparam_combinations)):
            lst_cur_param = lst_all_params[:, hyperparam_index]
            lst_avg.append(np.mean(lst_cur_param[lst_cur_param != 0]))
        val_results_dict[key] = lst_avg
    # get validation results at convergence
    deep_admm_log_dict = dict()
    siamese_log_dict = dict()
    lin_admm_log_dict = dict()
    deep_l2_admm_log_dict = dict()
    deep_admm_log_dict['time_cont_deep_admm'] = []
    deep_admm_log_dict['iter_deep_admm'] = []
    deep_admm_log_dict['val_acc_pi_tilde_deep_admm'] = []
    deep_admm_log_dict['val_acc_pi_deep_admm'] = []
    deep_admm_log_dict['val_kendall_pi_tilde_deep_admm'] = []
    deep_admm_log_dict['val_kendall_pi_deep_admm'] = []
    deep_admm_log_dict['test_acc_pi_tilde_deep_admm'] = []
    deep_admm_log_dict['test_kendall_pi_tilde_deep_admm'] = []
    deep_admm_log_dict['obj_pi_tilde_deep_admm'] = []
    deep_admm_log_dict['obj_pi_deep_admm'] = []
    siamese_log_dict['time_cont_siamese'] = []
    siamese_log_dict['iter_siamese'] = []
    siamese_log_dict['val_acc_siamese'] = []
    siamese_log_dict['val_kendall_siamese'] = []
    siamese_log_dict['test_acc_siamese'] = []
    siamese_log_dict['test_kendall_siamese'] = []
    siamese_log_dict['obj_siamese'] = []
    lin_admm_log_dict['time_cont_lin_admm'] = []
    lin_admm_log_dict['iter_lin_admm'] = []
    lin_admm_log_dict['val_acc_lin_admm'] = []
    lin_admm_log_dict['val_kendall_lin_admm'] = []
    lin_admm_log_dict['test_acc_lin_admm'] = []
    lin_admm_log_dict['test_kendall_lin_admm'] = []
    lin_admm_log_dict['obj_lin_admm'] = []
    deep_l2_admm_log_dict['time_cont_deep_l2_admm'] = []
    deep_l2_admm_log_dict['iter_deep_l2_admm'] = []
    deep_l2_admm_log_dict['val_acc_pi_tilde_deep_l2_admm'] = []
    deep_l2_admm_log_dict['val_kendall_pi_tilde_deep_l2_admm'] = []
    deep_l2_admm_log_dict['test_acc_pi_tilde_deep_l2_admm'] = []
    deep_l2_admm_log_dict['test_kendall_pi_tilde_deep_l2_admm'] = []
    deep_l2_admm_log_dict['obj_pi_tilde_deep_l2_admm'] = []
    # find best validation performance for each method and save corresponding results
    with open(results_dir + 'results.csv', "w") as infile:
        w = csv.writer(infile)
        if 'deep_admm' in tasks:
            best_model_index = np.argmax(val_results_dict['val_kendall_pi_tilde_deep_admm'])
            lamda, lr = hyperparam_combinations[best_model_index]
            for val_fold in range(n_fold):
                best_pickle_file = "deep_admm_" + str(val_fold) + '_lambda_' + str(lamda) + '_lr_' + str(lr) + ".pickle"
                if best_pickle_file in all_files:
                    with open(results_dir + best_pickle_file, mode='rb') as pickle_in:
                        log_dict = pickle.load(pickle_in)
                    deep_admm_log_dict['time_cont_deep_admm'].append(log_dict['time_cont_deep_admm'])
                    deep_admm_log_dict['iter_deep_admm'].append(log_dict['iter_deep_admm'])
                    deep_admm_log_dict['val_acc_pi_tilde_deep_admm'].append(log_dict['val_acc_pi_tilde_deep_admm'])
                    deep_admm_log_dict['val_acc_pi_deep_admm'].append(log_dict['val_acc_pi_deep_admm'])
                    deep_admm_log_dict['val_kendall_pi_tilde_deep_admm'].append(log_dict['val_kendall_pi_tilde_deep_admm'])
                    deep_admm_log_dict['val_kendall_pi_deep_admm'].append(log_dict['val_kendall_pi_deep_admm'])
                    deep_admm_log_dict['test_acc_pi_tilde_deep_admm'].append(log_dict['test_acc_pi_tilde_deep_admm'])
                    deep_admm_log_dict['test_kendall_pi_tilde_deep_admm'].append(log_dict['test_kendall_pi_tilde_deep_admm'])
                    deep_admm_log_dict['obj_pi_tilde_deep_admm'].append(log_dict['obj_pi_tilde_deep_admm'])
                    deep_admm_log_dict['obj_pi_deep_admm'].append(log_dict['obj_pi_deep_admm'])
            # average convergence results over folds
            for key, lst_all_params in deep_admm_log_dict.items():
                if "iter" not in key:
                    max_len = np.max([len(lst) for lst in lst_all_params])
                    lst_all_params_square = []
                    for lst in lst_all_params:
                        if len(lst) < max_len:
                            lst.extend(list(lst[-1] * np.ones((max_len-len(lst),))))
                        lst_all_params_square.append(lst)
                    deep_admm_log_dict[key] = np.mean(np.array(lst_all_params_square), axis=0)
                else:
                    deep_admm_log_dict[key] = int(np.mean(np.array(lst_all_params)))
            w.writerow(['Best deep admm model: ' + str(hyperparam_combinations[best_model_index])])
            w.writerow(['time_cont_deep_admm: ' + str(deep_admm_log_dict['time_cont_deep_admm'][-1])])
            w.writerow(['iter_deep_admm: ' + str(deep_admm_log_dict['iter_deep_admm'])])
            w.writerow(['val_acc_pi_tilde_deep_admm: ' + str(deep_admm_log_dict['val_acc_pi_tilde_deep_admm'][-1])])
            w.writerow(['val_kendall_pi_tilde_deep_admm: ' + str(deep_admm_log_dict['val_kendall_pi_tilde_deep_admm'][-1])])
            w.writerow(['test_acc_pi_tilde_deep_admm: ' + str(deep_admm_log_dict['test_acc_pi_tilde_deep_admm'][-1])])
            w.writerow(['test_kendall_pi_tilde_deep_admm: ' + str(deep_admm_log_dict['test_kendall_pi_tilde_deep_admm'][-1])])
            w.writerow(['obj_pi_tilde_deep_admm: ' + str(deep_admm_log_dict['obj_pi_tilde_deep_admm'][-1])])
        else:
            deep_admm_log_dict = None
        w.writerow([""])
        if 'siamese' in tasks:
            best_model_index = np.argmax(val_results_dict['val_kendall_siamese'])
            lamda, lr = hyperparam_combinations[best_model_index]
            for val_fold in range(n_fold):
                best_pickle_file = "siamese_" + str(val_fold) + '_lambda_' + str(lamda) + '_lr_' + str(lr) + ".pickle"
                if best_pickle_file in all_files:
                    with open(results_dir + best_pickle_file, mode='rb') as pickle_in:
                        log_dict = pickle.load(pickle_in)
                    siamese_log_dict['time_cont_siamese'].append(log_dict['time_cont_siamese'])
                    siamese_log_dict['iter_siamese'].append(log_dict['iter_siamese'])
                    siamese_log_dict['val_acc_siamese'].append(log_dict['val_acc_siamese'])
                    siamese_log_dict['val_kendall_siamese'].append(log_dict['val_kendall_siamese'])
                    siamese_log_dict['test_acc_siamese'].append(log_dict['test_acc_siamese'])
                    siamese_log_dict['test_kendall_siamese'].append(log_dict['test_kendall_siamese'])
                    siamese_log_dict['obj_siamese'].append(log_dict['obj_siamese'])
            # average convergence results over folds
            for key, lst_all_params in siamese_log_dict.items():
                if "iter" not in key:
                    max_len = np.max([len(lst) for lst in lst_all_params])
                    lst_all_params_square = []
                    for lst in lst_all_params:
                        if len(lst) < max_len:
                            lst.extend(list(lst[-1] * np.ones((max_len - len(lst),))))
                        lst_all_params_square.append(lst)
                    siamese_log_dict[key] = np.mean(np.array(lst_all_params_square), axis=0)
                else:
                    siamese_log_dict[key] = int(np.mean(np.array(lst_all_params)))
            w.writerow(['Best siamese model: ' + str(hyperparam_combinations[best_model_index])])
            w.writerow(['time_cont_siamese: ' + str(siamese_log_dict['time_cont_siamese'][-1])])
            w.writerow(['iter_siamese: ' + str(siamese_log_dict['iter_siamese'])])
            w.writerow(['val_acc_siamese: ' + str(siamese_log_dict['val_acc_siamese'][-1])])
            w.writerow(['val_kendall_siamese: ' + str(siamese_log_dict['val_kendall_siamese'][-1])])
            w.writerow(['test_acc_siamese: ' + str(siamese_log_dict['test_acc_siamese'][-1])])
            w.writerow(['test_kendall_siamese: ' + str(siamese_log_dict['test_kendall_siamese'][-1])])
            w.writerow(['obj_siamese: ' + str(siamese_log_dict['obj_siamese'][-1])])
        else:
            siamese_log_dict = None
        w.writerow([""])
        if 'lin_admm' in tasks:
            best_model_index = np.argmax(val_results_dict['val_kendall_pi_tilde_deep_admm'])
            lamda, lr = hyperparam_combinations[best_model_index]
            for val_fold in range(n_fold):
                best_pickle_file = "lin_admm_" + str(val_fold) + '_lambda_' + str(lamda) + '_lr_' + str(lr) + ".pickle"
                if best_pickle_file in all_files:
                    with open(results_dir + best_pickle_file, mode='rb') as pickle_in:
                        log_dict = pickle.load(pickle_in)
                    lin_admm_log_dict['time_cont_lin_admm'].append(log_dict['time_cont_lin_admm'])
                    lin_admm_log_dict['iter_lin_admm'].append(log_dict['iter_lin_admm'])
                    lin_admm_log_dict['val_acc_lin_admm'].append(log_dict['val_acc_lin_admm'])
                    lin_admm_log_dict['val_kendall_lin_admm'].append(log_dict['val_kendall_lin_admm'])
                    lin_admm_log_dict['test_acc_lin_admm'].append(log_dict['test_acc_lin_admm'])
                    lin_admm_log_dict['test_kendall_lin_admm'].append(log_dict['test_kendall_lin_admm'])
                    lin_admm_log_dict['obj_lin_admm'].append(log_dict['obj_lin_admm'])
            # average convergence results over folds
            for key, lst_all_params in lin_admm_log_dict.items():
                if "iter" not in key:
                    max_len = np.max([len(lst) for lst in lst_all_params])
                    lst_all_params_square = []
                    for lst in lst_all_params:
                        if len(lst) < max_len:
                            lst.extend(list(lst[-1] * np.ones((max_len - len(lst),))))
                        lst_all_params_square.append(lst)
                    lin_admm_log_dict[key] = np.mean(np.array(lst_all_params_square), axis=0)
                else:
                    lin_admm_log_dict[key] = int(np.mean(np.array(lst_all_params)))
            w.writerow(['time_cont_lin_admm: ' + str(lin_admm_log_dict['time_cont_lin_admm'][-1])])
            w.writerow(['iter_lin_admm: ' + str(lin_admm_log_dict['iter_lin_admm'])])
            w.writerow(['test_acc_lin_admm: ' + str(lin_admm_log_dict['test_acc_lin_admm'][-1])])
            w.writerow(['test_kendall_lin_admm: ' + str(lin_admm_log_dict['test_kendall_lin_admm'][-1])])
            w.writerow(['obj_lin_admm: ' + str(lin_admm_log_dict['obj_lin_admm'][-1])])
        else:
            lin_admm_log_dict = None
        w.writerow([""])
        if 'deep_l2_admm' in tasks:
            best_model_index = np.argmax(val_results_dict['val_kendall_pi_tilde_deep_l2_admm'])
            lamda, lr = hyperparam_combinations[best_model_index]
            for val_fold in range(n_fold):
                best_pickle_file = "deep_l2_admm_" + str(val_fold) + '_lambda_' + str(lamda) + '_lr_' + str(lr) + ".pickle"
                if best_pickle_file in all_files:
                    with open(results_dir + best_pickle_file, mode='rb') as pickle_in:
                        log_dict = pickle.load(pickle_in)
                    deep_l2_admm_log_dict['time_cont_deep_l2_admm'].append(log_dict['time_cont_deep_l2_admm'])
                    deep_l2_admm_log_dict['iter_deep_l2_admm'].append(log_dict['iter_deep_l2_admm'])
                    deep_l2_admm_log_dict['val_acc_pi_tilde_deep_l2_admm'].append(log_dict['val_acc_pi_tilde_deep_l2_admm'])
                    deep_l2_admm_log_dict['val_kendall_pi_tilde_deep_l2_admm'].append(log_dict['val_kendall_pi_tilde_deep_l2_admm'])
                    deep_l2_admm_log_dict['test_acc_pi_tilde_deep_l2_admm'].append(log_dict['test_acc_pi_tilde_deep_l2_admm'])
                    deep_l2_admm_log_dict['test_kendall_pi_tilde_deep_l2_admm'].append(log_dict['test_kendall_pi_tilde_deep_l2_admm'])
                    deep_l2_admm_log_dict['obj_pi_tilde_deep_l2_admm'].append(log_dict['obj_pi_tilde_deep_l2_admm'])
            # average convergence results over folds
            for key, lst_all_params in deep_l2_admm_log_dict.items():
                if "iter" not in key:
                    max_len = np.max([len(lst) for lst in lst_all_params])
                    lst_all_params_square = []
                    for lst in lst_all_params:
                        if len(lst) < max_len:
                            lst.extend(list(lst[-1] * np.ones((max_len - len(lst),))))
                        lst_all_params_square.append(lst)
                    deep_l2_admm_log_dict[key] = np.mean(np.array(lst_all_params_square), axis=0)
                else:
                    deep_l2_admm_log_dict[key] = int(np.mean(np.array(lst_all_params)))
            w.writerow(['Best deep admm model: ' + str(hyperparam_combinations[best_model_index])])
            w.writerow(['time_cont_deep_l2_admm: ' + str(deep_l2_admm_log_dict['time_cont_deep_l2_admm'][-1])])
            w.writerow(['iter_deep_l2_admm: ' + str(deep_l2_admm_log_dict['iter_deep_l2_admm'])])
            w.writerow(['val_acc_pi_tilde_deep_l2_admm: ' + str(deep_l2_admm_log_dict['val_acc_pi_tilde_deep_l2_admm'][-1])])
            w.writerow(['val_kendall_pi_tilde_deep_l2_admm: ' + str(deep_l2_admm_log_dict['val_kendall_pi_tilde_deep_l2_admm'][-1])])
            w.writerow(['test_acc_pi_tilde_deep_l2_admm: ' + str(deep_l2_admm_log_dict['test_acc_pi_tilde_deep_l2_admm'][-1])])
            w.writerow(['test_kendall_pi_tilde_deep_l2_admm: ' + str(deep_l2_admm_log_dict['test_kendall_pi_tilde_deep_l2_admm'][-1])])
            w.writerow(['obj_pi_tilde_deep_l2_admm: ' + str(deep_l2_admm_log_dict['obj_pi_tilde_deep_l2_admm'][-1])])
        else:
            deep_l2_admm_log_dict = None
    # plot convergence for best model for each method
    plot_convergence(deep_admm_log_dict, siamese_log_dict, deep_l2_admm_log_dict, tasks, "", dir)


if __name__ == "__main__":
    n_fold = 5
    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # list of algorithms to be run OR 'test' if saving metrics and CI
    parser.add_argument('tasks', help='delimited list of tasks', type=lambda s: [str(task) for task in s.split(',')])
    parser.add_argument('val_fold', type=int)
    parser.add_argument('no_of_layers', type=int)
    parser.add_argument('lamda', type=float)
    parser.add_argument('lr', type=float)
    parser.add_argument('dir', type=str)
    args = parser.parse_args()
    if "rop" in args.dir:
        basenet = "googlenet"
    else:
        basenet = "fc"
    if "deep_admm" in args.tasks or "deep_l2_admm" in args.tasks or "siamese" in args.tasks:
        plot_results = True
    else:
        plot_results = False
    # parameters for admm convergence
    max_inner_iter = n_iter
    rho_deep_admm = 1
    rho_deep_l2_admm = 1
    gamma = 1
    tau = 2
    mu = 10
    if "train" in args.tasks:
        run_save_all_methods(args.tasks, args.dir, args.val_fold, args.lamda, args.lr, basenet, args.no_of_layers,
                             max_inner_iter, rho_deep_admm, rho_deep_l2_admm, gamma, plot_results)
    else:
        # Plot and save results for best models wrt validation performance
        metric_and_CI(args.tasks, args.dir, args.val_fold, args.no_of_layers, plot_results)


