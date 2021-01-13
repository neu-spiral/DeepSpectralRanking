from __future__ import absolute_import
from __future__ import print_function

import keras.backend as K
import numpy as np
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from time import time
from googlenet_functional import *
from utils import epsilon


def PLobj(scores):
    # list of scores of ordered images, does not care about batch dimension since lambda layer
    ranking_length = len(scores)
    scores = [K.exp(score) for score in scores]
    ll = 0
    for winner_index in range(ranking_length - 1):
        winner_weight = scores[winner_index]
        sum_weights = K.sum(scores[winner_index:])
        ll += K.log(winner_weight + epsilon) - K.log(sum_weights + epsilon)
    return ll

def PLLoss(true_order, ll):
    # input image list is already ordered wrt rank
    return -ll

def BTPred(scalars):
    """
    This is the output when we predict comparison labels. s1, s2 = scalars (beta.*x)
    """
    s1 = scalars[0]
    s2 = scalars[1]
    return s1 - s2

def BTLoss(y_true, y_pred):
    """
    Negative log likelihood of Bradley-Terry Penalty, to be minimized. y = beta.*x
    y_true:-1 or 1
    y_pred:si-sj
    alpha: 0-1
    """
    exponent = K.exp(-y_true * (y_pred))
    return K.log(1 + exponent)


class comparison_siamese(object):
    """
    Training and testing the neural network with 5000 images.
    """
    def __init__(self, comp_labels, comp_imgs_lst_pair, imgs_lst,
                 reg_param=0.0002, learning_rate=1e-4,
                 init_last_layer_beta=None, save_model_name='siamese.h5', basenet="googlenet", no_of_layers=2):
        '''
        :param comp_labels: +1/-1
        :param comp_imgs_lst_pair: (2, m, 3, 224, 224) = [comp_imgs_1, comp_imgs_2]
        :param input_shape: (3, 224, 224)
        :param reg_param: regularization for network layers
        :param n: number of samples
        '''
        self.comp_imgs_lst_pair_left = comp_imgs_lst_pair[0]
        self.comp_imgs_lst_pair_right = comp_imgs_lst_pair[1]
        self.comp_labels = comp_labels
        self.imgs_lst = imgs_lst
        self.input_shape = imgs_lst.shape[1:]
        self.reg_param = reg_param
        self.lr = learning_rate
        self.batch_size = batch_size
        self.save_model_name = save_model_name
        self.comp_net = self.create_siamese(basenet=basenet, no_of_layers=no_of_layers,
                        init_last_layer_weights=[init_last_layer_beta, np.array([0])])

    def predict_weights(self):
        comp_test_model = Model(inputs=self.comp_net.input[0], outputs=self.comp_net.get_layer('comp').get_output_at(0))
        comp_test_model.load_weights(self.save_model_name, by_name=True)
        comp_test_model.compile(loss="mean_squared_error", optimizer=Adam(self.lr))
        x_beta_b = np.squeeze(comp_test_model.predict(self.imgs_lst))
        return x_beta_b

    def create_siamese(self, basenet="googlenet", no_of_layers=2, init_last_layer_weights=None):
        input1 = Input(shape=self.input_shape)
        input2 = Input(shape=self.input_shape)
        # get features from base network
        if basenet == "googlenet":
            comp_out1, comp_out2 = create_googlenet(input1, input2, reg_param=self.reg_param)
            distance = Lambda(BTPred, output_shape=(1,))([comp_out1, comp_out2])
            comp_net = Model([input1, input2], distance)
            comp_net.load_weights(GOOGLENET_INIT_WEIGHTS_PATH, by_name=True)
            if init_last_layer_weights[0] is not None:
                comp_net.get_layer('comp').set_weights(init_last_layer_weights)
        else:
            comp_out1, comp_out2 = create_fc_basenet(input1, input2, reg_param=self.reg_param, no_of_layers=no_of_layers)
            distance = Lambda(BTPred, output_shape=(1,))([comp_out1, comp_out2])
            comp_net = Model([input1, input2], distance)
        comp_net.compile(loss=BTLoss, optimizer=Adam(self.lr))
        comp_net.summary()
        return comp_net

    def train_one_epoch(self):
        start = time()
        self.comp_net.fit([self.comp_imgs_lst_pair_left, self.comp_imgs_lst_pair_right],
                          self.comp_labels, batch_size=self.batch_size, epochs=1)
        end = time()
        self.comp_net.save(self.save_model_name)
        x_beta_b = self.predict_weights()
        return x_beta_b, (end - start)


class multiway_siamese(object):
    def __init__(self, true_order_labels, rank_imgs_lst, imgs_lst,
                 reg_param=0.0002, learning_rate=1e-4,
                 save_model_name='siamese.h5', no_of_layers=2):
        '''
        :param true_order_labels: (number of rankings, ranking length)
        :param rank_imgs_lst: ranking length x (number of rankings, number of features)
        :param imgs_lst: (number of samples, number of features)
        '''
        self.rank_imgs_lst = rank_imgs_lst
        self.true_order_labels = true_order_labels
        self.ranking_length_for_siamese = len(rank_imgs_lst)
        self.imgs_lst = imgs_lst
        self.input_shape = imgs_lst.shape[1:]
        self.reg_param = reg_param
        self.lr = learning_rate
        self.batch_size = batch_size
        self.save_model_name = save_model_name
        self.comp_net = self.create_siamese(no_of_layers=no_of_layers)

    def predict_weights(self):
        comp_test_model = Model(inputs=self.comp_net.input[0], outputs=self.comp_net.get_layer('comp').get_output_at(0))
        comp_test_model.load_weights(self.save_model_name, by_name=True)
        comp_test_model.compile(loss="mean_squared_error", optimizer=Adam(self.lr))
        x_beta_b = np.squeeze(comp_test_model.predict(self.imgs_lst))
        return x_beta_b

    def create_siamese(self, no_of_layers=2):
        # inputs: ranking length x (number of rankings, number of features)
        inputs = []
        for _ in range(self.ranking_length_for_siamese):
            inputs.append(Input(shape=self.input_shape))
        inputs = [inputarray for inputarray in inputs]
        # get features from base network
        outputs = create_fc_multiway_basenet(inputs, reg_param=self.reg_param, no_of_layers=no_of_layers)
        # outputs: ranking length x * (number of rankings, 1)
        ll = Lambda(PLobj, output_shape=(1,))(outputs)
        comp_net = Model(inputs, ll)
        comp_net.compile(loss=PLLoss, optimizer=Adam(self.lr))
        comp_net.summary()
        return comp_net

    def train_one_epoch(self):
        start = time()
        self.comp_net.fit(self.rank_imgs_lst, self.true_order_labels, batch_size=self.batch_size, epochs=1)
        end = time()
        self.comp_net.save(self.save_model_name)
        x_beta_b = self.predict_weights()
        return x_beta_b, (end - start)







