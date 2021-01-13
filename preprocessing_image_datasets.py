import pandas as pd
import numpy as np
import pickle
import argparse
from utils import *
from PIL import Image
from os.path import exists
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Input
from keras.models import Model
from googlenet_functional import *
from scipy.sparse import save_npz
from scipy.misc import imresize
from itertools import combinations
import codecs

input_shape = (3,224,224)

def create_partitions(rankings_all, comparisons_all, n_fold):
    '''
    :param n_fold: number of cross validation folds
    :param rankings_all: [(i_1,i_2, ...), (i_1,i_2, ...), ...]
    :param comparisons_all: +1/-1
    :return: rankings_train (n_fold x d), rankings_test (n_fold x len(rankings_all/n_fold))
    '''
    d_all = len(rankings_all)
    d_fold = int(d_all / (n_fold + 1))  # last fold is the holdout/test set
    # partition observations into train, validation, and test
    np.random.seed(1)
    # stock indices to a matrix of (n_fold, indices)
    shuffled_ind = np.reshape(np.random.permutation(d_fold * (n_fold + 1)), ((n_fold + 1), d_fold))
    rankings_train = []
    rankings_val = []
    comparisons_train = []
    comparisons_val = []
    # create train and validation sets
    for test_fold in range(n_fold):
        train_ind = shuffled_ind[[fold for fold in range(n_fold) if (fold != test_fold)]]
        train_ind = train_ind.flatten()
        # get training rankings
        rankings_train.append(rankings_all[train_ind])
        # get training comparisons
        comparisons_train.append(comparisons_all[train_ind])
        # get validation rankings
        rankings_val.append(rankings_all[shuffled_ind[test_fold]])
        # get validation comparisons
        comparisons_val.append(comparisons_all[shuffled_ind[test_fold]])
    # get test rankings
    rankings_test = rankings_all[shuffled_ind[n_fold]]
    # dims for train and validation: (n_fold, d_train, number of ranked items at a time (A_l))
    return rankings_train, rankings_val, rankings_test, comparisons_train, comparisons_val

def create_partitions_wrt_sample(rankings_all, comparisons_all, n, n_fold):
    '''
    :param n: number of samples
    :param n_fold: number of cross validation folds
    :param rankings_all: [(i_1,i_2, ...), (i_1,i_2, ...), ...]
    :param comparisons_all: +1/-1
    partition rankings by the samples participating in train or test. no rankings across
    :return: rankings_train (n_fold x d), rankings_test (n_fold x len(rankings_all/n_fold))
    '''
    samp_fold = int(n / (n_fold + 1))
    # partition observations into train, validation, and test
    np.random.seed(1)
    # stock indices to a matrix of (n_fold, indices)
    shuffled_samp = np.reshape(np.random.permutation(samp_fold * (n_fold + 1)), ((n_fold + 1), samp_fold))
    rankings_train = []
    rankings_val = []
    comparisons_train = []
    comparisons_val = []
    train_samp_folds = []
    for test_fold in range(n_fold):
        train_samp = shuffled_samp[[fold for fold in range(n_fold) if (fold != test_fold)]]
        train_samp = train_samp.flatten()
        val_samp = shuffled_samp[test_fold]
        # get training rankings associated with only training samples
        rankings_train_fold = []
        comparisons_train_fold = []
        for i, rank in enumerate(rankings_all):
            if np.all(np.isin(rank, train_samp)):
                rankings_train_fold.append(rank)
                comparisons_train_fold.append(comparisons_all[i])
        rankings_train.append(rankings_train_fold)
        comparisons_train.append(comparisons_train_fold)
        train_samp.sort()
        train_samp_folds.append(train_samp)
        # get validation rankings associated with only validation samples
        rankings_val_fold = []
        comparisons_val_fold = []
        for i, rank in enumerate(rankings_all):
            if np.all(np.isin(rank, val_samp)):
                rankings_val_fold.append(rank)
                comparisons_val_fold.append(comparisons_all[i])
        rankings_val.append(rankings_val_fold)
        comparisons_val.append(comparisons_val_fold)
    # get rankings associated with only test samples
    test_samp = shuffled_samp[n_fold]
    rankings_test = []
    for rank in rankings_all:
        if np.all(np.isin(rank, test_samp)):
            rankings_test.append(rank)
    # dims for train and validation: (n_fold, d_train, number of ranked items at a time (A_l))
    return train_samp_folds, rankings_train, rankings_val, rankings_test, comparisons_train, comparisons_val

def partition_and_save(n_fold, dir, partition, rankings_all, comparisons_all, X, imgs_lst,
                       ranking_length_for_siamese=2):
    n = X.shape[0]
    rankings_all = np.array(rankings_all)
    comparisons_all = np.array(comparisons_all)
    if partition == 'per_samp':
        train_samp_folds, rankings_train, rankings_val, rankings_test, comparisons_train, comparisons_val = \
            create_partitions_wrt_sample(rankings_all, comparisons_all, n, n_fold)
    else:
        rankings_train, rankings_val, rankings_test, comparisons_train, comparisons_val = \
            create_partitions(rankings_all, comparisons_all, n_fold)
        train_samp_folds = [list(range(n)) for _ in range(n_fold)]
    ##########################################################################save
    for test_fold in range(n_fold):
        print(test_fold, 'folds have been saved')
        current_rankings = np.array(rankings_train[test_fold])
        np.save('../data/' + dir + 'data/' + str(test_fold) + '_train_samp', train_samp_folds[test_fold])
        np.save('../data/' + dir + 'data/' + str(test_fold) + '_features', X)
        np.save('../data/' + dir + 'data/' + str(test_fold) + '_imgs_lst', imgs_lst)
        np.save('../data/' + dir + 'data/' + str(test_fold) + '_train', current_rankings)
        np.save('../data/' + dir + 'data/' + str(test_fold) + '_val', np.array(rankings_val[test_fold]))
        ### save comp_imgs_lst_pair and comp_labels for siamese training
        current_comparisons = np.array(comparisons_train[test_fold])
        comp_imgs_lst_pair_left = imgs_lst[current_comparisons[:, 0]]
        comp_imgs_lst_pair_right = imgs_lst[current_comparisons[:, 1]]
        np.save('../data/' + dir + 'data/' + str(test_fold) + '_comp_train_imgs_lst_left', comp_imgs_lst_pair_left)
        np.save('../data/' + dir + 'data/' + str(test_fold) + '_comp_train_imgs_lst_right', comp_imgs_lst_pair_right)
        np.save('../data/' + dir + 'data/' + str(test_fold) + '_comp_train_labels', current_comparisons[:, 2])
        ### save rank_imgs_lst and true_order_labels for siamese training
        rank_imgs_lst = []  # ranking length x (number of rankings, number of features)
        true_order_labels = np.tile(list(range(ranking_length_for_siamese)),
            (len(current_rankings), 1))  # (number of rankings, ranking length). images are appended wrt ranking order
        for ranking_pos in range(ranking_length_for_siamese):
            rank_imgs_lst.append(imgs_lst[current_rankings[:, ranking_pos]])
        np.save('../data/' + dir + 'data/' + str(test_fold) + '_rank_imgs_lst', rank_imgs_lst)
        np.save('../data/' + dir + 'data/' + str(test_fold) + '_true_order_labels', true_order_labels)
        # Compute initial parameters and save
        mat_Pij = est_Pij(n, current_rankings)
        save_npz('../data/' + dir + 'data/' + str(test_fold) + '_mat_Pij', mat_Pij)
        (beta_init, b_init, time_beta_b_init), (exp_beta_init, time_exp_beta_init), (u_init, time_u_init) = \
                init_params(X, current_rankings, mat_Pij)
        all_init_params = [(beta_init, b_init, time_beta_b_init), (exp_beta_init, time_exp_beta_init),
                (u_init, time_u_init)]
        with open('../data/' + dir + 'data/' + str(test_fold) + '_init_params.pickle', "wb") as pickle_out:
            pickle.dump(all_init_params, pickle_out)
            pickle_out.close()
    # save test set
    np.save('../data/' + dir + 'data/rankings_test', np.array(rankings_test))

def save_gifgif_happy_data(n_fold, dir='gifgif_happy_', partition='per_obs', n_img = 50):
    '''
    n: number of items
    p: feature dimension
    X: n*p, feature matrix
    :param n_fold: number of cross validation folds
    :param dir: current directory to read features and labels
    '''
    # First pass over the data to transform GIFGIF HAPPINESS IDs to consecutive integers.
    image_ids = set([])
    with open('../data/' + dir + 'data/' + 'gifgif-dataset-20150121-v1.csv') as f:
        next(f)  # First line is header.
        for line in f:
            emotion, left, right, choice = line.strip().split(",")
            if len(left) > 0 and len(right) > 0 and (emotion == 'happiness' or emotion == 'sadness') and \
                exists('../data/' + dir + 'data/images/' + left + '.gif') and \
                                exists('../data/' + dir + 'data/images/' + right + '.gif'):
                    image_ids.add(left)
                    image_ids.add(right)
            # take n_gif images
            if len(image_ids) >= n_img:
                image_ids = list(image_ids)[:n_img]
                break
    # create googlenet feature extractor model
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)
    feature1, _ = create_googlenet(input1, input2)
    base_net = Model(input1, feature1)
    feature_model = Model(inputs=base_net.input, outputs=base_net.get_layer('feature_extractor').get_output_at(0))
    feature_model.load_weights(GOOGLENET_INIT_WEIGHTS_PATH, by_name=True)
    feature_model.compile(loss='mean_squared_error', optimizer='sgd')
    # load images and googlenet features
    X_imagenet = np.zeros((n_img, 1024), dtype=float)
    # Extract image matrix, (n, 3, 224, 224)
    int_to_idx = dict(enumerate(image_ids))
    idx_to_int = dict((v, k) for k, v in int_to_idx.items())
    imgs_lst = np.zeros((0, 3, 224, 224))
    for image_id, i in idx_to_int.items():
        # load
        image_mtx = img_to_array(load_img('../data/' + dir + 'data/images/' + image_id + '.gif')).astype(np.uint8)
        # resize
        image_mtx = np.reshape(imresize(image_mtx, input_shape[1:]), input_shape)
        # standardize
        image_mtx = (image_mtx - np.mean(image_mtx)) / np.std(image_mtx)
        image_mtx = image_mtx[np.newaxis, :, :, :]
        # concatenate
        imgs_lst = np.concatenate((imgs_lst, image_mtx), axis=0)
        # take googlenet features
        X_imagenet[i, :] = np.squeeze(feature_model.predict(image_mtx))
    # take rankings (ordered lists) and comparisons (+1/-1) of images in image_ids
    rankings_all = []
    comparisons_all = []
    with open('../data/' + dir + 'data/' + 'gifgif-dataset-20150121-v1.csv') as f:
        next(f)  # First line is header.
        for line in f:
            emotion, left, right, choice = line.strip().split(",")
            if left in image_ids and right in image_ids:
                if emotion == 'happiness':  # left is happier
                    # Map ids to integers.
                    left = idx_to_int[left]
                    right = idx_to_int[right]
                    if choice == "left":
                        # Left image won the happiness comparison.
                        rankings_all.append((left, right))
                        # Append to comparisons
                        comparisons_all.append((left, right, +1))
                    elif choice == "right":
                        # Right image won the happiness comparison.
                        rankings_all.append((right, left))
                        # Append to comparisons
                        comparisons_all.append((left, right, -1))
                elif emotion == 'sadness':  # right is happier
                    # Map ids to integers.
                    left = idx_to_int[left]
                    right = idx_to_int[right]
                    if choice == "right":
                        # Left image won the sadness comparison.
                        rankings_all.append((left, right))
                        # Append to comparisons
                        comparisons_all.append((left, right, +1))
                    elif choice == "left":
                        # Right image won the sadness comparison.
                        rankings_all.append((right, left))
                        # Append to comparisons
                        comparisons_all.append((left, right, -1))
    partition_and_save(n_fold, dir, partition, rankings_all, comparisons_all, X_imagenet, imgs_lst)

def save_fac_data(n_fold, dir='fac_', partition='per_obs', n_img = 50):
    '''
    n: number of items
    p: feature dimension
    X: n*p, feature matrix
    :param n_fold: number of cross validation folds
    :param dir: current directory to read features and labels
    '''
    comp_label_file = "/pairwise_comparison.pkl"
    with open('../data/' + dir + 'data/' + comp_label_file, 'rb') as f:
        comp_label_matrix = pickle.load(f)
    image_ids = set([])
    # get all unique images in category
    for row in comp_label_matrix:
        # category, f1, f2, workerID, passDup, imgId, ans
        if row['category'] == 0:
            left = row['f1'] + '/' + row['imgId'] + '.jpg'
            right = row['f2'] + '/' + row['imgId'] + '.jpg'
            if exists('../data/' + dir + 'data/' + left) and exists('../data/' + dir + 'data/' + right):
                image_ids.add(left)
                image_ids.add(right)
        # take n_img images
        if len(image_ids) >= n_img:
            image_ids = list(image_ids)[:n_img]
            break
    # create googlenet feature extractor model
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)
    feature1, _ = create_googlenet(input1, input2)
    base_net = Model(input1, feature1)
    feature_model = Model(inputs=base_net.input, outputs=base_net.get_layer('feature_extractor').get_output_at(0))
    feature_model.load_weights(GOOGLENET_INIT_WEIGHTS_PATH, by_name=True)
    feature_model.compile(loss='mean_squared_error', optimizer='sgd')
    # load images and googlenet features
    X_imagenet = np.zeros((n_img, 1024), dtype=float)
    # Extract image matrix, (n, 3, 224, 224)
    int_to_idx = dict(enumerate(image_ids))
    idx_to_int = dict((v, k) for k, v in int_to_idx.items())
    imgs_lst = np.zeros((0, 3, 224, 224))
    for image_id, i in idx_to_int.items():
        # load
        image_mtx = img_to_array(load_img('../data/' + dir + 'data/' + image_id)).astype(np.uint8)
        # resize
        image_mtx = np.reshape(imresize(image_mtx, input_shape[1:]), input_shape)
        # standardize
        image_mtx = (image_mtx - np.mean(image_mtx)) / np.std(image_mtx)
        image_mtx = image_mtx[np.newaxis, :, :, :]
        # concatenate
        imgs_lst = np.concatenate((imgs_lst, image_mtx), axis=0)
        # take googlenet features
        X_imagenet[i, :] = np.squeeze(feature_model.predict(image_mtx))
    # take rankings (ordered lists) and comparisons (+1/-1) of images in image_ids
    rankings_all = []
    comparisons_all = []
    for row in comp_label_matrix:
        # category, f1, f2, workerID, passDup, imgId, ans
        if row['category'] == 0:
            left = row['f1'] + '/' + row['imgId'] + '.jpg'
            right = row['f2'] + '/' + row['imgId'] + '.jpg'
            choice = row['ans']
            if left in image_ids and right in image_ids:
                # Map ids to integers.
                left = idx_to_int[left]
                right = idx_to_int[right]
                if choice == "left":
                    # Left image won the comparison.
                    rankings_all.append((left, right))
                    # Append to comparisons
                    comparisons_all.append((left, right, +1))
                elif choice == "right":
                    # Right image won the comparison.
                    rankings_all.append((right, left))
                    # Append to comparisons
                    comparisons_all.append((left, right, -1))
    partition_and_save(n_fold, dir, partition, rankings_all, comparisons_all, X_imagenet, imgs_lst)

def save_rop_data(n_fold, dir='rop_', partition='per_obs', manual_feature=False):
    '''
    n: number of items
    p: feature dimension
    X: n*p, feature matrix
    :param n_fold: number of cross validation folds
    :param dir: current directory to read features and labels
    '''
    n_img = 100
    # load all comparisons
    with open('../data/' + dir + 'data/' + 'Partitions.p', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        label_cmp = u.load()['cmpData']  # (expert,pair_index,label)
    df = pd.read_excel('../data/' + dir + 'data/' + '100Features.xlsx')
    image_ids = df.as_matrix()[:n_img, 0]
    image_ids = np.array([name[:-4] for name in image_ids])  # correct extension
    X = df.as_matrix()[:n_img, 1:144].astype('float')
    # standardize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + rtol
    X = (X - X_mean) / X_std
    # create googlenet feature extractor model
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)
    feature1, _ = create_googlenet(input1, input2)
    base_net = Model(input1, feature1)
    feature_model = Model(inputs=base_net.input, outputs=base_net.get_layer('feature_extractor').get_output_at(0))
    feature_model.load_weights(GOOGLENET_INIT_WEIGHTS_PATH, by_name=True)
    feature_model.compile(loss='mean_squared_error', optimizer='sgd')
    # load images and googlenet features
    X_imagenet = np.zeros((n_img, 1024), dtype=float)
    # Extract image matrix, (n, 3, 224, 224)
    int_to_idx = dict(enumerate(image_ids))
    idx_to_int = dict((v, k) for k, v in int_to_idx.items())
    imgs_lst = np.zeros((0, 3, 224, 224))
    for image_id, i in idx_to_int.items():
        # load
        image_mtx = img_to_array(load_img('../data/' + dir + 'data/images/' + image_id + '.png')).astype(np.uint8)
        # resize
        image_mtx = np.reshape(imresize(image_mtx, input_shape[1:]), input_shape)
        # standardize
        image_mtx = (image_mtx - np.mean(image_mtx)) / np.std(image_mtx)
        image_mtx = image_mtx[np.newaxis, :, :, :]
        # concatenate
        imgs_lst = np.concatenate((imgs_lst, image_mtx), axis=0)
        # take googlenet features
        X_imagenet[i, :] = np.squeeze(feature_model.predict(image_mtx))
    # take rankings (ordered lists) and comparisons (+1/-1) of images in image_ids
    M_per_expert = len(label_cmp[0]) # Number of comparisons per expert
    rankings_all = []
    comparisons_all = []
    for expert in range(5):
        for pair_ind in range(M_per_expert):
            item1 = np.where(image_ids == label_cmp[expert][pair_ind][0])[0]
            item2 = np.where(image_ids == label_cmp[expert][pair_ind][1])[0]
            if item1 != np.empty((1,)) and item2 != np.empty((1,)):
                item1 = np.asscalar(item1)
                item2 = np.asscalar(item2)
                if label_cmp[expert][pair_ind][2] == 1:
                    rankings_all.append((item1, item2))
                    # Append to comparisons
                    comparisons_all.append((item1, item2, +1))
                else:
                    rankings_all.append((item2, item1))
                    # Append to comparisons
                    comparisons_all.append((item1, item2, -1))
    if manual_feature:
        partition_and_save(n_fold, dir + "manual_", partition, rankings_all, comparisons_all, X, X)
    else:
        partition_and_save(n_fold, dir, partition, rankings_all, comparisons_all, X_imagenet, imgs_lst)

def save_candy_data(n_fold, dir='candy_', partition='per_obs', flip_noise_prob=0.0,
                    ranking_length_for_siamese=2):
    '''
    n: number of items
    p: feature dimension
    X: n*p, feature matrix
    :param n_fold: number of cross validation folds
    :param dir: current directory to read features and labels
    '''
    X = []  # 85x11
    win_percent = []
    # open file in read mode
    csv_reader = codecs.open('../data/' + dir + 'data/candy-data.csv', 'r', 'utf8')
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        row = row.rstrip().split(",")
        # row variable is a list that represents a row in csv, first column is row names
        X.append(row[1:-1])
        win_percent.append(row[-1])
    # first row is column names
    X = np.array(X[1:]).astype("float")
    win_percent = np.array(win_percent[1:])
    full_ranking_indices = np.flip(np.argsort(win_percent))
    X = X[full_ranking_indices]
    # standardize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + rtol
    X = (X - X_mean) / X_std
    n = X.shape[0]
    # generate comparisons and rankings w.r.t. win percent
    all_multiway_rankings = list(combinations(range(n), ranking_length_for_siamese))
    rankings_all = []
    comparisons_all = []
    for ranking in all_multiway_rankings:
        #temp_ranking = np.array(temp_ranking)
        #ranking = list(temp_ranking[np.flip(np.argsort(win_percent[temp_ranking]))])
        # flip ranking and comparison to add noise
        eps = np.random.uniform()
        if eps > flip_noise_prob:
            # correct one
            item1 = ranking[0]
            item2 = ranking[1]
            rankings_all.append(ranking)
        else:
            item1 = ranking[-1]
            item2 = ranking[0]
            if ranking_length_for_siamese > 2:
                rankings_all.append((item1, item2) + tuple(ranking[1:-1]))
            else:
                rankings_all.append((item1, item2))
        # choose which way to compare
        if np.random.uniform() > 0.5:
            comparisons_all.append((item1, item2, +1))
        else:
            comparisons_all.append((item2, item1, -1))
    # features and images are not separate for numerical datasets
    partition_and_save(n_fold, dir, partition, rankings_all, comparisons_all, X, X, ranking_length_for_siamese)

def save_living_cost_data(n_fold, dir='living_cost_', partition='per_obs', flip_noise_prob=0.0,
                          ranking_length_for_siamese=2, n_countries=50):
    '''
    n: number of items
    p: feature dimension
    X: n*p, feature matrix
    :param n_fold: number of cross validation folds
    :param dir: current directory to read features and labels
    '''
    X = []  # 216x6
    # open file in read mode
    csv_reader = codecs.open('../data/' + dir + 'data/movehubcostofliving.csv', 'r', 'utf8')
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        row = row.rstrip().split(",")
        # row variable is a list that represents a row in csv, first column is row names
        X.append(row[1:])
    # first row is column names
    X = np.array(X[1:]).astype("float")
    X = X[:n_countries]
    # standardize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + rtol
    X = (X - X_mean) / X_std
    # generate comparisons and rankings, ranking order w.r.t. indices
    all_multiway_rankings = list(combinations(range(n_countries), ranking_length_for_siamese))
    rankings_all = []
    comparisons_all = []
    for ranking in all_multiway_rankings:
        # flip ranking and comparison to add noise
        eps = np.random.uniform()
        if eps > flip_noise_prob:
            # correct one
            item1 = ranking[0]
            item2 = ranking[1]
            rankings_all.append(ranking)
        else:
            item1 = ranking[-1]
            item2 = ranking[0]
            if ranking_length_for_siamese > 2:
                rankings_all.append((item1, item2) + tuple(ranking[1:-1]))
            else:
                rankings_all.append((item1, item2))
        # choose which way to compare
        if np.random.uniform() > 0.5:
            comparisons_all.append((item1, item2, +1))
        else:
            comparisons_all.append((item2, item1, -1))
    # features and images are not separate for numerical datasets
    partition_and_save(n_fold, dir, partition, rankings_all, comparisons_all, X, X, ranking_length_for_siamese)

def save_living_quality_data(n_fold, dir='living_quality_', partition='per_obs', flip_noise_prob=0.0,
                            ranking_length_for_siamese=2, n_countries=50):
    '''
    n: number of items
    p: feature dimension
    X: n*p, feature matrix
    :param n_fold: number of cross validation folds
    :param dir: current directory to read features and labels
    '''
    X = []  # 216x6
    # open file in read mode
    csv_reader = codecs.open('../data/' + dir + 'data/movehubqualityoflife.csv', 'r', 'utf8')
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        row = row.rstrip().split(",")
        # row variable is a list that represents a row in csv, first column is row names
        X.append(row[1:])
    # first row is column names
    X = np.array(X[1:]).astype("float")
    X = X[:n_countries]
    # standardize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + rtol
    X = (X - X_mean) / X_std
    # generate comparisons and rankings, ranking order w.r.t. indices
    all_multiway_rankings = list(combinations(range(n_countries), ranking_length_for_siamese))
    rankings_all = []
    comparisons_all = []
    for ranking in all_multiway_rankings:
        # flip ranking and comparison to add noise
        eps = np.random.uniform()
        if eps > flip_noise_prob:
            # correct one
            item1 = ranking[0]
            item2 = ranking[1]
            rankings_all.append(ranking)
        else:
            item1 = ranking[-1]
            item2 = ranking[0]
            if ranking_length_for_siamese > 2:
                rankings_all.append((item1, item2) + tuple(ranking[1:-1]))
            else:
                rankings_all.append((item1, item2))
        # choose which way to compare
        if np.random.uniform() > 0.5:
            comparisons_all.append((item1, item2, +1))
        else:
            comparisons_all.append((item2, item1, -1))
    # features and images are not separate for numerical datasets
    partition_and_save(n_fold, dir, partition, rankings_all, comparisons_all, X, X, ranking_length_for_siamese)

def save_imdb_data(n_fold, dir='imdb_', partition='per_obs', n_movies = 50, flip_noise_prob=0.0):
    X = []  # n_moviesx36
    ratings = []
    row_ind = 0
    # open file in read mode
    csv_reader = codecs.open('../data/' + dir + 'data/imdb.csv', 'r', 'utf8')
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # first row is column names
        if row_ind > 0 and len(X) < n_movies:
            row = row.rstrip().split(",")
            # check for invalid rows
            try:
                cur_rating = float(row[5])
                feature1 = float(row[7])
                feature2 = float(row[8])
                feature3 = [float(elm) for elm in row[10:]]
                res = True
            except:
                res = False
            if res:
                ratings.append(cur_rating)
                features = []
                features.append(feature1)
                features.append(feature2)
                features.extend(feature3)
                X.append(features)
        row_ind += 1
    # first row is column names
    X = np.array(X).astype("float")
    # standardize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + rtol
    X = (X - X_mean) / X_std
    # generate comparisons w.r.t. ratings in decreasing order
    unique_ratings = np.flip(np.unique(ratings))
    movie_indices_grouped = []
    rankings_all = []
    comparisons_all = []
    for rating in unique_ratings:
        movie_indices_grouped.append([i for i in range(len(ratings)) if ratings[i] == rating])
    for i, movies1 in enumerate(movie_indices_grouped[:-1]):
        for movies2 in movie_indices_grouped[i + 1:]:
            for temp_item1 in movies1:
                for temp_item2 in movies2:
                    # flip ranking and comparison to add noise
                    eps = np.random.uniform()
                    if eps > flip_noise_prob:
                        # correct one
                        item1 = temp_item1
                        item2 = temp_item2
                    else:
                        item1 = temp_item2
                        item2 = temp_item1
                    rankings_all.append((item1, item2))
                    if np.random.uniform() > 0.5:
                        comparisons_all.append((item1, item2, +1))
                    else:
                        comparisons_all.append((item2, item1, -1))
    # features and images are not separate for numerical datasets
    partition_and_save(n_fold, dir, partition, rankings_all, comparisons_all, X, X)

def save_imdb_4way_data(n_fold, dir='imdb_multiway_', partition='per_obs', n_movies=50, flip_noise_prob=0.0):
    X = []  # n_moviesx36
    ratings = []
    row_ind = 0
    # open file in read mode
    csv_reader = codecs.open('../data/' + dir + 'data/imdb.csv', 'r', 'utf8')
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # first row is column names
        if row_ind > 0 and len(X) < n_movies:
            row = row.rstrip().split(",")
            # check for invalid rows
            try:
                cur_rating = float(row[5])
                feature1 = float(row[7])
                feature2 = float(row[8])
                feature3 = [float(elm) for elm in row[10:]]
                res = True
            except:
                res = False
            if res:
                ratings.append(cur_rating)
                features = []
                features.append(feature1)
                features.append(feature2)
                features.extend(feature3)
                X.append(features)
        row_ind += 1
    # first row is column names
    X = np.array(X).astype("float")
    # standardize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + rtol
    X = (X - X_mean) / X_std
    # generate comparisons w.r.t. ratings in decreasing order
    unique_ratings = np.flip(np.unique(ratings))
    movie_indices_grouped = []
    rankings_all = []
    comparisons_all = []
    for rating in unique_ratings:
        movie_indices_grouped.append([i for i in range(len(ratings)) if ratings[i] == rating])
    for first in np.arange(0, len(movie_indices_grouped) - 3):
        for second in np.arange(first + 1, len(movie_indices_grouped) - 2):
            for third in np.arange(second + 1, len(movie_indices_grouped) - 1):
                for forth in np.arange(third + 1, len(movie_indices_grouped)):
                    movies1 = movie_indices_grouped[first]
                    movies2 = movie_indices_grouped[second]
                    movies3 = movie_indices_grouped[third]
                    movies4 = movie_indices_grouped[forth]
                    print([first, second, third, forth])
                    for temp_item1 in movies1:
                        for temp_item2 in movies2:
                            for temp_item3 in movies3:
                                for temp_item4 in movies4:
                                    # flip ranking and comparison to add noise
                                    eps = np.random.uniform()
                                    ranking = [temp_item1, temp_item2, temp_item3, temp_item4]
                                    print(ranking)
                                    if eps > flip_noise_prob:
                                        # correct one
                                        item1 = ranking[0]
                                        item2 = ranking[1]
                                        rankings_all.append(ranking)
                                    else:
                                        item1 = ranking[-1]
                                        item2 = ranking[0]
                                        #rankings_all.append((item1, item2) + tuple(np.random.permutation(ranking[1:-1])))
                                        rankings_all.append((item1, item2) + tuple(ranking[1:-1]))
                                    # choose which way to compare
                                    if np.random.uniform() > 0.5:
                                        comparisons_all.append((item1, item2, +1))
                                    else:
                                        comparisons_all.append((item2, item1, -1))
    # features and images are not separate for numerical datasets
    partition_and_save(n_fold, dir, partition, rankings_all, comparisons_all, X, X, 4)

def save_iclr_3way_data(n_fold, dir='iclr_multiway_', partition='per_obs', n_docs=100, flip_noise_prob=0.0):
    """
    Crawled data is here: https://github.com/shaohua0116/ICLR2020-OpenReviewData
    Features are extracted by pre-trained BERT model

    from transformers import BertTokenizer, BertModel
    import torch

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)

    ratings = []
    embeddings = []
    for i, m in enumerate(meta_list):
    if len(m.rating) > 0:
        rating = np.mean(m.rating) / (np.std(m.rating) + 1e-4)
        inputs = tokenizer(m.abstract, return_tensors="pt")
        outputs = model(**inputs)  # 768 dimensions per document
        embedding = outputs.last_hidden_state[-1, -1].data.numpy()  # take the last element of the sequence and batch
        print("Paper count", i)
        print("Average rating", rating)
        ratings.append(rating)
        embeddings.append(embedding)
    """
    X = np.load("../data/iclr_3way_noisy_data/iclr_2020_embeddings.npy").astype("float")[:n_docs]  # n_abstracts x 768
    ratings = np.load("../data/iclr_3way_noisy_data/iclr_2020_ratings.npy")[:n_docs]  # n_abstracts x 1
    # standardize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + rtol
    X = (X - X_mean) / X_std
    # generate comparisons w.r.t. ratings in decreasing order
    unique_ratings = np.flip(np.unique(ratings))
    movie_indices_grouped = []
    rankings_all = []
    comparisons_all = []
    for rating in unique_ratings:
        movie_indices_grouped.append([i for i in range(len(ratings)) if ratings[i] == rating])
    for first in np.arange(0, len(movie_indices_grouped) - 2):
        for second in np.arange(first + 1, len(movie_indices_grouped) - 1):
            for third in np.arange(second + 1, len(movie_indices_grouped)):
                movies1 = movie_indices_grouped[first]
                movies2 = movie_indices_grouped[second]
                movies3 = movie_indices_grouped[third]
                print([first, second, third])
                for temp_item1 in movies1:
                    for temp_item2 in movies2:
                        for temp_item3 in movies3:
                            # flip ranking and comparison to add noise
                            eps = np.random.uniform()
                            ranking = [temp_item1, temp_item2, temp_item3]
                            print(ranking)
                            if eps > flip_noise_prob:
                                # correct one
                                item1 = ranking[0]
                                item2 = ranking[1]
                                rankings_all.append(ranking)
                            else:
                                item1 = ranking[-1]
                                item2 = ranking[0]
                                #rankings_all.append((item1, item2) + tuple(np.random.permutation(ranking[1:-1])))
                                rankings_all.append((item1, item2) + tuple(ranking[1:-1]))
                            # choose which way to compare
                            if np.random.uniform() > 0.5:
                                comparisons_all.append((item1, item2, +1))
                            else:
                                comparisons_all.append((item2, item1, -1))
    # features and images are not separate for numerical datasets
    partition_and_save(n_fold, dir, partition, rankings_all, comparisons_all, X, X, 3)

def save_iclr_4way_data(n_fold, dir='iclr_multiway_', partition='per_obs', n_docs=100, flip_noise_prob=0.0):
    """
    Crawled data is here: https://github.com/shaohua0116/ICLR2020-OpenReviewData
    Features are extracted by pre-trained BERT model
    """
    X = np.load("../data/iclr_3way_noisy_data/iclr_2020_embeddings.npy").astype("float")[:n_docs]  # n_abstracts x 768
    ratings = np.load("../data/iclr_3way_noisy_data/iclr_2020_ratings.npy")[:n_docs]  # n_abstracts x 1
    # standardize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + rtol
    X = (X - X_mean) / X_std
    # generate comparisons w.r.t. ratings in decreasing order
    unique_ratings = np.flip(np.unique(ratings))
    movie_indices_grouped = []
    rankings_all = []
    comparisons_all = []
    for rating in unique_ratings:
        movie_indices_grouped.append([i for i in range(len(ratings)) if ratings[i] == rating])
    for first in np.arange(0, len(movie_indices_grouped) - 3):
        for second in np.arange(first + 1, len(movie_indices_grouped) - 2):
            for third in np.arange(second + 1, len(movie_indices_grouped) - 1):
                for forth in np.arange(third + 1, len(movie_indices_grouped)):
                    movies1 = movie_indices_grouped[first]
                    movies2 = movie_indices_grouped[second]
                    movies3 = movie_indices_grouped[third]
                    movies4 = movie_indices_grouped[forth]
                    print([first, second, third, forth])
                    for temp_item1 in movies1:
                        for temp_item2 in movies2:
                            for temp_item3 in movies3:
                                for temp_item4 in movies4:
                                    # flip ranking and comparison to add noise
                                    eps = np.random.uniform()
                                    ranking = [temp_item1, temp_item2, temp_item3, temp_item4]
                                    print(ranking)
                                    if eps > flip_noise_prob:
                                        # correct one
                                        item1 = ranking[0]
                                        item2 = ranking[1]
                                        rankings_all.append(ranking)
                                    else:
                                        item1 = ranking[-1]
                                        item2 = ranking[0]
                                        # rankings_all.append((item1, item2) + tuple(np.random.permutation(ranking[1:-1])))
                                        rankings_all.append((item1, item2) + tuple(ranking[1:-1]))
                                    # choose which way to compare
                                    if np.random.uniform() > 0.5:
                                        comparisons_all.append((item1, item2, +1))
                                    else:
                                        comparisons_all.append((item2, item1, -1))
    # features and images are not separate for numerical datasets
    partition_and_save(n_fold, dir, partition, rankings_all, comparisons_all, X, X, 4)

def save_iclr_5way_data(n_fold, dir='iclr_multiway_', partition='per_obs', n_docs=100, flip_noise_prob=0.0):
    """
    Crawled data is here: https://github.com/shaohua0116/ICLR2020-OpenReviewData
    Features are extracted by pre-trained BERT model
    """
    X = np.load("../data/iclr_3way_noisy_data/iclr_2020_embeddings.npy").astype("float")[:n_docs]  # n_abstracts x 768
    ratings = np.load("../data/iclr_3way_noisy_data/iclr_2020_ratings.npy")[:n_docs]  # n_abstracts x 1
    # standardize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + rtol
    X = (X - X_mean) / X_std
    # generate comparisons w.r.t. ratings in decreasing order
    unique_ratings = np.flip(np.unique(ratings))
    movie_indices_grouped = []
    rankings_all = []
    comparisons_all = []
    for rating in unique_ratings:
        movie_indices_grouped.append([i for i in range(len(ratings)) if ratings[i] == rating])
    for first in np.arange(0, len(movie_indices_grouped) - 4):
        for second in np.arange(first + 1, len(movie_indices_grouped) - 3):
            for third in np.arange(second + 1, len(movie_indices_grouped) - 2):
                for forth in np.arange(third + 1, len(movie_indices_grouped) - 1):
                    for fifth in np.arange(forth + 1, len(movie_indices_grouped)):
                        movies1 = movie_indices_grouped[first]
                        movies2 = movie_indices_grouped[second]
                        movies3 = movie_indices_grouped[third]
                        movies4 = movie_indices_grouped[forth]
                        movies5 = movie_indices_grouped[fifth]
                        print([first, second, third, forth, fifth])
                        for temp_item1 in movies1:
                            for temp_item2 in movies2:
                                for temp_item3 in movies3:
                                    for temp_item4 in movies4:
                                        for temp_item5 in movies5:
                                            # flip ranking and comparison to add noise
                                            eps = np.random.uniform()
                                            ranking = [temp_item1, temp_item2, temp_item3, temp_item4, temp_item5]
                                            print(ranking)
                                            if eps > flip_noise_prob:
                                                # correct one
                                                item1 = ranking[0]
                                                item2 = ranking[1]
                                                rankings_all.append(ranking)
                                            else:
                                                item1 = ranking[-1]
                                                item2 = ranking[0]
                                                rankings_all.append((item1, item2) + tuple(ranking[1:-1]))
                                            # choose which way to compare
                                            if np.random.uniform() > 0.5:
                                                comparisons_all.append((item1, item2, +1))
                                            else:
                                                comparisons_all.append((item2, item1, -1))
    # features and images are not separate for numerical datasets
    partition_and_save(n_fold, dir, partition, rankings_all, comparisons_all, X, X, 5)

if __name__ == "__main__":
    n_fold = 5
    parser = argparse.ArgumentParser(description='prep', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dir', type=str)
    args = parser.parse_args()
    dir = args.dir
    flip_noise_prob = 0.1
    if dir == 'rop_':
        save_rop_data(n_fold, dir='rop_', manual_feature=False)
    if dir == 'rop_manual_':
        save_rop_data(n_fold, dir='rop_', manual_feature=True)
    elif dir == 'fac_':
        save_fac_data(n_fold, dir=dir)
    elif dir == 'gifgif_happy_':
        save_gifgif_happy_data(n_fold, dir=dir)
    elif dir == 'living_cost_noisy_':
        save_living_cost_data(n_fold, dir=dir, flip_noise_prob=flip_noise_prob)
    elif dir == 'living_cost_3way_noisy_':
        save_living_cost_data(n_fold, dir=dir, flip_noise_prob=flip_noise_prob, ranking_length_for_siamese=3)
    elif dir == 'living_cost_4way_noisy_':
        save_living_cost_data(n_fold, dir=dir, flip_noise_prob=flip_noise_prob, ranking_length_for_siamese=4)
    elif dir == 'living_cost_5way_noisy_':
        save_living_cost_data(n_fold, dir=dir, flip_noise_prob=flip_noise_prob, ranking_length_for_siamese=5)
    elif dir == 'living_cost_6way_noisy_':
        save_living_cost_data(n_fold, dir=dir, flip_noise_prob=flip_noise_prob, ranking_length_for_siamese=6)
    elif dir == 'living_quality_noisy_':
        save_living_quality_data(n_fold, dir=dir, flip_noise_prob=flip_noise_prob)
    elif dir == 'living_quality_3way_noisy_':
        save_living_quality_data(n_fold, dir=dir, flip_noise_prob=flip_noise_prob, ranking_length_for_siamese=3)
    elif dir == 'living_quality_4way_noisy_':
        save_living_quality_data(n_fold, dir=dir, flip_noise_prob=flip_noise_prob, ranking_length_for_siamese=4)
    elif dir == 'living_quality_5way_noisy_':
        save_living_quality_data(n_fold, dir=dir, flip_noise_prob=flip_noise_prob, ranking_length_for_siamese=5)
    elif dir == 'living_quality_6way_noisy_':
        save_living_quality_data(n_fold, dir=dir, flip_noise_prob=flip_noise_prob, ranking_length_for_siamese=6)
    elif dir == 'imdb_noisy_':
        save_imdb_data(n_fold, dir=dir, flip_noise_prob=flip_noise_prob)
    elif dir == 'imdb_4way_noisy_':
        save_imdb_4way_data(n_fold, dir=dir, flip_noise_prob=flip_noise_prob)
    elif dir == 'iclr_3way_noisy_':
        save_iclr_3way_data(n_fold, dir=dir, flip_noise_prob=flip_noise_prob)
    elif dir == 'iclr_4way_noisy_':
        save_iclr_4way_data(n_fold, dir=dir, flip_noise_prob=flip_noise_prob)
    elif dir == 'iclr_5way_noisy_':
        save_iclr_5way_data(n_fold, dir=dir, flip_noise_prob=flip_noise_prob)
    elif dir == 'rop_par_':
        save_rop_data(n_fold, dir='rop_par_', partition='per_samp', manual_feature=False)
    elif dir == 'rop_par_manual_':
        save_rop_data(n_fold, dir='rop_par_', partition='per_samp', manual_feature=True)
    elif dir == 'gifgif_happy_par_':
        save_gifgif_happy_data(n_fold, dir=dir, partition='per_samp')
    elif dir == 'fac_par_':
        save_fac_data(n_fold, dir=dir, partition='per_samp')
    elif dir == 'living_cost_noisy_par_':
        save_living_cost_data(n_fold, dir=dir, partition='per_samp', flip_noise_prob=flip_noise_prob)
    elif dir == 'living_cost_3way_noisy_par_':
        save_living_cost_data(n_fold, dir=dir, partition='per_samp', flip_noise_prob=flip_noise_prob,
                              ranking_length_for_siamese=3)
    elif dir == 'living_cost_4way_noisy_par_':
        save_living_cost_data(n_fold, dir=dir, partition='per_samp', flip_noise_prob=flip_noise_prob,
                              ranking_length_for_siamese=4)
    elif dir == 'living_cost_5way_noisy_par_':
        save_living_cost_data(n_fold, dir=dir, partition='per_samp', flip_noise_prob=flip_noise_prob,
                              ranking_length_for_siamese=5)
    elif dir == 'living_cost_6way_noisy_par_':
        save_living_cost_data(n_fold, dir=dir, partition='per_samp', flip_noise_prob=flip_noise_prob,
                              ranking_length_for_siamese=6)
    elif dir == 'living_quality_noisy_par_':
        save_living_quality_data(n_fold, dir=dir, partition='per_samp', flip_noise_prob=flip_noise_prob)
    elif dir == 'living_quality_3way_noisy_par_':
        save_living_quality_data(n_fold, dir=dir, partition='per_samp', flip_noise_prob=flip_noise_prob,
                              ranking_length_for_siamese=3)
    elif dir == 'living_quality_4way_noisy_par_':
        save_living_quality_data(n_fold, dir=dir, partition='per_samp', flip_noise_prob=flip_noise_prob,
                              ranking_length_for_siamese=4)
    elif dir == 'living_quality_5way_noisy_par_':
        save_living_quality_data(n_fold, dir=dir, partition='per_samp', flip_noise_prob=flip_noise_prob,
                              ranking_length_for_siamese=5)
    elif dir == 'living_quality_6way_noisy_par_':
        save_living_quality_data(n_fold, dir=dir, partition='per_samp', flip_noise_prob=flip_noise_prob,
                              ranking_length_for_siamese=6)
    elif dir == 'imdb_noisy_par_':
        save_imdb_data(n_fold, dir=dir, partition='per_samp', flip_noise_prob=flip_noise_prob)
    elif dir == 'imdb_4way_noisy_par_':
        save_imdb_4way_data(n_fold, dir=dir, partition='per_samp', flip_noise_prob=flip_noise_prob)
    elif dir == 'iclr_3way_noisy_par_':
        save_iclr_3way_data(n_fold, dir=dir, partition='per_samp', flip_noise_prob=flip_noise_prob)
    elif dir == 'iclr_4way_noisy_par_':
        save_iclr_4way_data(n_fold, dir=dir, partition='per_samp', flip_noise_prob=flip_noise_prob)
    elif dir == 'iclr_5way_noisy_par_':
        save_iclr_5way_data(n_fold, dir=dir, partition='per_samp', flip_noise_prob=flip_noise_prob)


