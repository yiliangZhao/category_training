# -*- coding: utf-8 -*-
"""
==========================================================
Classification of product categories using sparse features
==========================================================

This is an example showing how scikit-learn can be used to classify documents
by topics using a bag-of-words approach. This example uses a scipy.sparse
matrix to store the features and demonstrates various classifiers that can
efficiently handle sparse matrices.

"""

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck
# License: BSD 3 clause
#
# Modified for Product Categorization

# from __future__ import print_function
import logging
from collections import Counter

import numpy as np

import os, sys
import time
import pickle
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, Activation
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.metrics import top_k_categorical_accuracy
# from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
import csv
import datetime
import re
import collections
import random
from keras import backend as K
import tensorflow as tf
import keras
import pandas as pd
import cPickle
from data_helpers import load_data, load_gensim_w2v, add_unknown_words, get_W

# file contains the list of main categories
MAIN_CAT_LIST = "/data1/category_production/main_cat_list.pkl"
W2V_FILE = '/data1/w2v_files/w2v_tw/300_dim/largewv_300_2'

def inTop3(y_tue,y_pred):
    return top_k_categorical_accuracy(y_tue, y_pred, k=3)

def inTop2(y_tue,y_pred):
    return top_k_categorical_accuracy(y_tue, y_pred, k=2)


def remove_symbols(source):
    """
    Keep only Chinese Characters and letters of a string
    """
    if source is np.nan:
        return " "
    if type(source) in [int, datetime.datetime, float]:
        return str(source)
    try:
        temp = source.encode('utf-8').decode('utf-8')
    except:
        temp = source.decode('utf-8')
    reg = u"([\u4e00-\u9fa5a-zA-Z]+)"
    pattern = re.compile(reg)
    results = pattern.findall(temp)
    return " ".join(results)


def encode_train(train_label):
    name_set = set(train_label)
    name_dict = {k: v for v, k in enumerate(name_set)}
    return name_dict


def encode_apply(train_label, name_dict):
    y_train = [name_dict[label] for label in train_label]
    return y_train


def clean_text(doc):
    word_list = doc.split()
    if u"請" in word_list:
        idx = word_list.index(u"請")
        word_list = word_list[:idx]
    return ' '.join(word_list)


def train_cnn_individual(**kwargs):
    cat_idx = int(kwargs['cat_idx'])

    with open(MAIN_CAT_LIST, 'rb') as f:
        main_cat_list = pickle.load(f)

    cat_id = main_cat_list[cat_idx]

    all_title = []
    all_label = []
    data_path = "/data1/category_production/category_data/%d" % cat_id
    for filename in os.listdir(data_path):
        if filename.endswith(".csv"):
            l3_data = os.path.join(data_path, filename)
            l3_idx = int(filename.split('.csv')[0])
            df_train = pd.read_csv(l3_data, encoding='utf-8')
            X_train = df_train['tokenized_name'] + " " + df_train['tokenized_desc']
            train_title = [clean_text(doc) for doc in X_train.values]
            train_label = [l3_idx] * len(train_title)
            all_title = all_title + train_title
            all_label = all_label + train_label
    print('data loaded')
    dict_label = encode_train(all_label)
    train_y = encode_apply(all_label, dict_label)

    embedding_dim = 300
    max_dict_title = 70000  ### test whether it should be set to be max vocab
    max_len_title = 50
    train_vec_title, vocab_title, vocabulary_inv_title = load_data(all_title, max_dict_title, max_len_title)
    print 'keras is ready for running for %d' % cat_id
    print 'length of data is', train_vec_title.shape[1]
    # w2v_file = '/home/ld-sgdev/rui_zhao/w2v_average/wv_50_2'

    print "loading word2vec vectors...",
    w2v = load_gensim_w2v(W2V_FILE, vocab_title)
    print "word2vec loaded!"

    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab_title, embedding_dim)
    W = get_W(w2v, vocab_title, embedding_dim)
    print type(W)

    del w2v

    model_trainy = np_utils.to_categorical(train_y)

    # Model Hyperparameters

    filter_sizes = (1, 2)
    num_filters = 500
    dropout_prob = (0.5, 0.5, 0.5)
    hidden_dims = [1500, 3000]
    num_class = model_trainy.shape[1]
    # Training parameters

    # Prepossessing parameters
    sequence_length = train_vec_title.shape[1]
    input_shape = (sequence_length,)

    model_input = Input(shape=input_shape)
    z = Embedding(len(vocabulary_inv_title), embedding_dim, input_length=sequence_length, weights=[W],
                  name="embedding")(model_input)
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(z)
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(conv)
        conv = GlobalMaxPooling1D()(conv)
        # conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    # z = Dropout(dropout_prob[1])(z)
    z = Dense(hidden_dims[0])(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    z = Dropout(dropout_prob[2])(z)
    z = Dense(hidden_dims[1])(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    z = Dense(num_class)(z)
    z = BatchNormalization()(z)

    model_output = Activation('softmax')(z)
    model = Model(model_input, model_output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy", inTop2, inTop3])

    # batch_size = int(round(model_trainy.shape[0]/float(2560)))
    batch_size = 256
    num_epochs = 10
    print 'batch size', batch_size
    model.fit(train_vec_title, model_trainy, batch_size=batch_size, epochs=num_epochs, shuffle=True,
              verbose=2)  # , callbacks=[epoch_acc])

    print '\n'
    print '--------tmp results------------------'

    directory = 'model_saved/%d_files' % cat_id
    if not os.path.exists(directory):
        os.makedirs(directory)

    catid_l3 = {y: x for x, y in dict_label.iteritems()}
    model.save(directory + '/%d_model.h5' % cat_id)  # creates a HDF5 file 'my_model.h5'
    del model  # deletes the existing model
    cPickle.dump([embedding_dim, max_len_title, vocab_title, catid_l3],
                 open(directory + "/config_%d.p" % cat_id, "wb"))


def train_cnn(**kwargs):
    # Load the list of main catids
    with open(MAIN_CAT_LIST, 'rb') as f:
        cat_ids = pickle.load(f)

    for cat_id in cat_ids:
        all_title = []
        all_label = []
        data_path = "/data1/category_production/category_data/%d" % cat_id
        for filename in os.listdir(data_path):
            if filename.endswith(".csv"):
                l3_data = os.path.join(data_path, filename)
                l3_idx = int(filename.split('.csv')[0])
                df_train = pd.read_csv(l3_data, encoding='utf-8')
                X_train = df_train['tokenized_name'] + " " + df_train['tokenized_desc']
                train_title = [clean_text(doc) for doc in X_train.values]
                train_label = [l3_idx] * len(train_title)
                all_title = all_title + train_title
                all_label = all_label + train_label
        print('data loaded')
        dict_label = encode_train(all_label)
        train_y = encode_apply(all_label, dict_label)

        embedding_dim = 300
        max_dict_title = 70000  ### test whether it should be set to be max vocab
        max_len_title = 50
        train_vec_title, vocab_title, vocabulary_inv_title = load_data(all_title, max_dict_title, max_len_title)
        print 'keras is ready for running for %d' %cat_id
        print 'length of data is', train_vec_title.shape[1]
        # w2v_file = '/home/ld-sgdev/rui_zhao/w2v_average/wv_50_2'

        print "loading word2vec vectors...",
        w2v = load_gensim_w2v(W2V_FILE, vocab_title)
        print "word2vec loaded!"

        print "num words already in word2vec: " + str(len(w2v))
        add_unknown_words(w2v, vocab_title, embedding_dim)
        W = get_W(w2v, vocab_title, embedding_dim)
        print type(W)

        del w2v

        model_trainy = np_utils.to_categorical(train_y)

        # Model Hyperparameters

        filter_sizes = (1, 2)
        num_filters = 500
        dropout_prob = (0.5, 0.5, 0.5)
        hidden_dims = [1500, 3000]
        num_class = model_trainy.shape[1]
        # Training parameters

        # Prepossessing parameters
        sequence_length = train_vec_title.shape[1]
        input_shape = (sequence_length,)

        model_input = Input(shape=input_shape)
        z = Embedding(len(vocabulary_inv_title), embedding_dim, input_length=sequence_length, weights=[W],
                      name="embedding")(model_input)
        conv_blocks = []
        for sz in filter_sizes:
            conv = Convolution1D(filters=num_filters,
                                 kernel_size=sz,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(z)
            conv = Convolution1D(filters=num_filters,
                                 kernel_size=sz,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(conv)
            conv = GlobalMaxPooling1D()(conv)
            # conv = Flatten()(conv)
            conv_blocks.append(conv)
        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

        # z = Dropout(dropout_prob[1])(z)
        z = Dense(hidden_dims[0])(z)
        z = BatchNormalization()(z)
        z = Activation('relu')(z)
        z = Dropout(dropout_prob[2])(z)
        z = Dense(hidden_dims[1])(z)
        z = BatchNormalization()(z)
        z = Activation('relu')(z)
        z = Dense(num_class)(z)
        z = BatchNormalization()(z)

        model_output = Activation('softmax')(z)
        model = Model(model_input, model_output)
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy", inTop2, inTop3])

        # batch_size = int(round(model_trainy.shape[0]/float(2560)))
        batch_size = 256
        num_epochs = 10
        print 'batch size', batch_size
        model.fit(train_vec_title, model_trainy, batch_size=batch_size, epochs=num_epochs, shuffle=True,
                  verbose=2)  # , callbacks=[epoch_acc])

        print '\n'
        print '--------tmp results------------------'

        directory = 'model_saved/%d_files' % cat_id
        if not os.path.exists(directory):
            os.makedirs(directory)

        catid_l3 = {y: x for x, y in dict_label.iteritems()}
        model.save(directory + '/%d_model.h5' % cat_id)  # creates a HDF5 file 'my_model.h5'
        del model  # deletes the existing model
        cPickle.dump([embedding_dim, max_len_title, vocab_title, catid_l3],
                     open(directory + "/config_%d.p" % cat_id, "wb"))

if __name__ == '__main__':
    train_cnn()
