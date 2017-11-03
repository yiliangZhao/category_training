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

import numpy as np
import os
import pickle
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, Activation
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.merge import Concatenate
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.metrics import top_k_categorical_accuracy
import datetime
import re
import pandas as pd
import cPickle
from data_helpers import load_data, load_gensim_w2v, add_unknown_words, get_W

# file contains the list of main categories
MAIN_CAT_LIST = "/data1/category_production/main_cat_list.pkl"
W2V_FILE = '/data1/w2v_files/w2v_tw/300_dim/largewv_300_2'
CATID_NAME_MAPPING = "/data1/category_production/catid_catname.csv"
LEVEL3_CAT_LIST = "/data1/category_production/catlist.pickle"

def in_top3(y_tue, y_pred):
    return top_k_categorical_accuracy(y_tue, y_pred, k=3)


def in_top2(y_tue, y_pred):
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
    """
    Assign unique index to labels
    :param train_label: mapping from label to index
    :type: list[int]
    :return: dict that maps label to index
    """
    name_set = set(train_label)
    name_dict = {k: v for v, k in enumerate(name_set)}
    return name_dict


def encode_apply(train_label, name_dict):
    """
    Transform a list of labels to a list of indices
    :param train_label: list of labels, which are category ids (int) of level 3 categories
    :type list[int]
    :param name_dict: mapping from label to index
    :return: list of indices
    """
    y_train = [name_dict[label] for label in train_label]
    return y_train


def clean_text(doc):
    """
    Keep all words that are before "請"
    :param doc: title and description
    :type: str
    :return: str
    """
    index = doc.find(u"請")
    if index == -1:
        return doc
    if index == 0:
        return doc[index + 1:]
    return doc[:index]


def train_cnn(**kwargs):
    # Load the list of main catids
    with open(MAIN_CAT_LIST, 'rb') as f:
        cat_ids = pickle.load(f)

    with open(LEVEL3_CAT_LIST, 'rb') as f:
        level3_cat_ids = pickle.load(f)
    print ('Number of level3 categories to be predicted: %d' % len(level3_cat_ids))
    # df_cat_name = pd.read_csv(CATID_NAME_MAPPING)
    # set_others = set(df_cat_name['level3_cat'])
    for cat_id in cat_ids:
        print ('processing: %d' %cat_id)
        all_title = [] # list of title + description, list(str)
        all_label = [] # list of labels, list(int)
        data_path = "/data1/category_production/category_data/%d" % cat_id
        for filename in os.listdir(data_path):
            if filename.endswith(".csv"):
                l3_idx = int(filename.split('.csv')[0])
                if l3_idx in level3_cat_ids:
                    print ('load data for level3_cat: %d' %l3_idx)
                    df_train = pd.read_csv(os.path.join(data_path, filename), encoding='utf-8')
                    X_train = df_train['tokenized_name'] + " " + df_train['tokenized_desc']
                    train_title = [clean_text(doc) for doc in X_train.values]
                    train_label = [l3_idx] * len(train_title)
                    all_title = all_title + train_title
                    all_label = all_label + train_label
        print('data loaded for main cat: %d' %cat_id)

        # Create the mapping from label to index, where index ranges from 0 to len(all_label) - 1
        dict_label = encode_train(all_label)

        # Transform the list of labels to the list of indices
        train_y = encode_apply(all_label, dict_label)

        embedding_dim = 300
        max_dict_title = 70000  # test whether it should be set to be max vocab
        max_len_title = 50  # maximum number of tokens to be considered

        # train_vec_title is a 2-D numpy array of dimension: number_title * max_len_title
        # vocab_title is a dict that maps word token to index
        # vocabulary_inv_title is a list of word tokens
        train_vec_title, vocab_title, vocabulary_inv_title = load_data(all_title, max_dict_title, max_len_title)

        print ('keras is ready for running for %d' % cat_id)
        print ('Number of data points is', train_vec_title.shape[0])

        print ("loading word2vec vectors...",)
        w2v = load_gensim_w2v(W2V_FILE, vocab_title)

        print ("num words already in word2vec: " + str(len(w2v)))
        add_unknown_words(w2v, vocab_title, embedding_dim)
        W = get_W(w2v, vocab_title, embedding_dim)
        print (type(W))

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
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy", in_top2, in_top3])

        # batch_size = int(round(model_trainy.shape[0]/float(2560)))
        batch_size = 256
        num_epochs = 10

        model.fit(train_vec_title, model_trainy, batch_size=batch_size, epochs=num_epochs, shuffle=True,
                  verbose=2)  # , callbacks=[epoch_acc])

        directory = '/data1/category_production/models/model_saved/%d_files' % cat_id
        if not os.path.exists(directory):
            os.makedirs(directory)

        catid_l3 = {y: x for x, y in dict_label.iteritems()}
        model.save(os.path.join(directory, '%d_model.h5' % cat_id))  # creates a HDF5 file 'my_model.h5'
        cPickle.dump([embedding_dim, max_len_title, vocab_title, catid_l3],
                     open(os.path.join(directory, "config_%d.p" % cat_id), "wb"))

if __name__ == '__main__':
    train_cnn()
