# This file contains a function to obtain model results from itemid and shopid
import os
import sys
import time
import numpy as np
import re
import beeshop_db_pb2
import pandas as pd
import MySQLdb
from MySQLdb.cursors import DictCursor
import datetime
from sklearn.externals import joblib
from batch_tokenization import tokenization, tokenization_batch
from prediction_pipeline import predict

NUM_TABLE = 1000  # number of item tables in version 2 in production
THRESHOLD = 0.0
CATID_MODELS = dict()

# function to obtain connection to the database
def connect_db(host, db):
    """
    This function returns a connection to database
    :param host: IP of database
    :param db: name of database
    :return: connection to the database
    """
    conn_dict = {
        'host': host,
        'user': os.environ['SQL_DB_USER'],
        'passwd': os.environ['SQL_DB_PASSWORD'],
        'port': 6606,
        'db': db,
        'connect_timeout': 5,
        'charset': 'utf8'
    }
    conn_ = MySQLdb.connect(**conn_dict)
    conn_.autocommit(True)
    return conn_


# function to get name, description and main_cat from itemid and shopid
def extract_information(itemid, shopid):
    conn_item = connect_db(host=os.environ['HOST_ITEM_DB'], db='shopee_item_v2_db')
    shop_hash = shopid % NUM_TABLE
    cursor = conn_item.cursor(DictCursor)
    query = "SELECT itemid, shopid, name, description, extinfo " \
             "FROM item_v2_tab_%08d " \
             "WHERE itemid = %d;" % (shop_hash, itemid)
    cursor.execute(query)
    print cursor.rowcount
    if cursor.rowcount < 1:
        return None, None, None
    row = cursor.fetchone()
    itemid_fi, shopid_fi, name_fi, description_fi, extinfo_fi = int(row['itemid']), int(row['shopid']), row[
        'name'], row['description'], row['extinfo']
    item_ext = beeshop_db_pb2.ItemExtInfo()
    item_ext.ParseFromString(extinfo_fi)
    cats_1l = item_ext.cats[0].catid[0]
    return cats_1l, name_fi, description_fi


def extract_information_bulk(list_items):
    """
    Return a dataframe with columns: itemid, shopid, name, description, main_cat
    :param items:
    :return:
    """
    list_itemid = [x[0] for x in list_items]

    table_item_dict = dict()
    for itemid in list_itemid:
        key = itemid % 1000
        if key in table_item_dict:
            table_item_dict[key].append(str(itemid))
        else:
            table_item_dict[key] = [str(itemid)]

    table_shop_dict = dict()
    for pair in list_items:
        itemid = pair[0]
        shopid = pair[1]
        key = shopid % 1000
        if key in table_shop_dict:
            table_shop_dict[key].append(str(itemid))
        else:
            table_shop_dict[key] = [str(itemid)]

    conn_item = connect_db(host=os.environ['HOST_ITEM_DB'], db='shopee_item_v2_db')

    list_items = list()
    column_names = ['itemid', 'shopid', 'name', 'description']
    for shop_hash, items in table_shop_dict.iteritems():
        cursor = conn_item.cursor(DictCursor)
        query = "SELECT itemid, shopid, name, description " \
                "FROM item_v2_tab_%08d " \
                "WHERE itemid in (%s);" % (shop_hash, ','.join(list(set(items))))
        rows = cursor.execute(query)

        for row in cursor:
            itemid_fi, shopid_fi, name_fi, description_fi = int(row['itemid']), int(row['shopid']), row['name'], row['description']
            info = (itemid_fi, shopid_fi, name_fi, description_fi)
            list_items.append(info)
    df_information = pd.DataFrame.from_records(list_items, columns=column_names)
    start_time = time.time()
    df_tokenized = tokenization_batch(df_information)
    print('time elapsed (Tokenization): %s' % (time.time() - start_time))
    sys.stdout.flush()
    return df_tokenized


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


def tokenize_str(text):

    """
    text is an unicode string
    """
    title = remove_symbols(text)
    title = title.encode('utf-8')
    title = tokenization(title)
    if title == 'Empty input\n':
        return ''
    return title


def get_tokenized_text(name, description):
    name_decoded = unicode(str(name), 'utf-8')
    description_decoded = unicode(str(description), 'utf-8')
    text_decoded = name_decoded + ' ' + description_decoded
    tokenized_text = tokenize_str(text_decoded)
    return tokenized_text


def get_top_k(main_cat, predictor, tfidf, bow, tokenized_text, k=3):
    prob = predictor.predict_proba(tfidf.transform(bow.transform([tokenized_text])))
    tup_list = zip(predictor.classes_, prob[0])
    top_k = sorted(tup_list, key=lambda x: x[1], reverse=True)[:k]
    min_probability = top_k[k-1][1]
    if min_probability > THRESHOLD:
        return [{'score': x[1], 'cat': int(x[0].split('_')[1].split('.')[0])} for x in top_k]
        # return top_k
    else:
        return []


def get_model(L1):
    """
    get appropriate model based on L1
    :param L1:
    :return:
    """
    if not os.path.exists(os.path.join('../models', str(L1))):
        return None, None, None
    if L1 in CATID_MODELS:
        predictor = CATID_MODELS[L1][0]
        tfidf = CATID_MODELS[L1][1]
        bow = CATID_MODELS[L1][2]
    else:
        predictor = joblib.load(os.path.join('../models', str(L1), 'predictor.pkl'))
        tfidf = joblib.load(os.path.join('../models', str(L1), 'tfidf.pkl'))
        bow = joblib.load(os.path.join('../models', str(L1), 'bow.pkl'))
        CATID_MODELS[L1] = [predictor, tfidf, bow]
    return predictor, tfidf, bow


def get_categories(itemid, shopid):
    """
    For get API
    :param itemid:
    :param shopid:
    :return:
    """
    start_time = time.time()
    _, name, description = extract_information(itemid, shopid)
    print('time elapsed (extract info): %s' % (time.time() - start_time))
    tokenized_text = get_tokenized_text(name, description)
    print('time elapsed (tokenization): %s' % (time.time() - start_time))
    predictor, tfidf, bow = get_model(L1)
    if predictor:
        return get_top_k(L1, predictor, tfidf, bow, tokenized_text, 3), 200
    else:
        return list(), 300


def top_k_category_prediction(catid, name, description, k=3):
    """
    For post API (list of items)
    :param catid: main category id
    :param name:
    :param description:
    :param k:
    :return:
    """
    if catid is None:
        return list()
    predictor, tfidf, bow = get_model(catid)
    if predictor:
        return get_top_k(catid, predictor, tfidf, bow, name + ' ' + description, k)
    else:
        return list()


if __name__ == '__main__':
    list_items = [[556324, 67000], [982869, 67000], [1919083, 67000], [1919522, 67000], [1920312, 67000],
                  [1920598, 67000], [1920740, 67000], [1921333, 67000], [1922152, 67000], [5580304, 1621000], [6110398, 1621000],
                  [6161116, 1621000], [6395354, 1621000], [6396373, 1621000], [6396724, 1621000], [6747041, 1621000], [7495889, 1621000],
                  [7579054, 1621000], [7810626, 67000], [8071410, 1621000], [9995789, 1621000], [10003314, 2543000], [11278655, 1621000]]
    df_features = extract_information_bulk(list_items)

    df_features['category_suggestions'] = df_features.apply(
        lambda row: predict(row['tokened_name'], row['tokened_description']),
        axis=1)

    # prediction = list()
    # for index, row in df_features.iterrows():
    #    prediction.append(predict(row['tokened_name'], row['tokened_description']))
    # df_features['category_suggestions'] = prediction
    df_features.to_csv('debug_category_recommendation.csv', index=None, encoding='utf-8')
