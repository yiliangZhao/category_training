#coding: utf-8
import numpy as np
import pandas as pd
import fasttext
from keras.models import Model
from keras.metrics import top_k_categorical_accuracy
from keras import backend as K
import tensorflow as tf
import keras
import cPickle
from keras.models import load_model
import os


CAT_MAPPING_PATH='/data1/category_production/cat_mapping.csv'
FASTTEXT_MODEL_PATH='/data1/category_production/model/fasttext_model.bin'
CNN_MODEL_PATH='/data1/category_production/core/tasks/model_saved/'
THRESHOLD = 0.0

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
df_cat = pd.read_csv(CAT_MAPPING_PATH)


def intop2(y_tue, y_pred):
    return top_k_categorical_accuracy(y_tue, y_pred, k=2)


def intop3(y_tue, y_pred):
    return top_k_categorical_accuracy(y_tue, y_pred, k=3)


def load_cnn_model(cat_id):
    directory = '%s/%d_files/' % (CNN_MODEL_PATH, cat_id)
    with open(directory+"config_%d.p" %cat_id, 'rb') as fp:
        d1 = cPickle.load(fp)
    embedding_dim, max_len, vocab, inv_dict = d1[0], d1[1], d1[2], d1[3]
    model = load_model(directory+'%d_model.h5' %cat_id, custom_objects={'inTop3': intop3, 'inTop2': intop2})
    model._make_predict_function()  # have to initialize before threading
    return (max_len, vocab, inv_dict, model)

# Load models
fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

cat_ids = [72, 73, 74, 75, 1611, 1657, 1837, 1859, 2580, 10076, 62, 67, 69, 70, 71, 65, 64, 63, 100, 66]

cnn_model = {}
for cat_id in cat_ids:
    cnn_model[cat_id] = load_cnn_model(cat_id)


def clean_text(doc):
    """

    :param doc:
    :return:
    """
    index = doc.find(u"請")
    if index <= 0:
        return doc
    return doc[:index]


def build_input_data(sentences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = []
    for sentence in sentences:
        t = []
        for word in sentence:
            if word in vocabulary:
                t.append(vocabulary[word])
        x.append(t)
    x = np.array(x)
    return x


def infer_data(all_corpus, vocab, max_len=300, padding_word="<PAD/>"):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    padded_sentences = []
    sentences = [s.split(" ") for s in all_corpus]
    for i in range(len(sentences)):
        sentence = [word for word in sentences[i] if word in vocab]
        sequence_length = len(sentence)
        num_padding = max_len - sequence_length
        if num_padding > 0:
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:max_len]
        padded_sentences.append(new_sentence)
    #print padded_sentences
    x = build_input_data(padded_sentences, vocab)
    return x


def predict_label(text_strings, vocab, max_len, model):   
    test_vecs = infer_data(text_strings, vocab, max_len=max_len)
    # print 'test_vecs shape: ', test_vecs.shape
    hh = model.predict(test_vecs, batch_size=1000)
    # keras.backend.tensorflow_backend.clear_session()
    return hh


def predict(title, desc):
    title_and_desc = title + " " + desc
    clean_text_strings = [clean_text(title_and_desc)]
    
    # level 1 prediction
    if len(clean_text_strings[0]) < 1:
        print (title_and_desc)
        print ('###' + clean_text_strings[0] + '###')
        return []
    level_1_preds = fasttext_model.predict_proba([title_and_desc], k=3)
    level_1_preds = [(int(label.replace('__label__', '')), prob) for label, prob in level_1_preds[0]]
    # print level_1_preds

    # level 3 prediction
    level_3_preds = []
    for cat_id, prob_l1 in level_1_preds:
        # print cat_id, prob_l1
        max_len, vocab, inv_dict, model = cnn_model[cat_id]
        # print model.summary()
        prob_l3 = predict_label(clean_text_strings, vocab, max_len, model)[0]
        prob_joint = prob_l3 * prob_l1
        hh = [(inv_dict[ith], label_prob, cat_id) for ith, label_prob in enumerate(prob_joint)]
        sorted_hh = sorted(hh, key=lambda x: x[1], reverse=True)[:3]
        level_3_preds += sorted_hh

    level_3_preds = sorted(level_3_preds, key=lambda x: x[1], reverse=True)[:3]
    if level_3_preds[0][1] >= THRESHOLD:
        preds = []
        for i in range(3):
            # preds.append({
            #    'main_cat': level_3_preds[i][2],
            #    'sub_cat': df_cat[df_cat['level3_cat']==level_3_preds[i][0]]['sub_cat'].values[0],
            #    'level3_cat': level_3_preds[i][0],
            # })
            preds.append({'cat': int(level_3_preds[i][0]), 'score': float(level_3_preds[i][1])})
        return preds
    else:
        return []


if __name__ == '__main__':
    result = predict(u'日本 takara  tomy  可 愛達 草莓 水果 屋  臺中 現貨 可自取', u'品牌 國家  日本 takara  tomy  產地  中國  內容物  本體 \
         收銀臺  蛋糕架  水果  三色 各 五  連結 配件  主材質  塑膠  適用 年齡  歲 以上  備註  本 商品 不包含 人偶  商品 安全 標章  m  \
        人物 需 另外 購買  謝謝您 的 配合')
    print result
