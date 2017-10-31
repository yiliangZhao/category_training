import pandas as pd
import os, sys, time
import pickle
from joblib import Parallel, delayed

# file contains the list of main categories
MAIN_CAT_LIST = "/data1/category_production/main_cat_list.pkl"
# This dir should contain downloaded data for several cat ID sub-dirs ("62", "63", etc.)
RAW_DATA_DIR = "/data1/category_production/category_data/"
# This dir stores the preprocessed data
DATA_DIR = "/data1/category_production/train_data/"

# This is the path of input file for fast text
FASTTEXT_TRAIN_FILE = os.path.join(DATA_DIR, "train_FT.txt")

"""
def load_category_df(cat_id_to_name_file, main_cat):
    df_cat = pd.read_csv(cat_id_to_name_file, encoding='utf-8')
    df_main = df_cat[df_cat['main_cat'] == main_cat]
    return df_main[(df_main['level3_name']!='Others') & (
        df_main['level3_name']!='Default')]
    
def prepro(cat_id_to_name_file, output_dir, cat_id):
    df = pd.read_csv(os.path.join(output_dir, str(cat_id) + '.csv'), encoding='utf-8')
    df = df.rename(columns={'catid': 'level3_cat'})

    non_default_label_set = load_category_df(cat_id_to_name_file, cat_id)['level3_cat'].unique()
    df_train = df[df['level3_cat'].isin(non_default_label_set)]
    print "cat %d size: %d (before) => %d (after), #labels=%d" % (cat_id, len(df), len(df_train), len(df_train['level3_cat'].unique()))
    df_train.to_csv(os.path.join(output_dir, str(cat_id) + '.csv'), index=False, encoding='utf-8')
    return df_train
"""


def concat_level3(cat_id):
    cat_data_dir = os.path.join(RAW_DATA_DIR, str(cat_id))
    list_df_l3 = list()
    for csv_file in os.listdir(cat_data_dir):
       list_df_l3.append(pd.read_csv(os.path.join(cat_data_dir, csv_file), encoding='utf-8'))

    df_merge =  pd.concat(list_df_l3)
    df_merge.to_csv(os.path.join(DATA_DIR, str(cat_id) + '.csv'), encoding='utf-8', index=None)


def prepare_data(**kwargs):
    with open(MAIN_CAT_LIST, 'rb') as f:
        all_cat_list = pickle.load(f)

    # Concatenate all items from level 3 categories under the same level 1 category
    Parallel(n_jobs=10, verbose=1)(delayed(concat_level3)(cat_id) for cat_id in all_cat_list)
    list_df = list()

    for cat_id in all_cat_list:
        list_df.append(pd.read_csv(os.path.join(DATA_DIR, str(cat_id) + '.csv')))
    df = pd.concat(list_df, ignore_index=True)

    # Output fastText format
    df['__label__'] = '__label__' + df['main_cat'].astype(str)
    df['fasttext'] = df['__label__'] + " " + df['tokenized_name'] + " " + df['tokenized_desc']
    df['fasttext'].to_csv(FASTTEXT_TRAIN_FILE, encoding='utf-8', header=False, index=False)

if __name__ == '__main__':
    # main_cat_list, data_dir, cat_id_to_name_file, output_dir, fasttext_data = sys.argv[1:6]
    prepare_data()    

