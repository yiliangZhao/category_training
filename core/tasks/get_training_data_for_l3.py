# coding: utf-8
import os
import sys
import pickle
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, broadcast
from utils import get_spark_session

# file contains the list of main categories
MAIN_CAT_LIST = "/data1/category_production/main_cat_list.pkl"

# This dir should contain downloaded data for several cat ID sub-dirs ("62", "63", etc.)
RAW_DATA_DIR = "/data1/category_production/category_data/"


def get_data(**kwargs):
    cat_idx = kwargs['cat_idx']

    with get_spark_session(app_name='category recommendation_%d' %cat_idx[0], num_executors=200) as spark_session:
        spark_session.sql("use shopee")

        with open(MAIN_CAT_LIST, 'rb') as f:
            main_cat_list = pickle.load(f)

        for idx in cat_idx:
            cat_id = main_cat_list[idx]

            # get name for level 3 catid mapping to remove "Others" and "Default"
            query = "SELECT main_cat, level3_cat, level3_name FROM dim_category " \
                    "where country = 'TW' and status = 1 and level3_cat is not null and main_cat = %d" % cat_id
            df_level3_cat_name = spark_session.sql(query)

            # df_level3_cat_name.toPandas().to_csv(CAT_FILE, index=None, encoding='utf-8')

            # Load tokenized data
            item_name_path = "/products/shopee/profile/item_profile/item_tokenization/item_name/TW/segmentation.csv"
            df_item_name = spark_session.read.option("header","false").csv(item_name_path).toDF('itemid','tokenized_name')
            df_item_name.registerTempTable('table_tokenized_name')
            df_item_name = spark_session.sql(
                "select itemid, decode(unbase64(tokenized_name), 'utf-8') as tokenized_name from table_tokenized_name")

            item_description_path = "/products/shopee/profile/item_profile/item_tokenization/item_desc/TW/segmentation.csv"
            df_item_desc = spark_session.read.option("header","false").csv(
                item_description_path).toDF('itemid','tokenized_desc')
            df_item_desc.registerTempTable('table_tokenized_desc')
            df_item_desc = spark_session.sql(
                    "select itemid, decode(unbase64(tokenized_desc), 'utf-8') as tokenized_desc from table_tokenized_desc")

            df_category = spark_session.sql("SELECT catid, status FROM shopee_db__category_tab "
                                            "where status = 1 and country = 'TW'")

            # get itemid, main_cat, level3_cat
            print 'process ', cat_id
            query = "SELECT itemid, cat_group.main_cat, cat_group.sub_cat, cat_group.level3_cat as catid " \
                    "FROM shopee.item_profile where country = 'TW' and cat_group.main_cat = %d" % cat_id
            df_cat = spark_session.sql(query).dropna().distinct()
            df_cat = df_cat.join(broadcast(df_category), 'catid')

            # Generate dictionary to map cat3 to frac
            df_cat3_count = df_cat.groupBy('catid').count()
            df_cat3_count = df_cat3_count.withColumn('sampled', lit(120000.0))
            df_cat3_count = df_cat3_count.withColumn('frac', df_cat3_count['sampled'] / df_cat3_count['count'])
            pandas_cat3_count = df_cat3_count.toPandas()
            pandas_cat3_count['true_frac'] = pandas_cat3_count['frac'].apply(lambda x: x if x < 1 else 1)
            cat3_frac = pandas_cat3_count[['catid', 'true_frac']].set_index('catid').to_dict(orient='dict')['true_frac']

            # sample from original dataframe based on frac
            df_cat_sampled = df_cat.sampleBy("catid", fractions=cat3_frac, seed=0)
            df_final = df_cat_sampled.join(df_item_name, 'itemid')
            df_final = df_final.join(df_item_desc, 'itemid')
            df_final.cache()

            # df_cat_sampled.printSchema()

            pandas_cats = df_cat_sampled[['catid']].distinct().toPandas()
            list_cat3 = list(pandas_cats['catid'])

            # Skip level 3 == "Others" or "Default"
            df_level3_cats = df_level3_cat_name.where(df_level3_cat_name['main_cat'] == cat_id)
            df_level3_cats = df_level3_cats.where(
                (df_level3_cats['level3_name'] == 'Others') | (df_level3_cats['level3_name'] == 'Default'))
            cats_to_filter = set(df_level3_cats.toPandas()['level3_name'])
            list_cat3 = list(set(list_cat3).difference(cats_to_filter))

            category_data_dir = os.path.join(RAW_DATA_DIR, str(cat_id))
            if not os.path.exists(category_data_dir):
                os.makedirs(category_data_dir)

            for cat3 in list_cat3:
                # if os.path.isfile('/data1/category_data/%d/%d.csv'):
                #    continue
                pandas_final = df_final.where(df_final['catid'] == cat3).toPandas()
                pandas_final.to_csv(os.path.join(category_data_dir, str(cat3) + '.csv'), index=None, encoding='utf-8')

if __name__ == '__main__':
    get_data(cat_idx=range(20))
