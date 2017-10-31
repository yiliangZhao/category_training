#coding: utf-8
import pickle
from utils import get_spark_session

# file contains the list of main categories
MAIN_CAT_LIST = "/data1/category_production/main_cat_list.pkl"
# CAT_FILE = "/data1/category_production/models/cat_mapping.csv"
# CATID_NAME_MAPPING = "/data1/category_production/catid_catname.csv"
LEVEL3_CAT_LIST = "/data1/category_production/catlist.pickle"

def keep_others_defaults(name):
    if 'other ' in name.lower():
        return True
    if ' other' in name.lower():
        return True
    if 'default' in name.lower():
        return True
    return False


def get_list_main_cats(**kwargs):
    with get_spark_session(app_name='category recommendation (get list of main cats)',
                           num_executors=200) as spark_session:
        spark_session.sql("use shopee")

        # get list of all main category ids (excluding help_buy and everything else
        # 76 is catid for helpbuy, 68 is catid for everything else
        query = "select catid from shopee_db__category_tab " \
                "where status = 1 and country = 'TW' and parent_category = 0 and catid != 76 and catid != 68"
        df_main_cats = spark_session.sql(query)
        pandas_main_cats = df_main_cats.toPandas()
        list_main_cats = list(pandas_main_cats['catid'])
        with open(MAIN_CAT_LIST, 'wb') as handle:
            pickle.dump(list_main_cats, handle, pickle.HIGHEST_PROTOCOL)

        """
        query = "select main_cat, sub_cat, level3_cat from dim_category " \
                "where level3_cat is not null and country = 'TW' and status = 1"
        spark_session.sql(query).toPandas().to_csv(CAT_FILE, index=None, encoding='utf-8')
        """

	"""
        query = "SELECT level3_cat, level3_name FROM dim_category " \
                "where level3_name is not null and country = 'TW' and main_cat != 76 and main_cat != 68"

        df_id_name = spark_session.sql(query).toPandas()
        df_id_name['is_other'] = df_id_name['level3_name'].apply(keep_others_defaults)
        df_id_name[df_id_name['is_other']].to_csv(CATID_NAME_MAPPING, index=None, encoding='utf-8')
	"""
	
	query = "select catid from shopee_db__category_tab \
         where is_default_subcat=0 and unbase64(display_name) not like '%其%' \
         and unbase64(display_name) not like '%Other%' and unbase64(display_name) not like '%Default%' \
         and unbase64(display_name) not like '%default%' and unbase64(display_name) not like '%other%' \
         and status=1 and country='TW' and parent_category in (select catid from shopee_db__category_tab \
         where status=1 and parent_category in (select catid from shopee_db__category_tab \
         where status = 1 and parent_category = 0))"

	additional = {10087, 10085, 10083, 10081, 10079, 10180}        
        df_cat_candidates = spark_session.sql(query).toPandas()
        catids = set(df_cat_candidates['catid']).union(additional)
        
        with open(LEVEL3_CAT_LIST, 'wb') as handle:
    	    pickle.dump(catids, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    get_list_main_cats()
