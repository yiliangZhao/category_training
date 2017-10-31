# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
from builtins import range
import airflow
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.models import DAG
import time
from datetime import datetime, timedelta
from pprint import pprint
import pickle
sys.path.append('/data1/category_production')
from core.tasks.get_list_main_cats import get_list_main_cats
from core.tasks.get_training_data_for_l3 import get_data
from core.tasks.prepare_train_data import prepare_data
from core.tasks.cnn_training import train_cnn

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2017, 9, 22),
    'email': ['yiliang.zhao@shopee.com'],
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}

dag = DAG(dag_id='category_training_new', default_args=default_args, schedule_interval='0 0 * * 0')

get_main_list = PythonOperator(
    task_id='get_list_main_cats',
    provide_context=True,
    python_callable=get_list_main_cats,
    retries=3,
    dag=dag)

data_preparation = PythonOperator(
    task_id='data_preprocessing',
    provide_context=True,
    python_callable=prepare_data,
    retries=3,
    dag=dag)

training_l1 = BashOperator(
    task_id='training_fasttext',
    bash_command='/data1/category_production/core/tasks/fasttest_train.sh ', 
    retries=3,
    dag=dag)

training_l3 = PythonOperator(
    task_id='training_cnn',
    provide_context=True,
    python_callable=train_cnn,
    retries=3,
    dag=dag)

templated_command = """
echo {{params.index}}
/data1/category_production/core/tasks/data_download.sh {{params.index}}
"""

base = [0, 1, 2, 3]

for index in range(5):
    print('download_for_cat_%d' % index)
    task_download = PythonOperator(
        task_id='download_for_cat_%d' % index,
        python_callable=get_data,
        op_kwargs={'cat_idx': [x + len(base) * index for x in base]},
        dag=dag)
    """
    task_cnn = PythonOperator(
        task_id='train_cnn_individual_%d' % cat_idx,
        python_callable=train_cnn_individual,
        op_kwargs={'cat_idx':cat_idx},
        dag=dag)
    
    task_download = BashOperator(
        task_id='download_for_cat_%d' % cat_idx,
        bash_command=templated_command,
        params={'index':cat_idx},
        dag=dag
    )
    """

    task_download.set_upstream(get_main_list)
    data_preparation.set_upstream(task_download)
    training_l3.set_upstream(task_download)
    # task_cnn.set_upstream(task_download)

training_l1.set_upstream(data_preparation)
