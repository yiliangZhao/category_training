#!/bin/bash
spark-submit --driver-memory 16G --executor-memory 10G --executor-cores 3 --num-executors 300 --driver-cores 3 /data1/category_production/core/tasks/get_training_data_for_l3.py $1