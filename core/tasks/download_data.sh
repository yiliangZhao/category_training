#!/bin/bash
for i in {0..19}
do
    spark-submit get_training_data_for_l3.py $i
done
