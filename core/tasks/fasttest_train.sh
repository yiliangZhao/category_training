# This dir stores the preprocessed data
DATA_DIR=/data1/category_production/train_data/
FASTTEXT_TRAIN_FILE=$DATA_DIR/train_FT.txt

MODEL_DIR=/data1/category_production/models
FASTTEXT_MODEL=$MODEL_DIR/fasttext_model
W2V_MODEL=/data1/category_production/largewv_300_2.txt

/data1/category_production/fastText.py/fasttext/cpp/fasttext supervised -input $FASTTEXT_TRAIN_FILE -output $FASTTEXT_MODEL -epoch 10 -lr 0.05 -wordNgrams 2 -dim 300 -pretrainedVectors $W2V_MODEL -verbose 1
