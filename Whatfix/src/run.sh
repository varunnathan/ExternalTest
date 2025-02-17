#!/bin/bash

# Check if MODE is passed as an argument
while getopts m: flag
do
    case "${flag}" in
        m) MODE=${OPTARG};;
    esac
done

# Default MODE if not provided
MODE=${MODE:-train}

export PYTHON_PATH=/home/ec2-user/SageMaker/vnathan/ExternalTest/Whatfix
export MODEL_NAME=ZAM
export DATA_DIR=/home/ec2-user/SageMaker/vnathan/external/data/preprocessed/min_count5
export INP_TRAIN_DIR=/home/ec2-user/SageMaker/vnathan/external/data/preprocessed/min_count5/seq_query_split
export DECAY_METHOD=adam
export LR=0.0005
export BATCH_SIZE=384
export TEST_BATCH_SIZE=256
export VALID_BATCH_SIZE=256
export MAX_NUM_TRAIN_EPOCHS=6
export UPREV_REVIEW_LIMIT=20
export EMBEDDING_DIM=128
export NUM_LAYERS=1
export FF_SIZE=256
export NUM_HEADS=4
export DEVICE=cuda

# Set SAVE_DIR based on MODEL_NAME
case $MODEL_NAME in
    item_transformer)
        SAVE_DIR=/home/ec2-user/SageMaker/vnathan/external/model/tem
        ;;
    ZAM)
        SAVE_DIR=/home/ec2-user/SageMaker/vnathan/external/model/zam
        ;;
    AEM)
        SAVE_DIR=/home/ec2-user/SageMaker/vnathan/external/model/aem
        ;;
    QEM)
        SAVE_DIR=/home/ec2-user/SageMaker/vnathan/external/model/qem
        ;;
    *)
        echo "Invalid MODEL_NAME: $MODEL_NAME"
        exit 1
        ;;
esac

PYTHONPATH=$PYTHON_PATH python ./modelling/main.py --model_name $MODEL_NAME --mode $MODE --data_dir $DATA_DIR \
 --input_train_dir $INP_TRAIN_DIR --save_dir $SAVE_DIR --decay_method $DECAY_METHOD --device $DEVICE \
 --max_train_epoch $MAX_NUM_TRAIN_EPOCHS --lr $LR --batch_size $BATCH_SIZE --uprev_review_limit $UPREV_REVIEW_LIMIT \
 --embedding_size $EMBEDDING_DIM --inter_layers $NUM_LAYERS --ff_size $FF_SIZE --heads $NUM_HEADS \
 --candi_batch_size $TEST_BATCH_SIZE --valid_batch_size $VALID_BATCH_SIZE
