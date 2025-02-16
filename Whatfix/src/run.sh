export PYTHON_PATH=/Users/varunnathan/Documents/Personal/interview_external/whatfix/training
export MODEL_NAME=item_transformer
export MODE=train
export DATA_DIR=/Users/varunnathan/Documents/Personal/interview_external/whatfix/training/data/preprocessed/min_count5
export INP_TRAIN_DIR=/Users/varunnathan/Documents/Personal/interview_external/whatfix/training/data/preprocessed/min_count5/seq_query_split
export DECAY_METHOD=adam
export LR=0.0005
export BATCH_SIZE=384
export MAX_NUM_TRAIN_EPOCHS=20
export UPREV_REVIEW_LIMIT=20
export EMBEDDING_DIM=128
export NUM_LAYERS=1
export FF_SIZE=256
export NUM_HEADS=4
export DEVICE=cpu

# Set SAVE_DIR based on MODEL_NAME
case $MODEL_NAME in
    item_transformer)
        SAVE_DIR=/Users/varunnathan/Documents/Personal/interview_external/whatfix/training/model/tem
        ;;
    ZAM)
        SAVE_DIR=/Users/varunnathan/Documents/Personal/interview_external/whatfix/training/model/zam
        ;;
    AEM)
        SAVE_DIR=/Users/varunnathan/Documents/Personal/interview_external/whatfix/training/model/aem
        ;;
    QEM)
        SAVE_DIR=/Users/varunnathan/Documents/Personal/interview_external/whatfix/training/model/qem
        ;;
    *)
        echo "Invalid MODEL_NAME: $MODEL_NAME"
        exit 1
        ;;
esac

PYTHONPATH=$PYTHON_PATH python ./modelling/main.py --model_name $MODEL_NAME --mode $MODE --data_dir $DATA_DIR \
 --input_train_dir $INP_TRAIN_DIR --save_dir $SAVE_DIR --decay_method $DECAY_METHOD --device $DEVICE \
 --max_train_epoch $MAX_NUM_TRAIN_EPOCHS --lr $LR --batch_size $BATCH_SIZE --uprev_review_limit $UPREV_REVIEW_LIMIT \
 --embedding_size $EMBEDDING_DIM --inter_layers $NUM_LAYERS --ff_size $FF_SIZE --heads $NUM_HEADS