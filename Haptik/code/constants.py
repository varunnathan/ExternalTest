import os

LOCAL_ROOT = "/Users/varunnathan/Documents/General"
PROJ_DIR = os.path.join(LOCAL_ROOT, "ExternalTest/Haptik")
RAW_DATA_DIR = os.path.join(PROJ_DIR, "raw")
INTER_DATA_DIR = os.path.join(PROJ_DIR, "intermediate")
RESULTS_DIR = os.path.join(PROJ_DIR, "results")
TRAIN_FN = os.path.join(RAW_DATA_DIR, "sofmattress_train.csv")
TEST_FN = os.path.join(RAW_DATA_DIR, "sofmattress_test.csv")
BOW_PIPELINE_FN = os.path.join(INTER_DATA_DIR, "bow_pipeline_CA{}.pkl")
LE_FN = os.path.join(INTER_DATA_DIR, "label_encoder.pkl")
OHE_FN = os.path.join(INTER_DATA_DIR, "one_hot_encoder.pkl")
FT_PRETRAINED_FN = os.path.join(LOCAL_ROOT, "crawl-300d-2M-subword/crawl-300d-2M-subword.vec")
FT_TRAIN_FN = os.path.join(INTER_DATA_DIR, "ft_prepared_data_train.txt")
FT_TEST_FN = os.path.join(INTER_DATA_DIR, "ft_prepared_data_test.txt")
FT_SCRATCH_MODEL_FN = os.path.join(INTER_DATA_DIR, "ft_scratch_model.bin")
FT_FINETUNED_MODEL_FN = os.path.join(INTER_DATA_DIR, "ft_finetuned_model.bin")
FT_TRAIN_Wnonalpha_FN = os.path.join(INTER_DATA_DIR, "ft_prepared_data_Wnonalpha_train.txt")
FT_TEST_Wnonalpha_FN = os.path.join(INTER_DATA_DIR, "ft_prepared_data_Wnonalpha_test.txt")
FT_FINETUNED_MODEL_Wnonalpha_FN = os.path.join(INTER_DATA_DIR, "ft_finetuned_model_Wnonalpha.bin")
FT_BERT_OUTPUT_DIR = os.path.join(INTER_DATA_DIR, "ft_bert_output")
FT_BERT_LOGS_DIR = os.path.join(INTER_DATA_DIR, "ft_bert_logs")
FT_PRE_BERT_OUTPUT_DIR = os.path.join(INTER_DATA_DIR, "ft_pre_bert_output")
FT_PRE_BERT_LOGS_DIR = os.path.join(INTER_DATA_DIR, "ft_pre_bert_logs")
FT_SENTBERT_OUTPUT_DIR = os.path.join(INTER_DATA_DIR, "ft_sentbert_output")
FT_PRE_SENTBERT_OUTPUT_DIR = os.path.join(INTER_DATA_DIR, "ft_pre_sentbert_output")