import os


LOCAL_ROOT = "/Users/varunnathan/Documents/General"
PROJ_DIR = os.path.join(LOCAL_ROOT, "ExternalTest/Prodigal")
RAW_DATA_DIR = os.path.join(PROJ_DIR, "raw")
INTER_DATA_DIR = os.path.join(PROJ_DIR, "intermediate")
RESULTS_DIR = os.path.join(PROJ_DIR, "results")
MODEL_DIR = os.path.join(PROJ_DIR, "model")
TRAIN_FN = os.path.join(RAW_DATA_DIR, "final_train.json")
SPACY_MODEL_DIR = os.path.join(MODEL_DIR, "spacy_en_lg_v1")
FLAIR_FT_MODEL_DIR = os.path.join(MODEL_DIR, "flair_ft_v1")
FLAIR_MODEL_DIR = os.path.join(MODEL_DIR, "flair_v1")
BERT_MODEL_DIR_V1 = os.path.join(MODEL_DIR, "bert_base_cased_v1")
BERT_MODEL_DIR_V2 = os.path.join(MODEL_DIR, "bert_base_cased_v2")
BERT_TAG_IDX_INV_FN = os.path.join(BERT_MODEL_DIR_V2, 'tag_index_inv.json')
