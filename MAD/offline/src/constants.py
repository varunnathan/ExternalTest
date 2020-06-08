import os


LOCAL_DIR = '/Users/varunn/Documents/'
PROJ_DIR = os.path.join(LOCAL_DIR, 'ExternalTest/MAD')
DATA_DIR = os.path.join(LOCAL_DIR, 'ExternalTest_Data/MAD')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')
METADATA_DIR = os.path.join(DATA_DIR, 'metadata')
BASELINE_FEATS_DIR = os.path.join(DATA_DIR, 'baseline_feats')
MODEL_DIR = os.path.join(DATA_DIR, 'model')
PREDICTION_DIR = os.path.join(DATA_DIR, 'prediction')

RAW_INP_FN = os.path.join(RAW_DATA_DIR, '000{}_part_0{}.gz')

USER2IDX_FN = os.path.join(METADATA_DIR, 'user2idx.json')
IDX2USER_FN = os.path.join(METADATA_DIR, 'idx2user.json')
USER2IDX_SEGGE20_FN = os.path.join(METADATA_DIR, 'user2idx_segGE20.json')
IDX2USER_SEGGE20_FN = os.path.join(METADATA_DIR, 'idx2user_segGE20.json')
USER2IDX_SEGLE3_FN = os.path.join(METADATA_DIR, 'user2idx_segLE3.json')
IDX2USER_SEGLE3_FN = os.path.join(METADATA_DIR, 'idx2user_segLE3.json')
USER2IDX_SEG4TO19_FN = os.path.join(METADATA_DIR, 'user2idx_seg4-19.json')
IDX2USER_SEG4TO19_FN = os.path.join(METADATA_DIR, 'idx2user_seg4-19.json')
FINAL_USER2IDX_SEGLT20_FN = os.path.join(METADATA_DIR, 'final_user2idx_segLT20.json')
FINAL_USER2IDX_SEGGE20_FN = os.path.join(METADATA_DIR, 'final_user2idx_segGE20.json')
FINAL_IDX2USER_SEGLT20_FN = os.path.join(METADATA_DIR, 'final_idx2user_segLT20.json')
FINAL_IDX2USER_SEGGE20_FN = os.path.join(METADATA_DIR, 'final_idx2user_segGE20.json')

USER2EMB_SEGGE20_FN = os.path.join(METADATA_DIR, 'user2embedding_segGE20.json')
BRAND2EMB_SEGGE20_FN = os.path.join(METADATA_DIR, 'brand2embedding_segGE20.json')
USER2EMB_SEGLT20_FN = os.path.join(METADATA_DIR, 'user2embedding_segLT20.json')
BRAND2EMB_SEGLT20_FN = os.path.join(METADATA_DIR, 'brand2embedding_segLT20.json')

ITEM2EPOCH_1_FN = os.path.join(METADATA_DIR, 'item2epoch_1.json')
ITEM2EPOCH_2_FN = os.path.join(METADATA_DIR, 'item2epoch_2.json')
ITEM2EPOCH_3_FN = os.path.join(METADATA_DIR, 'item2epoch_3.json')

USER2EPOCH_SEGGE20_1_FN = os.path.join(METADATA_DIR, 'user2epoch_segGE20_1.json')
USER2EPOCH_SEGGE20_2_FN = os.path.join(METADATA_DIR, 'user2epoch_segGE20_2.json')
USER2EPOCH_SEGGE20_3_FN = os.path.join(METADATA_DIR, 'user2epoch_segGE20_3.json')
USER2EPOCH_SEGLT20_1_FN = os.path.join(METADATA_DIR, 'user2epoch_segLT20_1.json')
USER2EPOCH_SEGLT20_2_FN = os.path.join(METADATA_DIR, 'user2epoch_segLT20_2.json')
USER2EPOCH_SEGLT20_3_FN = os.path.join(METADATA_DIR, 'user2epoch_segLT20_3.json')
USER2EPOCH_SEGLT20_4_FN = os.path.join(METADATA_DIR, 'user2epoch_segLT20_4.json')
USER2EPOCH_SEGLT20_5_FN = os.path.join(METADATA_DIR, 'user2epoch_segLT20_5.json')

ITEM2IDX_FN = os.path.join(METADATA_DIR, 'item2idx.json')
IDX2ITEM_FN = os.path.join(METADATA_DIR, 'idx2item.json')
ONT2IDX_FN = os.path.join(METADATA_DIR, 'ont2idx.json')
IDX2ONT_FN = os.path.join(METADATA_DIR, 'idx2ont.json')
BRAND2IDX_FN = os.path.join(METADATA_DIR, 'brand2idx.json')
IDX2BRAND_FN = os.path.join(METADATA_DIR, 'idx2brand.json')

PDT_MAPPING_FN = os.path.join(METADATA_DIR, 'pdt_mapping.json')

ONT_PDT_MAPPING_FN = os.path.join(METADATA_DIR, 'ont_pdt_mapping.json')
BRAND_PDT_MAPPING_FN = os.path.join(METADATA_DIR, 'brand_pdt_mapping.json')

USER_PDT_MAPPING_FN = os.path.join(METADATA_DIR, 'user_pdt_mapping.json')
USER_ONT_MAPPING_FN = os.path.join(METADATA_DIR, 'user_ont_mapping.json')

USER_BRAND_MAPPING_FN = os.path.join(METADATA_DIR, 'user_brand_mapping.json')
USER_BRAND_MAPPING_SEGLT20_1_FN = os.path.join(METADATA_DIR, 'user_brand_mapping_segLT20_1.json')
USER_BRAND_MAPPING_SEGLT20_2_FN = os.path.join(METADATA_DIR, 'user_brand_mapping_segLT20_2.json')
USER_BRAND_MAPPING_SEGLT20_3_FN = os.path.join(METADATA_DIR, 'user_brand_mapping_segLT20_3.json')
USER_BRAND_MAPPING_SEGLT20_4_FN = os.path.join(METADATA_DIR, 'user_brand_mapping_segLT20_4.json')

USER_ONT_MAPPING_SEGLT20_1_FN = os.path.join(METADATA_DIR, 'user_ont_mapping_segLT20_1.json')
USER_ONT_MAPPING_SEGLT20_2_FN = os.path.join(METADATA_DIR, 'user_ont_mapping_segLT20_2.json')
USER_ONT_MAPPING_SEGLT20_3_FN = os.path.join(METADATA_DIR, 'user_ont_mapping_segLT20_3.json')
USER_ONT_MAPPING_SEGLT20_4_FN = os.path.join(METADATA_DIR, 'user_ont_mapping_segLT20_4.json')

USER_BASELINE_FEATS_FN = os.path.join(BASELINE_FEATS_DIR, 'user_baseline_feats.json')
USER_NUM_INTERACTIONS_FN = os.path.join(BASELINE_FEATS_DIR, 'user_num_interactions.json')
MAPPED_USER_BASELINE_FEATS_SEGLT20_1_FN = os.path.join(
    BASELINE_FEATS_DIR, 'mapped_user_baseline_feats_segLT20_1.json')
MAPPED_USER_BASELINE_FEATS_SEGLT20_2_FN = os.path.join(
    BASELINE_FEATS_DIR, 'mapped_user_baseline_feats_segLT20_2.json')
MAPPED_USER_BASELINE_FEATS_SEGLT20_3_FN = os.path.join(
    BASELINE_FEATS_DIR, 'mapped_user_baseline_feats_segLT20_3.json')
MAPPED_USER_BASELINE_FEATS_SEGLT20_4_FN = os.path.join(
    BASELINE_FEATS_DIR, 'mapped_user_baseline_feats_segLT20_4.json')
MAPPED_USER_BASELINE_FEATS_SEGLT20_5_FN = os.path.join(
    BASELINE_FEATS_DIR, 'mapped_user_baseline_feats_segLT20_5.json')
MAPPED_USER_BASELINE_FEATS_SEGGE20_1_FN = os.path.join(
    BASELINE_FEATS_DIR, 'mapped_user_baseline_feats_segGE20_1.json')
MAPPED_USER_BASELINE_FEATS_SEGGE20_2_FN = os.path.join(
    BASELINE_FEATS_DIR, 'mapped_user_baseline_feats_segGE20_2.json')
MAPPED_USER_BASELINE_FEATS_SEGGE20_3_FN = os.path.join(
    BASELINE_FEATS_DIR, 'mapped_user_baseline_feats_segGE20_3.json')

ITEM_BASELINE_FEATS_FN = os.path.join(BASELINE_FEATS_DIR,
                                      'item_baseline_feats.json')
MAPPED_ITEM_BASELINE_FEATS_1_FN = os.path.join(
    BASELINE_FEATS_DIR, 'mapped_item_baseline_feats_1.json')
MAPPED_ITEM_BASELINE_FEATS_2_FN = os.path.join(
    BASELINE_FEATS_DIR, 'mapped_item_baseline_feats_2.json')
MAPPED_ITEM_BASELINE_FEATS_3_FN = os.path.join(
    BASELINE_FEATS_DIR, 'mapped_item_baseline_feats_3.json')

MODEL_SEGLT20_FN = os.path.join(MODEL_DIR, 'Class_model_SegLT20_E1_ckpt.pt')
MODEL_SEGGE20_FN = os.path.join(MODEL_DIR, 'Class_model_SegGE20_E1_ckpt.pt')

USER_COL = 'uuid'
ITEM_COL = 'sourceprodid'
ONTOLOGY_COL = 'ontology'
BRAND_COL = 'brand'
PRICE_COL = 'price'
DV_COL = 'userevent'
DATE_COL = 'clicked_epoch'
CAT_COLS = [USER_COL, ITEM_COL, 'ontology', 'brand']
NUM_COLS = ['price']
LOG_DIR = DATA_DIR
LOG_FILE = os.path.join(LOG_DIR, 'app.log')
PORT = 8880
CLASS_LABELS = ['pageView', 'addToCart', 'buy']
