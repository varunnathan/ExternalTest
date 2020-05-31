import os


LOCAL_DIR = '/data/'
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
ITEM2IDX_FN = os.path.join(METADATA_DIR, 'item2idx.json')
IDX2ITEM_FN = os.path.join(METADATA_DIR, 'idx2item.json')
ONT2IDX_FN = os.path.join(METADATA_DIR, 'ont2idx.json')
IDX2ONT_FN = os.path.join(METADATA_DIR, 'idx2ont.json')
BRAND2IDX_FN = os.path.join(METADATA_DIR, 'brand2idx.json')
IDX2BRAND_FN = os.path.join(METADATA_DIR, 'idx2brand.json')
PDT_MAPPING_FN = os.path.join(METADATA_DIR, 'pdt_mapping.json')
USER_PDT_MAPPING_FN = os.path.join(METADATA_DIR, 'user_pdt_mapping.json')
USER_ONT_MAPPING_FN = os.path.join(METADATA_DIR, 'user_ont_mapping.json')
USER_BRAND_MAPPING_FN = os.path.join(METADATA_DIR, 'user_brand_mapping.json')
USER_BASELINE_FEATS_FN = os.path.join(BASELINE_FEATS_DIR, 'user_baseline_feats.json')
ITEM_BASELINE_FEATS_FN = os.path.join(BASELINE_FEATS_DIR, 'item_baseline_feats.json')
USER_COL = 'uuid'
ITEM_COL = 'sourceprodid'
ONTOLOGY_COL = 'ontology'
BRAND_COL = 'brand'
PRICE_COL = 'price'
DV_COL = 'userevent'
DATE_COL = 'clicked_epoch'
CAT_COLS = [USER_COL, ITEM_COL, 'ontology', 'brand']
NUM_COLS = ['price']