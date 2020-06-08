from constants import *

N_USERS = 10130223
N_USERS_SEGLT20 = 8686053
N_USERS_SEGGE20 = 1444170
N_ITEMS = 1175648
N_ONTOLOGIES = 801
N_BRANDS = 1686
EMB_DIM = 150
N_CLASSES = 3
N_CONT = 18
CAT_FEATS = [USER_COL, ITEM_COL, ONTOLOGY_COL, BRAND_COL]
NUMERIC_FEATS = [
    PRICE_COL, 'uuid_num_interactions', 'uuid_mean_price_interactions',
    'uuid_min_num_interactions_per_pdt', 'uuid_max_num_interactions_per_pdt',
    'uuid_mean_num_interactions_per_pdt', 'uuid_min_num_interactions_per_ont',
    'uuid_max_num_interactions_per_ont', 'uuid_mean_num_interactions_per_ont',
    'uuid_min_num_interactions_per_brand', 'uuid_max_num_interactions_per_brand',
    'uuid_mean_num_interactions_per_brand', 'uuid_days_since_earliest_interaction',
    'sourceprodid_num_interactions', 'sourceprodid_min_num_interactions_per_user',
    'sourceprodid_max_num_interactions_per_user',
    'sourceprodid_mean_num_interactions_per_user',
    'sourceprodid_days_since_earliest_interaction']
