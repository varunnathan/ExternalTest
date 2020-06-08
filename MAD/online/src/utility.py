import torch
import sys
sys.path.append("../../offline/src/")
from network import ProductRecommendationModel
from model_utils import choose_embedding_size
from model_params import *


def load_model(segment, user_col, item_col, ontology_col, brand_col, model_fn):

    if segment != 'GE20':
        cat_cols = [item_col, ontology_col, brand_col]
        cat_num_values = [N_ITEMS, N_ONTOLOGIES, N_BRANDS]
    else:
        cat_cols = [user_col, item_col, ontology_col, brand_col]
        cat_num_values = [N_USERS_SEGGE20, N_ITEMS, N_ONTOLOGIES, N_BRANDS]

    print('define embedding sizes\n')
    embedding_sizes = choose_embedding_size(cat_cols, cat_num_values, EMB_DIM)

    print('model class instantiation\n')
    model = ProductRecommendationModel(embedding_sizes, N_CONT, N_CLASSES)

    print('load state dict')
    ckpt = torch.load(model_fn, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model_state_dict'])

    return model


def format_response(message=None, pred_class=None, pred_prob=None,
                    embeddings=None, recommended_items=None):
    out = {}
    if message is not None:
        out["message"] = message
    if pred_class is not None:
        out["pred_class"] = pred_class
    if pred_prob is not None:
        out["pred_prob"] = pred_prob
    if embeddings is not None:
        out["embeddings"] = embeddings
    if recommended_items is not None:
        out["recommended_items"] = recommended_items

    return out
