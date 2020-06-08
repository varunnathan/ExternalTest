import os, sys, json, logging

import torch
import torch.nn.functional as F
sys.path.append('/Users/varunn/Documents/ExternalTest/MAD/offline/src/')
from network import ProductRecommendationModel
from constants import *
from model_utils import choose_embedding_size
from model_params import *


class Model(object):
    """
    This class provides utility methods for loading pytorch models and inference
    """
    def __init__(self):
        self.model_segLT20 = self.initialize('LT20')
        self.model_segGE20 = self.initialize('GE20')
        self.model_segLT20.eval()
        self.model_segGE20.eval()

    def initialize(self, segment):

        logging.info("load model")

        logging.info("define model path and arguments for the base class")
        if segment == 'LT20':
            model_fn = MODEL_SEGLT20_FN
            cat_cols = [ITEM_COL, ONTOLOGY_COL, BRAND_COL]
            cat_num_values = [N_ITEMS, N_ONTOLOGIES, N_BRANDS]
        else:
            model_fn = MODEL_SEGGE20_FN
            cat_cols = [USER_COL, ITEM_COL, ONTOLOGY_COL, BRAND_COL]
            cat_num_values = [N_USERS_SEGGE20, N_ITEMS, N_ONTOLOGIES, N_BRANDS]

        logging.info("model_fn: {}".format(model_fn))

        embedding_sizes = choose_embedding_size(cat_cols, cat_num_values,
                                                EMB_DIM)
        model = ProductRecommendationModel(embedding_sizes, N_CONT, N_CLASSES)
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if
                                   torch.cuda.is_available() else "cpu")

        logging.info("load ckpt and state_dict")
        ckpt = torch.load(model_fn, map_location=self.device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(self.device)

        logging.info('Model file {0} loaded successfully'.format(model_fn))
        return model

    def preprocess(self, data):

        cat_feats = data.get('cat_feats')
        numeric_feats = data.get('numeric_feats')

        logging.info('convert feature lists to tensors')
        cat_feat_tensor = torch.tensor(cat_feats)
        if cat_feat_tensor.dim() == 1:
            cat_feat_tensor = cat_feat_tensor.view(1, cat_feat_tensor.size()[0])

        numeric_feat_tensor = torch.tensor(numeric_feats)
        if numeric_feat_tensor.dim() == 1:
            numeric_feat_tensor = numeric_feat_tensor.view(
                1, numeric_feat_tensor.size()[0])
        return cat_feat_tensor, numeric_feat_tensor

    def inference(self, model, data):

        logging.info("preprocessing")
        x1, x2 = self.preprocess(data)

        logging.info("prediction")
        out = model(x1.to(self.device), x2.to(self.device))

        logging.info("postprocessing")
        probs = self.postprocess(out)
        return probs

    def postprocess(self, inference_output):

        probs = F.softmax(
            inference_output, dim=1
        )
        return probs.cpu().detach().numpy()

    def get_embedding(self, model, segment, kind, mapped_id):

        if segment == 'LT20':
            user_idx = None
            item_idx, ont_idx, brand_idx = range(3)
        else:
            user_idx, item_idx, ont_idx, brand_idx = range(4)

        kindidx_dct = {'user': user_idx, 'item': item_idx,
                       'ontology': ont_idx, 'brand': brand_idx}

        return model.embeddings[kindidx_dct[kind]].weight.data[
            mapped_id].cpu().detach().numpy().tolist()
