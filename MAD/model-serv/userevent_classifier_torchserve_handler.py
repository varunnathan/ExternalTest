from abc import ABC
import os, sys, json, logging

import torch
import torch.nn.functional as F
#sys.path.append('../offline/src/')
from network import ProductRecommendationModel
from constants import *
from model_utils import choose_embedding_size
from model_params import *

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class ModelHandler(BaseHandler, ABC):
    """
    handler class for the userevent classifier
    """
    def __init__(self):

        super(ModelHandler, self).__init__()
        self.initialized = False

    def initialize(self, context):

        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        model_name = self.manifest['model']['modelName']
        serialized_file = self.manifest['model']['serializedFile']
        model_fn = os.path.join(model_dir, serialized_file)

        logger.debug("model_dir: {}; model_name: {}; serialized_file: {}".format(
            model_dir, model_name, serialized_file))

        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if
                                   torch.cuda.is_available() else "cpu")

        logger.debug("Instantiate base class for the model")
        if model_name.find('GE20') == -1:
            cat_cols = [ITEM_COL, ONTOLOGY_COL, BRAND_COL]
            cat_num_values = [N_ITEMS, N_ONTOLOGIES, N_BRANDS]
        else:
            cat_cols = [USER_COL, ITEM_COL, ONTOLOGY_COL, BRAND_COL]
            cat_num_values = [N_USERS_SEGGE20, N_ITEMS, N_ONTOLOGIES, N_BRANDS]

        embedding_sizes = choose_embedding_size(cat_cols, cat_num_values,
                                                EMB_DIM)
        self.model = ProductRecommendationModel(embedding_sizes, N_CONT, N_CLASSES)

        logger.debug("load ckpt and state_dict")
        ckpt = torch.load(model_fn, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        logger.debug('Model file {0} loaded successfully'.format(model_fn))
        self.initialized = True

    def preprocess(self, data):

        cat_feats = data.get('cat_feats')
        numeric_feats = data.get('numeric_feats')

        logger.debug('convert feature lists to tensors')
        cat_feat_tensor = torch.tensor(cat_feats)
        if cat_feat_tensor.dim() == 1:
            cat_feat_tensor = cat_feat_tensor.view(1, cat_feat_tensor.size()[0])

        numeric_feat_tensor = torch.tensor(numeric_feats)
        if numeric_feat_tensor.dim() == 1:
            numeric_feat_tensor = numeric_feat_tensor.view(
                1, numeric_feat_tensor.size()[0])
        return cat_feat_tensor, numeric_feat_tensor

    def inference(self, x1, x2):

        return self.model(x1.to(self.device), x2.to(self.device))

    def postprocess(self, inference_output):

        probs = F.softmax(
            inference_output, dim=1
        )
        return probs.cpu().detach().numpy()


_service = ModelHandler()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    x1, x2 = _service.preprocess(data)
    out = _service.inference(x1, x2)
    probs = _service.postprocess(out)

    return probs
