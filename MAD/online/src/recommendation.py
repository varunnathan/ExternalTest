import numpy as np
import os, sys, json, logging
import redis
from sklearn.metrics.pairwise import cosine_similarity
from settings import *
sys.path.append("../../offline/src/")
from constants import *
from baseline_feats_utils import feat_type_feats_dct
from model_params import (N_USERS, N_USERS_SEGLT20, N_USERS_SEGGE20, N_ITEMS,
                          NUMERIC_FEATS)


class Recommendation(object):
    """
    helper class to recommend items to a given user
    """

    def __init__(self, model, n_candidates):
        self.model = model
        self.n_candidates = n_candidates

        logging.info('init redis DB')
        self.redis_db = redis.StrictRedis(host=DB_HOST, port=DB_PORT, db=DB_NO)

        logging.info('read user_num_interactions dct')
        self.num_interactions_dct = json.load(open(USER_NUM_INTERACTIONS_FN))

        logging.info('read idx2item dct')
        self.idx2item = json.load(open(IDX2ITEM_FN))

        logging.info('read brand pdt mapping dct - %s' % (BRAND_PDT_MAPPING_FN))
        self.brand2pdt = json.load(open(BRAND_PDT_MAPPING_FN))

        logging.info('read pdt mapping dct - %s' % (PDT_MAPPING_FN))
        self.pdtmap_dct = json.load(open(PDT_MAPPING_FN))

        logging.info('read item2epoch dct')
        self.item2epoch_1 = json.load(open(ITEM2EPOCH_1_FN))
        self.item2epoch_2 = json.load(open(ITEM2EPOCH_2_FN))
        self.item2epoch_3 = json.load(open(ITEM2EPOCH_3_FN))

        logging.info('read item baseline feats dct')
        self.item_baseline_feats1 = json.load(open(
            MAPPED_ITEM_BASELINE_FEATS_1_FN))
        self.item_baseline_feats2 = json.load(open(
            MAPPED_ITEM_BASELINE_FEATS_2_FN))
        self.item_baseline_feats3 = json.load(open(
            MAPPED_ITEM_BASELINE_FEATS_3_FN))

        logging.info('read brand2embedding_segGE20 file')
        self.brand2emb = json.load(open(BRAND2EMB_SEGGE20_FN))

    def _get_user_attributes(self, user_id, num_user_interactions):
        """
        returns the model segment and mapped_user_id
        """
        segment = 'LT20' if num_user_interactions < 20 else 'GE20'
        mapped_user_id = int(self.redis_db["user2idx::{}::{}".format(
            segment, str(user_id))])

        return mapped_user_id, segment

    def _get_user_baseline_feats(self, mapped_user_id, clicked_epoch, segment):

        feats = {}
        for feat_pos, feat_name in enumerate(feat_type_feats_dct['user']):
            val = float(self.redis_db.lindex(
                str(segment)+'::'+str(mapped_user_id), feat_pos))
            key = 'uuid_'+feat_name
            if feat_name == 'earliest_interaction_date':
                key = 'uuid_days_since_earliest_interaction'
                val = (float(clicked_epoch)-float(val))/(60*60*24)
                if val < 0:
                    val = -1
            feats[key] = val

        return feats

    def _get_item_baseline_feats(self, mapped_item_id, clicked_epoch):
        """
        Input: user/item, clicked_epoch and user_feats_dct/item_feats_dct
        Returns: dictionary of user/item baseline features
        """
        #logging.info('%s features' % (out_type))
        ent_col = 'sourceprodid'
        if str(mapped_item_id) in self.item_baseline_feats1:
            ent_feats = self.item_baseline_feats1
        elif str(mapped_item_id) in self.item_baseline_feats2:
            ent_feats = self.item_baseline_feats2
        else:
            ent_feats = self.item_baseline_feats3

        feats = {}
        for feat_pos, feat_name in enumerate(feat_type_feats_dct['item']):
            val = ent_feats[str(mapped_item_id)][feat_pos]
            key = ent_col+'_'+feat_name
            if feat_name == 'earliest_interaction_date':
                key = ent_col+'_days_since_earliest_interaction'
                val = (float(clicked_epoch)-float(val))/(60*60*24)
                if val < 0:
                    val = -1
            feats[key] = val

        return feats

    def _find_file_for_epoch_mapping(self, ent, mapped_ent_id, segment=None):
        """
        returns the file name which has the user/item and epoch mapping
        """
        k = int(mapped_ent_id)
        if ent == 'item':
            size_per_dct = N_ITEMS//3
            if k <= size_per_dct:
                return ITEM2EPOCH_1_FN
            elif k <= 2*size_per_dct:
                return ITEM2EPOCH_2_FN
            else:
                return ITEM2EPOCH_3_FN

        elif ent == 'user':
            if segment == 'LT20':
                size_per_dct = N_USERS_SEGLT20//5
                if k <= size_per_dct:
                    return USER2EPOCH_SEGLT20_1_FN
                elif k <= 2*size_per_dct:
                    return USER2EPOCH_SEGLT20_2_FN
                elif k <= 3*size_per_dct:
                    return USER2EPOCH_SEGLT20_3_FN
                elif k <= 4*size_per_dct:
                    return USER2EPOCH_SEGLT20_4_FN
                else:
                    return USER2EPOCH_SEGLT20_5_FN

            elif segment == 'GE20':
                size_per_dct = N_USERS_SEGGE20//3
                if k <= size_per_dct:
                    return USER2EPOCH_SEGGE20_1_FN
                elif k <= 2*size_per_dct:
                    return USER2EPOCH_SEGGE20_2_FN
                else:
                    return USER2EPOCH_SEGGE20_3_FN

    def _find_file_for_user_brand_mapping(self, mapped_userid):
        """
        returns the file name which has the user-brand mapping
        """
        size_per_dct = N_USERS_SEGLT20//4
        u = int(mapped_userid)
        if u <= size_per_dct:
            return USER_BRAND_MAPPING_SEGLT20_1_FN
        elif u <= 2*size_per_dct:
            return USER_BRAND_MAPPING_SEGLT20_2_FN
        elif u <= 3*size_per_dct:
            return USER_BRAND_MAPPING_SEGLT20_3_FN
        else:
            return USER_BRAND_MAPPING_SEGLT20_4_FN

    def _find_file_for_baseline_feats_mapping(self, out_type, mapped_userid,
                                              segment=None):
        """
        returns the file name which has the user-baseline_feats mapping
        """
        if out_type == 'item':
            size_per_dct = N_ITEMS//3
            if mapped_userid <= size_per_dct:
                return MAPPED_ITEM_BASELINE_FEATS_1_FN
            elif mapped_userid <= 2*size_per_dct:
                return MAPPED_ITEM_BASELINE_FEATS_2_FN
            else:
                return MAPPED_ITEM_BASELINE_FEATS_3_FN

        elif out_type == 'user':
            if segment == 'LT20':
                size_per_dct = N_USERS_SEGLT20//5
                if mapped_userid <= size_per_dct:
                    return MAPPED_USER_BASELINE_FEATS_SEGLT20_1_FN
                elif mapped_userid <= 2*size_per_dct:
                    return MAPPED_USER_BASELINE_FEATS_SEGLT20_2_FN
                elif mapped_userid <= 3*size_per_dct:
                    return MAPPED_USER_BASELINE_FEATS_SEGLT20_3_FN
                elif mapped_userid <= 4*size_per_dct:
                    return MAPPED_USER_BASELINE_FEATS_SEGLT20_4_FN
                else:
                    return MAPPED_USER_BASELINE_FEATS_SEGLT20_5_FN

            elif segment == 'GE20':
                size_per_dct = N_USERS_SEGGE20//3
                if mapped_userid <= size_per_dct:
                    return MAPPED_USER_BASELINE_FEATS_SEGGE20_1_FN
                elif mapped_userid <= 2*size_per_dct:
                    return MAPPED_USER_BASELINE_FEATS_SEGGE20_2_FN
                else:
                    return MAPPED_USER_BASELINE_FEATS_SEGGE20_3_FN

    def _find_candidate_brands_segLT20(self, mapped_userid):
        """
        returns the candidate brands based on the user's past purchasing patterns
        """

        logging.info('candidate selection process begins')
        candidates = []
        count = 0
        for k in ['buy', 'addToCart', 'pageView']:
            db_k = "brand::LT20::{}::{}".format(k, str(mapped_userid))
            candidate = self.redis_db.lrange(db_k, 0, -1)
            if candidate:
                candidate = [int(x) for x in candidate]
            count += len(candidate)
            candidates += candidate
            if count >= self.n_candidates:
                candidates = candidates[:self.n_candidates]
                break
        assert len(candidates) <= self.n_candidates
        return candidates

    def _pick_model(self, segment):
        """
        returns the model object based on segment
        """
        if segment == 'LT20':
            model = self.model.model_segLT20
        elif segment == 'GE20':
            model = self.model.model_segGE20
        return model

    def _find_candidate_brands_segGE20(self, mapped_userid):
        """
        returns the candidate brands based on trained embeddings of user and brand
        """
        logging.info('get user embeddings')
        model = self.model.model_segGE20
        segment = 'GE20'
        user_embedding = self.model.get_embedding(model, segment, 'user',
                                                  mapped_userid)
        user_embedding = np.array(user_embedding).reshape((1, len(user_embedding)))

        logging.info('candidate selection process begins')
        keys = list(self.brand2emb.keys())
        embs = np.array(list(self.brand2emb.values()))
        sims = cosine_similarity(user_embedding, embs)
        sims = sims.reshape(sims.shape[1])
        candidates = sims.argsort()[::-1][:self.n_candidates]
        candidates = [keys[idx] for idx in candidates]
        return candidates

    def _item_ids_from_brands(self, brand_lst):
        """
        returns item_id list for a given list of brands
        """
        out = []
        prices = []
        for brand in brand_lst:
            item_lst = self.brand2pdt[str(brand)]
            out += [[i]+self.pdtmap_dct[str(i)][:2] for i in item_lst]
            prices += [self.pdtmap_dct[str(i)][2] for i in item_lst]
        return out, prices

    def recommend(self, user_id, n_items_recommended):
        """
        returns a list of recommended item_ids
        """

        logging.info('get uuid_num_interactions')
        num_user_interactions = self.num_interactions_dct[user_id]
        logging.info('num_user_interactions: %d' % (num_user_interactions))

        logging.info('get the segment and mapped user_id')
        self.mapped_user_id, self.segment = self._get_user_attributes(
            user_id, num_user_interactions)
        logging.info('mapped_user_id: %d' % (self.mapped_user_id))
        logging.info('segment: %s' % (self.segment))

        logging.info('get candidate brands')
        if self.segment == 'LT20':
            candidate_brands = self._find_candidate_brands_segLT20(self.mapped_user_id)
        elif self.segment == 'GE20':
            candidate_brands = self._find_candidate_brands_segGE20(self.mapped_user_id)
        logging.info('num candidate_brands: %d' % (len(candidate_brands)))

        logging.info('get clicked_epoch for user')
        user_clicked_epoch = int(self.redis_db.get("epoch::{}::{}".format(
            str(self.segment), str(self.mapped_user_id))))

        logging.info('get item_id, ontology and price for the candidate brands')
        cat_lst, prices = self._item_ids_from_brands(candidate_brands)
        logging.info('num candidate items: %d' % (len(cat_lst)))

        logging.info('get clicked_epoch for items')
        item_clicked_epochs = []
        for i, cat_item in enumerate(cat_lst):
            if i % 10 == 0:
                logging.info('num completed: %d' % (i))
            mapped_item_id = cat_item[0]
            if str(mapped_item_id) in self.item2epoch_1:
                item_clicked_epoch = self.item2epoch_1[str(mapped_item_id)]
            elif str(mapped_item_id) in self.item2epoch_2:
                item_clicked_epoch = self.item2epoch_2[str(mapped_item_id)]
            else:
                item_clicked_epoch = self.item2epoch_3[str(mapped_item_id)]

            item_clicked_epochs.append(item_clicked_epoch)

        logging.info('calculate numeric feats')
        numeric_feats = []

        logging.info('get user baseline features')
        feats = self._get_user_baseline_feats(
            mapped_user_id=self.mapped_user_id,
            clicked_epoch=user_clicked_epoch, segment=self.segment)

        logging.info('get item baseline features')
        for i, item_clicked_epoch in enumerate(item_clicked_epochs):
            if i % 10 == 0:
                logging.info('num completed: %d' % (i))
            feats.update(self._get_item_baseline_feats(
                mapped_item_id=cat_lst[i][0], clicked_epoch=item_clicked_epoch))

            numeric_feat = [prices[i]] + [feats[col] for col in NUMERIC_FEATS[1:]]
            numeric_feats.append(numeric_feat)

        logging.info('calculate cat feats')
        if self.segment == 'LT20':
            cat_feats = cat_lst
        else:
            cat_feats = [[self.mapped_user_id]+i for i in cat_lst]

        logging.info('calculate pred_prob and rank')
        model = self._pick_model(self.segment)
        data = {'cat_feats': cat_feats, 'numeric_feats': numeric_feats}
        pred_probs = self.model.inference(model, data)
        buy_pred_probs = pred_probs[:, 2]
        indices = buy_pred_probs.argsort()[::-1][:n_items_recommended]
        recommended_items = [cat_lst[i][0] for i in indices]
        buy_pred_probs = [buy_pred_probs[i] for i in indices]
        recommended_items = [self.idx2item[str(x)] for x in recommended_items]

        return recommended_items, buy_pred_probs
