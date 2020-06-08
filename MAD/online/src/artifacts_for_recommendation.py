import pandas as pd
import os, sys, json, time, argparse, logging
import torch
import collections
from itertools import chain
from utility import load_model
sys.path.append("../../offline/src/")
from constants import *
from metadata_utils import _helper_for_cat_pdt_mapping
from model_params import N_USERS, N_USERS_SEGLT20, N_USERS_SEGGE20, N_ITEMS


def _helper_for_user_idx_mapping(segment):

    logging.info('Segment: %s' % (segment))
    logging.info('\n')

    logging.info('read file %s' % (IDX2USER_FN))
    idx2user = json.load(open(IDX2USER_FN))

    logging.info('read file %s' % (USER2IDX_SEGGE20_FN))
    user2idx_seg = json.load(open(USER2IDX_SEGGE20_FN))

    if segment == 'LT20':

        logging.info('get the users in segment GE20\n')
        users_ge20 = set(user2idx_seg.keys())

        logging.info('get the users in segment LT20\n')
        users_lt20 = list(set(idx2user.keys()) - users_ge20)

        logging.info('filter global dct based on users in LT20 segment\n')
        final_dct = {idx2user[v]: int(v) for v in users_lt20}

    elif segment == 'GE20':

        final_dct = {idx2user[k]: v for k, v in user2idx_seg.items()}

    logging.info('final_dct_inv')
    final_dct_inv = {v: k for k, v in final_dct.items()}

    logging.info('save')
    if segment == 'LT20':
        save_fn = FINAL_USER2IDX_SEGLT20_FN
        inv_save_fn = FINAL_IDX2USER_SEGLT20_FN
    else:
        save_fn = FINAL_USER2IDX_SEGGE20_FN
        inv_save_fn = FINAL_IDX2USER_SEGGE20_FN
    json.dump(final_dct, open(save_fn, 'w'))
    json.dump(final_dct_inv, open(inv_save_fn, 'w'))


def _helper_for_emb_mapping(segment, cat):

    logging.info('Category: %s' % (cat))
    logging.info('\n')

    logging.info('load model object')
    model_fn = MODEL_SEGLT20_FN if segment == 'LT20' else MODEL_SEGGE20_FN
    model = load_model(segment, USER_COL, ITEM_COL, ONTOLOGY_COL, BRAND_COL,
                       model_fn)

    logging.info('read idx2cat file')
    if cat == 'user':
        idx = 0
        if segment == 'LT20':
            idx2cat_fn = FINAL_IDX2USER_SEGLT20_FN
            save_fn = USER2EMB_SEGLT20_FN
        else:
            idx2cat_fn = FINAL_IDX2USER_SEGGE20_FN
            save_fn = USER2EMB_SEGGE20_FN
    elif cat == 'brand':
        idx = 3
        idx2cat_fn = IDX2BRAND_FN
        save_fn = BRAND2EMB_SEGGE20_FN if segment == 'GE20' else BRAND2EMB_SEGLT20_FN
    idx2cat = json.load(open(idx2cat_fn))

    final_dct = {v: model.embeddings[idx].weight.data[int(k)].tolist()
                 for k, v in idx2cat.items()}

    logging.info('save')
    json.dump(final_dct, open(save_fn, 'w'))


def _combine_dcts(d1, d2):

    Cdict = collections.defaultdict(int)

    for key, val in chain(d1.items(), d2.items()):
        Cdict[key] = max(val, Cdict[key])

    return Cdict


def _helper_for_item_latest_epoch_mapping(interim_data_dir=INTERIM_DATA_DIR):

    print('get the list of files to be processed\n')
    files = [os.path.join(interim_data_dir, fn) for fn in
             os.listdir(interim_data_dir)]

    global_dct = {}
    for i, fn in enumerate(files):
        if i % 2 == 0:
            print('num files completed: %d' % (i))
            print('\n\n')
        print('reading file %s' % (fn))
        df = pd.read_csv(fn, compression='gzip', sep='|')
        print('shape: ', df.shape)

        print('deduplication\n')
        tmp_df = df[[ITEM_COL, DATE_COL]]
        del df
        tmp_df.drop_duplicates(inplace=True)
        tmp_df.reset_index(drop=True, inplace=True)

        print('groupby and aggregation\n')
        table = tmp_df.groupby(ITEM_COL)[DATE_COL].max().rename(
            'latest_ts').reset_index()
        local_dct = dict(zip(table[ITEM_COL], table['latest_ts']))
        del table

        print('combine global and local dcts')
        global_dct = _combine_dcts(global_dct, local_dct)

        print('size of global dct: %d' % (len(global_dct)))

    print('split and save')
    out_fn = [ITEM2EPOCH_1_FN, ITEM2EPOCH_2_FN, ITEM2EPOCH_3_FN]
    d1, d2, d3 = {}, {}, {}
    size_per_dct = N_ITEMS//3
    for k, v in global_dct.items():
        k = int(k)
        if k <= size_per_dct:
            d1[k] = v
        elif k <= 2*size_per_dct:
            d2[k] = v
        else:
            d3[k] = v

    out = [d1, d2, d3]
    for i, item in enumerate(out):
        json.dump(item, open(out_fn[i], 'w'))


def _helper_for_user_latest_epoch_mapping(segment,
                                          interim_data_dir=INTERIM_DATA_DIR,
                                          user2idx_seg_fn=USER2IDX_SEGGE20_FN):
    print('read user2idx_seg\n')
    user2idx_seg = json.load(open(user2idx_seg_fn))

    print('get the list of files to be processed\n')
    files = [os.path.join(interim_data_dir, fn) for fn in
             os.listdir(interim_data_dir)]

    global_dct = {}
    for i, fn in enumerate(files):
        if i % 2 == 0:
            print('num files completed: %d' % (i))
            print('\n\n')
        print('reading file %s' % (fn))
        df = pd.read_csv(fn, compression='gzip', sep='|')
        print('shape before filtering: ', df.shape)

        print('filter based on segment')
        keys = [int(x) for x in list(user2idx_seg.keys())]
        mask = df[USER_COL].isin(keys)
        if segment == 'LT20':
            df = df.loc[~mask, :]
        elif segment == 'GE20':
            df = df.loc[mask, :]
        df.reset_index(drop=True, inplace=True)
        print('shape after filtering: ', df.shape)

        if segment == 'GE20':
            print('add uuid_mapped column\n')
            df['uuid_mapped'] = df[USER_COL].apply(
                lambda x: user2idx_seg[str(x)])
        else:
            df['uuid_mapped'] = df[USER_COL]

        print('deduplication\n')
        tmp_df = df[['uuid_mapped', DATE_COL]]
        del df
        tmp_df.drop_duplicates(inplace=True)
        tmp_df.reset_index(drop=True, inplace=True)

        print('groupby and aggregation\n')
        table = tmp_df.groupby('uuid_mapped')[DATE_COL].max().rename(
            'latest_ts').reset_index()
        local_dct = dict(zip(table['uuid_mapped'], table['latest_ts']))
        del table

        print('combine global and local dcts')
        global_dct = _combine_dcts(global_dct, local_dct)

        print('size of global dct: %d' % (len(global_dct)))

    print('split and save')
    if segment == 'LT20':
        out_fn = [USER2EPOCH_SEGLT20_1_FN, USER2EPOCH_SEGLT20_2_FN,
                  USER2EPOCH_SEGLT20_3_FN, USER2EPOCH_SEGLT20_4_FN,
                  USER2EPOCH_SEGLT20_5_FN]
        d1, d2, d3, d4, d5 = {}, {}, {}, {}, {}
        size_per_dct = N_USERS_SEGLT20//5
        for k, v in global_dct.items():
            k = int(k)
            if k <= size_per_dct:
                d1[k] = v
            elif k <= 2*size_per_dct:
                d2[k] = v
            elif k <= 3*size_per_dct:
                d3[k] = v
            elif k <= 4*size_per_dct:
                d4[k] = v
            else:
                d5[k] = v

        out = [d1, d2, d3, d4, d5]
        for i, item in enumerate(out):
            json.dump(item, open(out_fn[i], 'w'))

    elif segment == 'GE20':
        out_fn = [USER2EPOCH_SEGGE20_1_FN, USER2EPOCH_SEGGE20_2_FN,
                  USER2EPOCH_SEGGE20_3_FN]
        d1, d2, d3 = {}, {}, {}
        size_per_dct = N_USERS_SEGGE20//3
        for k, v in global_dct.items():
            k = int(k)
            if k <= size_per_dct:
                d1[k] = v
            elif k <= 2*size_per_dct:
                d2[k] = v
            else:
                d3[k] = v

        out = [d1, d2, d3]
        for i, item in enumerate(out):
            json.dump(item, open(out_fn[i], 'w'))


def _helper_for_user_brand_mapping(entity):

    logging.info("reading user2idx_dct - %s" % (FINAL_USER2IDX_SEGLT20_FN))
    user2idx_lt = json.load(open(FINAL_USER2IDX_SEGLT20_FN))

    if entity == 'brand':
        fn = BRAND2IDX_FN
    elif entity == 'ontology':
        fn = ONT2IDX_FN
    logging.info("reading ent2idx_dct - %s" % (fn))
    ent2idx = json.load(open(fn))

    if entity == 'brand':
        inp_fn = USER_BRAND_MAPPING_FN
        out_fn = [USER_BRAND_MAPPING_SEGLT20_1_FN,
                  USER_BRAND_MAPPING_SEGLT20_2_FN,
                  USER_BRAND_MAPPING_SEGLT20_3_FN,
                  USER_BRAND_MAPPING_SEGLT20_4_FN]
    elif entity == 'ontology':
        inp_fn = USER_ONT_MAPPING_FN
        out_fn = [USER_ONT_MAPPING_SEGLT20_1_FN,
                  USER_ONT_MAPPING_SEGLT20_2_FN,
                  USER_ONT_MAPPING_SEGLT20_3_FN,
                  USER_ONT_MAPPING_SEGLT20_4_FN]

    logging.info("reading input json %s" % (inp_fn))
    df = json.load(open(inp_fn))

    logging.info("mapping begins...")
    new_df = {}
    for k in df:
        v = df[k]
        new_df[k] = {}
        count = 0
        k_count = 0
        for u_k in v:
            if count % 100000 == 0:
                print(count)
            if u_k in user2idx_lt:
                new_df[k][user2idx_lt[u_k]] = [ent2idx[x] for x in v[u_k]]
                k_count += 1
            count += 1

    logging.info('split and save')
    d1, d2, d3, d4 = {}, {}, {}, {}
    size_per_dct = N_USERS_SEGLT20//4
    for k, v in new_df.items():
        d1[k], d2[k], d3[k], d4[k] = {}, {}, {}, {}
        for u_k in v:
            if int(u_k) <= size_per_dct:
                d1[k][u_k] = v[u_k]
            elif int(u_k) <= 2*size_per_dct:
                d2[k][u_k] = v[u_k]
            elif int(u_k) <= 3*size_per_dct:
                d3[k][u_k] = v[u_k]
            else:
                d4[k][u_k] = v[u_k]

    out = [d1, d2, d3, d4]
    for i, item in enumerate(out):
        json.dump(item, open(out_fn[i], 'w'))


def _helper_for_pdt_mapping():

    logging.info('reading pdt_mapping_dct %s' % (PDT_MAPPING_FN))
    df = json.load(open(PDT_MAPPING_FN))

    logging.info('reading item2idx_dct %s' % (ITEM2IDX_FN))
    item2idx = json.load(open(ITEM2IDX_FN))

    logging.info('reading ont2idx_dct %s' % (ONT2IDX_FN))
    ont2idx = json.load(open(ONT2IDX_FN))

    logging.info('reading brand2idx_dct %s' % (BRAND2IDX_FN))
    brand2idx = json.load(open(BRAND2IDX_FN))

    df1 = {}
    for k, v in df.items():
        new_v = [ont2idx[v[0]], brand2idx[v[1]], v[2]]
        df1[item2idx[k]] = new_v

    logging.info('save')
    json.dump(df1, open(PDT_MAPPING_FN, 'w'))


def _helper_for_baseline_feats_updation(entity):

    if entity == 'user':
        inp_fn = USER_BASELINE_FEATS_FN
        out_fn_num_interactions = USER_NUM_INTERACTIONS_FN
        out_fn_segLT20 = [
            MAPPED_USER_BASELINE_FEATS_SEGLT20_1_FN,
            MAPPED_USER_BASELINE_FEATS_SEGLT20_2_FN,
            MAPPED_USER_BASELINE_FEATS_SEGLT20_3_FN,
            MAPPED_USER_BASELINE_FEATS_SEGLT20_4_FN,
            MAPPED_USER_BASELINE_FEATS_SEGLT20_5_FN]
        out_fn_segGE20 = [
            MAPPED_USER_BASELINE_FEATS_SEGGE20_1_FN,
            MAPPED_USER_BASELINE_FEATS_SEGGE20_2_FN,
            MAPPED_USER_BASELINE_FEATS_SEGGE20_3_FN]
        user2idx_segGE20_fn = FINAL_USER2IDX_SEGGE20_FN
        user2idx_segLT20_fn = FINAL_USER2IDX_SEGLT20_FN

    elif entity == 'item':
        inp_fn = ITEM_BASELINE_FEATS_FN
        out_fn = [
            MAPPED_ITEM_BASELINE_FEATS_1_FN, MAPPED_ITEM_BASELINE_FEATS_2_FN,
            MAPPED_ITEM_BASELINE_FEATS_3_FN]
        item2idx_fn = ITEM2IDX_FN

    logging.info('reading file %s' % (inp_fn))
    df = json.load(open(inp_fn))

    if entity == 'user':
        logging.info('reading file %s' % (user2idx_segGE20_fn))
        user2idx_segGE20 = json.load(open(user2idx_segGE20_fn))

        logging.info('reading file %s' % (user2idx_segLT20_fn))
        user2idx_segLT20 = json.load(open(user2idx_segLT20_fn))

        logging.info('updation begins')
        num_interactions_dct = {}
        d1_LT20, d2_LT20, d3_LT20, d4_LT20, d5_LT20 = {}, {}, {}, {}, {}
        d1_GE20, d2_GE20, d3_GE20 = {}, {}, {}
        size_per_dct_LT20 = N_USERS_SEGLT20//5
        size_per_dct_GE20 = N_USERS_SEGGE20//3

        for k, v in df.items():
            num_interactions_dct[k] = v[0]
            if k in user2idx_segGE20:
                new_k = user2idx_segGE20[k]
                if new_k <= size_per_dct_GE20:
                    d1_GE20[new_k] = v
                elif new_k <= 2*size_per_dct_GE20:
                    d2_GE20[new_k] = v
                else:
                    d3_GE20[new_k] = v

            elif k in user2idx_segLT20:
                new_k = user2idx_segLT20[k]
                if new_k <= size_per_dct_LT20:
                    d1_LT20[new_k] = v
                elif new_k <= 2*size_per_dct_LT20:
                    d2_LT20[new_k] = v
                elif new_k <= 3*size_per_dct_LT20:
                    d3_LT20[new_k] = v
                elif new_k <= 4*size_per_dct_LT20:
                    d4_LT20[new_k] = v
                else:
                    d5_LT20[new_k] = v

        logging.info('save')

        logging.info('user_num_interactions')
        json.dump(num_interactions_dct, open(out_fn_num_interactions, 'w'))

        logging.info('segLT20')
        out = [d1_LT20, d2_LT20, d3_LT20, d4_LT20, d5_LT20]
        for i, item in enumerate(out):
            json.dump(item, open(out_fn_segLT20[i], 'w'))

        logging.info('segGE20')
        out = [d1_GE20, d2_GE20, d3_GE20]
        for i, item in enumerate(out):
            json.dump(item, open(out_fn_segGE20[i], 'w'))

    elif entity == 'item':
        logging.info('reading file %s' % (item2idx_fn))
        item2idx = json.load(open(item2idx_fn))

        logging.info('updation begins')
        d1, d2, d3 = {}, {}, {}
        size_per_dct = N_ITEMS//3

        for k, v in df.items():
            new_k = item2idx[k]
            if new_k <= size_per_dct:
                d1[new_k] = v
            elif new_k <= 2*size_per_dct:
                d2[new_k] = v
            else:
                d3[new_k] = v

        logging.info('save')
        out = [d1, d2, d3]
        for i, item in enumerate(out):
            json.dump(item, open(out_fn[i], 'w'))


def get_artifacts_for_recommendation(task):

    logging.info('Task: %s' % (task))

    if task == "get-ont-pdt-mapping":
        _helper_for_cat_pdt_mapping(cat=ONTOLOGY_COL)

    elif task == "get-brand-pdt-mapping":
        _helper_for_cat_pdt_mapping(cat=BRAND_COL)

    elif task == "get-user-index-mapping-LT20":
        _helper_for_user_idx_mapping(segment='LT20')

    elif task == "get-user-index-mapping-GE20":
        _helper_for_user_idx_mapping(segment='GE20')

    elif task == "get-user-emb-mapping-GE20":
        _helper_for_emb_mapping(segment='GE20', cat='user')

    elif task == "get-brand-emb-mapping-GE20":
        _helper_for_emb_mapping(segment='GE20', cat='brand')

    elif task == "get-latest-epoch-for-user":
        print('LT20')
        _helper_for_user_latest_epoch_mapping(segment='LT20')
        print('\n\n\n')
        print('GE20')
        _helper_for_user_latest_epoch_mapping(segment='GE20')

    elif task == "get-latest-epoch-for-item":
        _helper_for_item_latest_epoch_mapping()

    elif task == "get-user-brand-mapping-LT20":
        _helper_for_user_brand_mapping(entity='brand')

    elif task == "get-user-ont-mapping-LT20":
        _helper_for_user_brand_mapping(entity='ontology')

    elif task == "get-pdt-mapping":
        _helper_for_pdt_mapping()

    elif task == "get-user-baseline-feats":
        _helper_for_baseline_feats_updation(entity='user')

    elif task == "get-item-baseline-feats":
        _helper_for_baseline_feats_updation(entity='item')


if __name__ == '__main__':
    logging.info('artifacts for recommendation')

    parser = argparse.ArgumentParser()
    choices = ["get-user-index-mapping-LT20", "get-user-index-mapping-GE20",
               "get-user-brand-mapping-LT20", "get-user-ont-mapping-LT20",
               "get-ont-pdt-mapping", "get-brand-pdt-mapping",
               "get-user-emb-mapping-GE20", "get-brand-emb-mapping-GE20",
               "get-latest-epoch-for-user", "get-latest-epoch-for-item",
               "get-pdt-mapping", "get-user-baseline-feats",
               "get-item-baseline-feats"]
    parser.add_argument('task', choices=choices+["all"],
                        help="task to perform")
    args = parser.parse_args()
    task = args.task

    if task != "all":
        get_artifacts_for_recommendation(task)
    else:
        for task in choices:
            print('Task: %s' % (task))
            get_artifacts_for_recommendation(task)
