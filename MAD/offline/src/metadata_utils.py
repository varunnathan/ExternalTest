import pandas as pd
import os, json, time, argparse
from constants import *

# GLOBALS


def _find_files(_dir, end_token='.gz'):
    return [x for x in os.listdir(_dir) if x.endswith(end_token)]


def _helper_for_cat_pdt_mapping(cat, pdt_mapping_fn=PDT_MAPPING_FN,
                                item2idx_fn=ITEM2IDX_FN):

    if os.path.exists(pdt_mapping_fn):
        print('pdt mapping dct is present in disk. Use it for %s pdt mapping creation\n' % (cat))

        print('read file %s \n' % (pdt_mapping_fn))
        df = json.load(open(pdt_mapping_fn))

        print('read file %s \n' % (item2idx_fn))
        item2idx = json.load(open(item2idx_fn))

        if cat == ONTOLOGY_COL:
            cat2idx_fn = ONT2IDX_FN
        else:
            cat2idx_fn = BRAND2IDX_FN

        print('read file %s \n' % (cat2idx_fn))
        cat2idx = json.load(open(cat2idx_fn))

        print('cat dct creation\n')
        cat_dct = {}
        idx = 0 if cat == ONTOLOGY_COL else 1
        count = 0
        for k, v in df.items():
            if count % 1000 == 0:
                print('num keys completed: ', count)
            cat_v = v[idx]
            if cat_v in cat_dct:
                cat_dct[cat2idx[cat_v]] = list(set(cat_dct[cat2idx[cat_v]]).union(
                    set([item2idx[k]])))
            else:
                cat_dct[cat2idx[cat_v]] = [item2idx[k]]
            count += 1

        print('save')
        save_fn = ONT_PDT_MAPPING_FN if cat == ONTOLOGY_COL else BRAND_PDT_MAPPING_FN
        json.dump(cat_dct, open(save_fn, 'w'))
    else:
        print('pdt mapping dct is not present in disk. So run the usual flow.')



def _helper_for_user_mapping_df(data, user_col=USER_COL,
                                userevent_col=DV_COL,
                                value_col=ITEM_COL, filter_flag=True):

    group_cols = [user_col, userevent_col]
    needed_cols = group_cols + [value_col]

    if filter_flag:
        mask = data[userevent_col] != 'pageView'
        df = data.loc[mask, needed_cols]
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        df = data[needed_cols]
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
    print('shape: ', df.shape)

    table = pd.DataFrame(df.groupby(group_cols)[value_col].apply(list))
    table.reset_index(inplace=True)

    return table


def _combine_user_mapping_lst(global_lst, local_lst):

    if (global_lst != "") and (local_lst != ""):
        return list(set(global_lst).union(set(local_lst)))
    elif global_lst != "":
        return global_lst
    elif local_lst != "":
        return local_lst
    else:
        return ""


def _combine_user_mapping_df(global_df, local_df, name, value_col,
                             user_col=USER_COL, userevent_col=DV_COL):

    global_df = pd.merge(
        global_df, local_df, on=[user_col, userevent_col], how='outer')
    desired_col = '_'.join([name, 'lst'])
    global_df.fillna(value={desired_col: "", value_col: ""}, inplace=True)
    global_df[desired_col] = list(
        map(lambda x, y: _combine_user_mapping_lst(x, y),
            global_df[desired_col], global_df[value_col]))
    mask = global_df[desired_col] != ""
    global_df = global_df.loc[mask, :]
    global_df.reset_index(drop=True, inplace=True)
    global_df.drop(value_col, axis=1, inplace=True)

    return global_df


def _get_entity_index_map(entity_lst):

    user2idx = {user: i for i, user in enumerate(entity_lst)}
    idx2user = {i: user for i, user in enumerate(entity_lst)}

    return user2idx, idx2user


def _convert_global_df_to_dct(data, value_col, user_col=USER_COL,
                              userevent_col=DV_COL):

    d = {}
    for cat in data[userevent_col].unique():
        mask = data[userevent_col] == cat
        d[cat] = dict(zip(data.loc[mask, user_col], data.loc[mask, value_col]))

    return d


def get_metadata(task, raw_data_dir=RAW_DATA_DIR, user_col=USER_COL,
                 item_col=ITEM_COL, ontology_col=ONTOLOGY_COL,
                 brand_col=BRAND_COL, price_col=PRICE_COL,
                 userevent_col=DV_COL, end_token='.gz'):

    print('get all the files to be read')
    files = _find_files(raw_data_dir, end_token)
    print('num files: %d' % (len(files)))

    print('init global df')

    if task == 'get-num-categories':
        user_ids, item_ids, ontologies, brands = set(), set(), set(), set()

    if task == 'get-pdt-mapping':
        pdt_mapping_df = pd.DataFrame(columns=[item_col, 'pdt_tup'])

    if task == 'get-user-pdt-mapping':
        user_pdt_mapping_df = pd.DataFrame(columns=[user_col, userevent_col, 'pdt_lst'])

    if task == 'get-user-ont-mapping':
        user_ont_mapping_df = pd.DataFrame(columns=[user_col, userevent_col, 'ontology_lst'])

    if task == 'get-user-brand-mapping':
        user_brand_mapping_df = pd.DataFrame(columns=[user_col, userevent_col, 'brand_lst'])

    for i, file in enumerate(files):
        if i % 2 == 0:
            print('num files completed: %d' % (i))
            print('\n')

        fn = os.path.join(raw_data_dir, file)
        print('reading file %s' % (fn))
        df = pd.read_csv(fn, compression='gzip', sep='|')
        print('shape: ', df.shape)

        print('deduplicate after dropping clicked_epoch')
        if 'clicked_epoch' in df:
            df.drop('clicked_epoch', axis=1, inplace=True)
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True, inplace=True)
            print('shape: ', df.shape)

        print('missing value imputation')
        df.fillna(value={ontology_col: 'missing', brand_col: 'missing'},
                  inplace=True)

        if task == 'get-num-categories':
            print('update number of unique categories for categorical variables')
            user_lst = df[user_col].unique().tolist()
            item_lst = df[item_col].unique().tolist()
            ontology_lst = df[ontology_col].unique().tolist()
            brand_lst = df[brand_col].unique().tolist()

            user_ids = user_ids.union(set(user_lst))
            item_ids = item_ids.union(set(item_lst))
            ontologies = ontologies.union(set(ontology_lst))
            brands = brands.union(set(brand_lst))

        if task == 'get-pdt-mapping':
            print('update product mapping df')
            tmp_df = df[[item_col, ontology_col, brand_col, price_col]]
            tmp_df.drop_duplicates(inplace=True)
            tmp_df.reset_index(drop=True, inplace=True)

            assert tmp_df[item_col].nunique() == tmp_df.shape[0]

            tmp_df['pdt_tup'] = list(map(
                lambda x, y, z: tuple([x, y, z]), tmp_df[ontology_col],
                tmp_df[brand_col], tmp_df[price_col]))

            print('concat global and local vars')
            cols = [item_col, 'pdt_tup']
            pdt_mapping_df = pd.concat([pdt_mapping_df[cols], tmp_df[cols]], axis=0)
            pdt_mapping_df.drop_duplicates(inplace=True)
            pdt_mapping_df.reset_index(drop=True, inplace=True)

            assert pdt_mapping_df[item_col].nunique() == pdt_mapping_df.shape[0]
            del tmp_df

        print('update user mapping df')

        if task == 'get-user-pdt-mapping':
            iterator = [('pdt', item_col)]

        elif task == 'get-user-ont-mapping':
            iterator = [('ontology', ontology_col)]

        elif task == 'get-user-brand-mapping':
            iterator = [('brand', brand_col)]

        for name, value_col in iterator:

            print('Entity: %s' % (name))

            if name == 'pdt':
                tmp_df = _helper_for_user_mapping_df(df, user_col, userevent_col,
                                                     value_col, True)
            else:
                tmp_df = _helper_for_user_mapping_df(df, user_col, userevent_col,
                                                     value_col, False)

            print('merge global and local vars')
            if name == 'pdt':
                user_pdt_mapping_df = _combine_user_mapping_df(
                    user_pdt_mapping_df, tmp_df, name, value_col)
            elif name == 'ontology':
                user_ont_mapping_df = _combine_user_mapping_df(
                    user_ont_mapping_df, tmp_df, name, value_col)
            elif name == 'brand':
                user_brand_mapping_df = _combine_user_mapping_df(
                    user_brand_mapping_df, tmp_df, name, value_col)
            del tmp_df
        del df

    print('\n\n\n')

    if task == 'get-num-categories':
        print('num users: %d' % (len(user_ids)))
        print('num items: %d' % (len(item_ids)))
        print('num ontologies: %d' % (len(ontologies)))
        print('num brands: %d' % (len(brands)))
        print('\n\n')

        print('create entity to index mapping')
        user2idx, idx2user = _get_entity_index_map(user_ids)
        item2idx, idx2item = _get_entity_index_map(item_ids)
        ont2idx, idx2ont = _get_entity_index_map(ontologies)
        brand2idx, idx2brand = _get_entity_index_map(brands)
        del user_ids, item_ids, ontologies, brands

        print('save artifacts \n')

        print('user2idx \n')
        json.dump(user2idx, open(USER2IDX_FN, 'w'))

        print('idx2user \n')
        json.dump(idx2user, open(IDX2USER_FN, 'w'))

        print('item2idx \n')
        json.dump(item2idx, open(ITEM2IDX_FN, 'w'))

        print('idx2item \n')
        json.dump(idx2item, open(IDX2ITEM_FN, 'w'))

        print('ont2idx \n')
        json.dump(ont2idx, open(ONT2IDX_FN, 'w'))

        print('idx2ont \n')
        json.dump(idx2ont, open(IDX2ONT_FN, 'w'))

        print('brand2idx \n')
        json.dump(brand2idx, open(BRAND2IDX_FN, 'w'))

        print('idx2brand \n')
        json.dump(idx2brand, open(IDX2BRAND_FN, 'w'))

    if task == 'get-pdt-mapping':
        print('shape of pdt_mapping_df: ', pdt_mapping_df.shape)
        print('\n\n')

        print('convert global df to global dict')
        pdt_mapping_dct = dict(zip(pdt_mapping_df[item_col], pdt_mapping_df['pdt_tup']))
        del pdt_mapping_df

        print('save artifacts \n')

        print('pdt_mapping_dct \n')
        json.dump(pdt_mapping_dct, open(PDT_MAPPING_FN, 'w'))

    if task == 'get-user-pdt-mapping':
        print('shape of user_pdt_mapping_df: ', user_pdt_mapping_df.shape)
        print('\n\n')

        print('convert global df to global dict')
        user_pdt_mapping_dct = _convert_global_df_to_dct(user_pdt_mapping_df,
                                                         value_col='pdt_lst')
        del user_pdt_mapping_df

        print('save artifacts \n')

        print('user_pdt_mapping_dct \n')
        json.dump(user_pdt_mapping_dct, open(USER_PDT_MAPPING_FN, 'w'))

    if task == 'get-user-ont-mapping':
        print('shape of user_ont_mapping_df: ', user_ont_mapping_df.shape)
        print('\n\n')

        print('convert global df to global dict')
        user_ont_mapping_dct = _convert_global_df_to_dct(user_ont_mapping_df,
                                                         value_col='ontology_lst')
        del user_ont_mapping_df

        print('save artifacts \n')

        print('user_ont_mapping_dct \n')
        json.dump(user_ont_mapping_dct, open(USER_ONT_MAPPING_FN, 'w'))

    if task == 'get-user-brand-mapping':
        print('shape of user_brand_mapping_df: ', user_brand_mapping_df.shape)
        print('\n\n')

        print('convert global df to global dict')
        user_brand_mapping_dct = _convert_global_df_to_dct(user_brand_mapping_df,
                                                           value_col='brand_lst')
        del user_brand_mapping_df

        print('save artifacts \n')

        print('user_brand_mapping_dct \n')
        json.dump(user_brand_mapping_dct, open(USER_BRAND_MAPPING_FN, 'w'))


if __name__ == '__main__':
    print('getting metadata...')

    parser = argparse.ArgumentParser()
    choices = ["get-num-categories", "get-pdt-mapping", "get-brand-pdt-mapping",
               "get-ont-pdt-mapping", "get-user-pdt-mapping",
               "get-user-ont-mapping", "get-user-brand-mapping"]
    parser.add_argument('task', choices=choices+["all"],
                        help="task to perform")
    args = parser.parse_args()
    TASK = args.task

    start = time.time()
    if TASK != 'all':
        get_metadata(task=TASK)
    else:
        for task in choices:
            print('Task: %s' % (task))
            get_metadata(task=task)

    print('total time taken: %0.2f' % (time.time() - start))
