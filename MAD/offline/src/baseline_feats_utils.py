"""
Use the last 7 part files (starting with 0005) as the test set
"""
import pandas as pd
import numpy as np
import os, json, time
from constants import *
from metadata_utils import _find_files


# GLOBALS
feat_type_group_cols_dct = {'user': [USER_COL], 'item': [ITEM_COL]}
feat_type_feats_dct = {'user':
    ['num_interactions', 'mean_price_interactions',
     'earliest_interaction_date', 'min_num_interactions_per_pdt',
     'max_num_interactions_per_pdt', 'mean_num_interactions_per_pdt',
     'min_num_interactions_per_ont', 'max_num_interactions_per_ont',
     'mean_num_interactions_per_ont', 'min_num_interactions_per_brand',
     'max_num_interactions_per_brand', 'mean_num_interactions_per_brand'],
                       'item':
    ['num_interactions', 'earliest_interaction_date',
     'min_num_interactions_per_user', 'max_num_interactions_per_user',
     'mean_num_interactions_per_user']}


def _helper_baseline_feat_calc(data, feat_type):

    group_cols = feat_type_group_cols_dct[feat_type]
    feats = feat_type_feats_dct[feat_type]

    if feat_type == 'user':
        print('User features\n')
        out_df = pd.DataFrame({USER_COL: data[USER_COL].unique().tolist()})

        print(feats[0], '\n')
        f = data.groupby(group_cols)[ITEM_COL].count().rename(feats[0]).reset_index()
        out_df = pd.merge(out_df, f, on=group_cols, how='left')

        print(feats[1], '\n')
        f = data.groupby(group_cols)[PRICE_COL].mean().rename(feats[1]).reset_index()
        out_df = pd.merge(out_df, f, on=group_cols, how='left')

        print(feats[2], '\n')
        f = data.groupby(group_cols)[DATE_COL].min().rename(feats[2]).reset_index()
        out_df = pd.merge(out_df, f, on=group_cols, how='left')

        for feat_cols, value_col in [([feats[3], feats[4], feats[5]], ITEM_COL),
                                     ([feats[6], feats[7], feats[8]], ONTOLOGY_COL),
                                     ([feats[9], feats[10], feats[11]], BRAND_COL)]:
            print('user-%s interactions' % (value_col))
            f = data.groupby([USER_COL, value_col])[DATE_COL].count().rename('count').reset_index()
            f = f.groupby(group_cols).agg({'count': ['min', 'max', 'mean']}).reset_index()
            f.columns = group_cols + feat_cols
            out_df = pd.merge(out_df, f, on=group_cols, how='left')
            del f

    elif feat_type == 'item':
        print('Item features\n')
        out_df = pd.DataFrame({ITEM_COL: data[ITEM_COL].unique().tolist()})

        print(feats[0], '\n')
        f = data.groupby(group_cols)[USER_COL].count().rename(feats[0]).reset_index()
        out_df = pd.merge(out_df, f, on=group_cols, how='left')

        print(feats[1], '\n')
        f = data.groupby(group_cols)[DATE_COL].min().rename(feats[1]).reset_index()
        out_df = pd.merge(out_df, f, on=group_cols, how='left')

        print('user-item interactions')
        f = data.groupby([USER_COL, ITEM_COL])[DATE_COL].count().rename('count').reset_index()
        f = f.groupby(group_cols).agg({'count': ['min', 'max', 'mean']}).reset_index()
        f.columns = group_cols + [feats[2], feats[3], feats[4]]
        out_df = pd.merge(out_df, f, on=group_cols, how='left')
        del f

    return out_df


def _combine_feats_across_files_row(global_val, local_val, how):

    if how == 'min':
        func = lambda x, y: min(x, y)
    elif how == 'max':
        func = lambda x, y: max(x, y)
    elif how == 'count':
        func = lambda x, y: sum([x, y])
    elif how == 'mean':
        func = lambda x, y: np.mean([x, y])

    if (pd.notnull(global_val)) and (pd.notnull(local_val)):
        return func(global_val, local_val)
    elif pd.notnull(global_val):
        return global_val
    elif pd.notnull(local_val):
        return local_val
    else:
        return None


def _combine_feats_across_files_df(global_df, local_df, feat_type):

    if global_df.empty:
        return local_df

    how_dct = {
     'num_interactions': 'count', 'mean_price_interactions': 'mean',
     'earliest_interaction_date': 'min', 'min_num_interactions_per_pdt': 'min',
     'max_num_interactions_per_pdt': 'max', 'mean_num_interactions_per_pdt': 'mean',
     'min_num_interactions_per_ont': 'min', 'max_num_interactions_per_ont': 'max',
     'mean_num_interactions_per_ont': 'mean', 'min_num_interactions_per_brand': 'min',
     'max_num_interactions_per_brand': 'max', 'mean_num_interactions_per_brand': 'mean',
     'min_num_interactions_per_user': 'min', 'max_num_interactions_per_user': 'max',
     'mean_num_interactions_per_user': 'mean'}

    feats = feat_type_feats_dct[feat_type]
    group_cols = feat_type_group_cols_dct[feat_type]
    global_feats = ['global_'+x for x in feats]
    local_feats = ['local_'+x for x in feats]
    global_df.rename(columns=dict(zip(feats, global_feats)), inplace=True)
    local_df.rename(columns=dict(zip(feats, local_feats)), inplace=True)
    global_df = pd.merge(global_df, local_df, on=group_cols, how='outer')

    print('combining global and local feats')
    for feat in feats:
        print('Feature: ', feat)
        how = how_dct[feat]
        global_df[feat] = list(map(
            lambda x, y: _combine_feats_across_files_row(x, y, how),
            global_df['global_'+feat], global_df['local_'+feat]))

    global_df.drop(global_feats+local_feats, axis=1, inplace=True)

    return global_df


def get_baseline_feats(raw_data_dir=RAW_DATA_DIR, user_col=USER_COL,
                       item_col=ITEM_COL, ontology_col=ONTOLOGY_COL,
                       brand_col=BRAND_COL, price_col=PRICE_COL,
                       end_token='.gz'):

    print('get all the files to be read')
    files = _find_files(raw_data_dir, end_token)
    print('num files: %d' % (len(files)))

    print('split files into train and test')
    train_files = [x for x in files if not x.startswith('0005')]
    test_files = [x for x in files if x.startswith('0005')]

    print('make sample file name dict')
    file_dct = {'train': train_files, 'test': test_files}
    del files, train_files, test_files

    print('initialize global df')
    user_df = pd.DataFrame(
        columns=[user_col] + feat_type_feats_dct['user'])
    item_df = pd.DataFrame(
        columns=[item_col] + feat_type_feats_dct['item'])
    train_users, train_items = set(), set()

    count = 0
    for key, files in file_dct.items():
        print('Sample: ', key, '\n')

        for file in files:
            if count % 2 == 0:
                print('num files completed: %d' % (count))
                print('\n')

            fn = os.path.join(raw_data_dir, file)
            print('reading file %s' % (fn))
            df = pd.read_csv(fn, compression='gzip', sep='|')
            print('shape: ', df.shape)

            print('missing value imputation')
            df.fillna(value={ontology_col: 'missing', brand_col: 'missing'},
                      inplace=True)

            if key == 'train':

                print('Feat Type: ', 'user', '\n')
                tmp_df = _helper_baseline_feat_calc(df, 'user')
                print('shape: ', tmp_df.shape)
                user_df = _combine_feats_across_files_df(user_df, tmp_df, 'user')
                del tmp_df
                user_lst = df[user_col].unique().tolist()
                train_users = train_users.union(set(user_lst))
                del user_lst
                print('\n\n\n')

                print('Feat Type: ', 'item', '\n')
                tmp_df = _helper_baseline_feat_calc(df, 'item')
                print('shape: ', tmp_df.shape)
                item_df = _combine_feats_across_files_df(item_df, tmp_df, 'item')
                del tmp_df
                item_lst = df[item_col].unique().tolist()
                train_items = train_items.union(set(item_lst))
                del item_lst
                print('\n\n\n')

            elif key == 'test':

                print('find users not present in train sample')
                missing_users = df[user_col].unique().tolist()
                missing_users = list(set(missing_users) - set(train_users))

                if missing_users:
                    print('impute features with median')
                    feats = feat_type_feats_dct['user']
                    medians = [user_df[feat].median() for feat in feats]
                    d = {user_col: missing_users}
                    for i, feat in enumerate(feats):
                        d.update({feat: [medians[i]]*len(missing_users)})
                    tmp_df = pd.DataFrame(d)
                    del d
                    print('shape: ', tmp_df.shape)
                    user_df = _combine_feats_across_files_df(user_df, tmp_df, 'user')
                    del tmp_df
                    train_users = train_users.union(set(missing_users))
                print('\n\n\n')

                print('find items not present in train sample')
                missing_items = df[item_col].unique().tolist()
                missing_items = list(set(missing_items) - set(train_items))

                if missing_items:
                    print('impute features with median')
                    feats = feat_type_feats_dct['item']
                    medians = [item_df[feat].median() for feat in feats]
                    d = {item_col: missing_items}
                    for i, feat in enumerate(feats):
                        d.update({feat: [medians[i]]*len(missing_items)})
                    tmp_df = pd.DataFrame(d)
                    del d
                    print('shape: ', tmp_df.shape)
                    item_df = _combine_feats_across_files_df(item_df, tmp_df, 'item')
                    del tmp_df
                    train_items = train_items.union(set(missing_items))
                print('\n\n\n')
            count += 1

    print('convert global df to global dct')

    print('User DF')
    feats = feat_type_feats_dct['user']
    user_df['feature'] = user_df[feats].apply(tuple, axis=1)
    assert user_df.shape[0] == user_df[user_col].nunique()
    user_dct = dict(zip(user_df[user_col], user_df['feature']))
    del train_users, user_df
    print('\n\n\n')

    print('Item DF')
    feats = feat_type_feats_dct['item']
    item_df['feature'] = item_df[feats].apply(tuple, axis=1)
    assert item_df.shape[0] == item_df[item_col].nunique()
    item_dct = dict(zip(item_df[item_col], item_df['feature']))
    del train_items, item_df
    print('\n\n\n')

    print('save artifacts \n')

    print('user_dct \n')
    json.dump(user_dct, open(USER_BASELINE_FEATS_FN, 'w'))
    del user_dct

    print('item_dct \n')
    json.dump(item_dct, open(ITEM_BASELINE_FEATS_FN, 'w'))


if __name__ == '__main__':
    print('calculating baseline features...')

    start = time.time()

    get_baseline_feats()

    print('total time taken: %0.2f' % (time.time() - start))
