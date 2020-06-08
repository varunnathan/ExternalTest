from metadata_utils import _find_files
from baseline_feats_utils import feat_type_feats_dct
from constants import *
import pandas as pd
import os, time, json


class PrepareData(object):

    def __init__(self, raw_data_dir=RAW_DATA_DIR,
                 interim_data_dir=INTERIM_DATA_DIR, user_col=USER_COL,
                 item_col=ITEM_COL, ontology_col=ONTOLOGY_COL,
                 brand_col=BRAND_COL, price_col=PRICE_COL, dv_col=DV_COL,
                 date_col=DATE_COL, end_token='.gz', user2idx_fn=USER2IDX_FN,
                 item2idx_fn=ITEM2IDX_FN, ont2idx_fn=ONT2IDX_FN,
                 brand2idx_fn=BRAND2IDX_FN, user_feats_fn=USER_BASELINE_FEATS_FN,
                 item_feats_fn=ITEM_BASELINE_FEATS_FN):

        files = _find_files(raw_data_dir, end_token)
        self.files = [os.path.join(raw_data_dir, x) for x in files]
        self.out_files = [os.path.join(interim_data_dir, x) for x in files]
        del files
        self.user_col = user_col
        self.item_col = item_col
        self.ontology_col = ontology_col
        self.brand_col = brand_col
        self.price_col = price_col
        self.date_col = date_col
        self.dv_col = dv_col
        self.dv_map = {'buy': 3, 'addToCart': 2, 'pageView': 1}
        self.user2idx = json.load(open(user2idx_fn))
        self.item2idx = json.load(open(item2idx_fn))
        self.ont2idx = json.load(open(ont2idx_fn))
        self.brand2idx = json.load(open(brand2idx_fn))
        self.user_feats = json.load(open(user_feats_fn))
        self.item_feats = json.load(open(item_feats_fn))
        self.feat_type_feats_dct = feat_type_feats_dct

    def read_file(self, fn):
        df = pd.read_csv(fn, compression='gzip', sep='|')
        return df

    def preprocess(self, data):

        print('missing value imputation')
        data.fillna(value={self.ontology_col: 'missing', self.brand_col: 'missing'},
                    inplace=True)

        print('label encoding for DV')
        data[self.dv_col] = data[self.dv_col].map(self.dv_map)

        print('adding user and item baseline features')

        print('User Features')
        for feat_pos, feat_name in enumerate(self.feat_type_feats_dct['user']):
            data['{}_{}'.format(self.user_col, feat_name)] = data[self.user_col].apply(
                lambda x: self.user_feats[x][feat_pos])
            if feat_name == 'earliest_interaction_date':
                data['{}_days_since_earliest_interaction'.format(self.user_col)] = list(
                    map(lambda x, y: (float(x)-float(y))/(60*60*24),
                        data[self.date_col], data['{}_{}'.format(self.user_col, feat_name)]))
                data.drop('{}_{}'.format(self.user_col, feat_name), axis=1, inplace=True)
                mask = data['{}_days_since_earliest_interaction'.format(self.user_col)] < 0
                data.loc[mask, '{}_days_since_earliest_interaction'.format(self.user_col)] = -1

        print('Item Features')
        for feat_pos, feat_name in enumerate(self.feat_type_feats_dct['item']):
            data['{}_{}'.format(self.item_col, feat_name)] = data[self.item_col].apply(
                lambda x: self.item_feats[x][feat_pos])
            if feat_name == 'earliest_interaction_date':
                data['{}_days_since_earliest_interaction'.format(self.item_col)] = list(
                    map(lambda x, y: (float(x)-float(y))/(60*60*24),
                        data[self.date_col], data['{}_{}'.format(self.item_col, feat_name)]))
                data.drop('{}_{}'.format(self.item_col, feat_name), axis=1, inplace=True)
                mask = data['{}_days_since_earliest_interaction'.format(self.item_col)] < 0
                data.loc[mask, '{}_days_since_earliest_interaction'.format(self.item_col)] = -1

        print('encoding for categorical variables')
        data[self.user_col] = data[self.user_col].map(self.user2idx)
        data[self.item_col] = data[self.item_col].map(self.item2idx)
        data[self.ontology_col] = data[self.ontology_col].map(self.ont2idx)
        data[self.brand_col] = data[self.brand_col].map(self.brand2idx)

        return data

    def prepare_data(self):

        for i, fn in enumerate(self.files):
            if os.path.isfile(self.out_files[i]):
                print('file %s already exists in the disk.' % (self.out_files[i]))
                continue

            if i % 2 == 0:
                print('num completed files: %d' % (i))

            start = time.time()

            print('read file %s' % (fn))
            df = self.read_file(fn)
            print('time taken: %0.2f' % (time.time() - start))
            print('\n\n\n')

            print('preprocess data')
            df = self.preprocess(df)
            print('time taken: %0.2f' % (time.time() - start))
            print('\n\n\n')

            print('save')
            out_fn = self.out_files[i]
            print('out file: %s' % (out_fn))
            df.to_csv(out_fn, index=False, compression='gzip', sep='|')
            del df
            print('time taken: %0.2f' % (time.time() - start))
            print('\n\n\n')


if __name__ == '__main__':
    print('interim data preparation...')

    start = time.time()

    print('Instantiate prepare data class')
    prep_data = PrepareData()
    print('time taken: %0.2f' % (time.time() - start))
    print('\n\n\n')

    print('data preparation begins\n')
    prep_data.prepare_data()
    print('time taken: %0.2f' % (time.time() - start))
    print('\n\n\n')
