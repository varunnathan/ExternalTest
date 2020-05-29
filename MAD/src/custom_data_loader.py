from metadata_utils import _find_files
from baseline_feats_utils import feat_type_feats_dct
from constants import *
import torch
from torch.utils.data import IterableDataset
from itertools import chain


class InteractionsStream(IterableDataset):

    def __init__(self, sample, model_type, file_name=None,
                 interim_data_dir=INTERIM_DATA_DIR, user_col=USER_COL,
                 item_col=ITEM_COL, ontology_col=ONTOLOGY_COL,
                 brand_col=BRAND_COL, price_col=PRICE_COL, dv_col=DV_COL,
                 date_col=DATE_COL, end_token='.gz', chunksize=10):

        data_dir = interim_data_dir

        if file_name is None:
            files = _find_files(data_dir, end_token)
            if sample == 'train':
                self.files = [os.path.join(data_dir, x) for x in files
                              if not x.startswith('0005')]
            elif sample == 'test':
                self.files = [os.path.join(data_dir, x) for x in files
                              if x.startswith('0005')]
        else:
            self.files = [os.path.join(data_dir, file_name)]
        print(self.files)

        self.model_type = model_type
        self.user_col = user_col
        self.item_col = item_col
        self.ontology_col = ontology_col
        self.brand_col = brand_col
        self.price_col = price_col
        self.date_col = date_col
        self.dv_col = dv_col
        self.feat_type_feats_dct = feat_type_feats_dct
        self.chunksize = chunksize
        user_feats = ['{}_{}'.format(self.user_col, x) for x in
                      self.feat_type_feats_dct['user']]
        item_feats = ['{}_{}'.format(self.item_col, x) for x in
                      self.feat_type_feats_dct['item']]
        self.numeric_feats = [self.price_col] + user_feats + item_feats
        self.cat_feats = [self.user_col, self.item_col, self.ontology_col,
                          self.brand_col]


    def read_file(self, fn):

        df = pd.read_csv(fn, compression='gzip', sep='|', iterator=True,
                         chunksize=self.chunksize)
        return df

    def get_dv_for_classification(self, dv_lst):

        if self.model_type == 'classification':
            return [int(x-1) for x in dv_lst]
        else:
            return [int(x) for x in dv_lst]

    def process_data(self, fn):

        print('read data')
        data = self.read_file(fn)

        for row in data:
            x1 = row[self.cat_feats].values.tolist()
            x2 = row[self.numeric_feats].values.tolist()
            y = self.get_dv_for_classification(
                    row[self.dv_col].tolist())
            yield (x1, x2, y)

    def get_stream(self, files):
        return chain.from_iterable(map(self.process_data, files))

    def __iter__(self):
        return self.get_stream(self.files)
