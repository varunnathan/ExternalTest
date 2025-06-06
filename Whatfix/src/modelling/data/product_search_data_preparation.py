import gzip
import os
from collections import defaultdict
import numpy as np

from src.modelling.utils.logging_utils import logger
from src.modelling.utils.padding_utils import pad


class ProdSearchData:
    def __init__(self, args, input_train_dir, set_name, global_data):
        self.args = args
        self.neg_per_pos = args.neg_per_pos
        self.set_name = set_name
        self.global_data = global_data
        self.product_size = global_data.product_size
        self.user_size = global_data.user_size
        self.vocab_size = global_data.vocab_size
        self.sub_sampling_rate = None
        self.neg_sample_products = None
        self.word_dists = None
        self.subsampling_rate = args.subsampling_rate
        self.uq_pids = None
        if args.fix_emb:
            self.subsampling_rate = 0
        if set_name == "train":
            self.vocab_distribute = self.read_reviews("{}/{}.txt.gz".format(input_train_dir, set_name))
            self.vocab_distribute = self.vocab_distribute.tolist()
            self.sub_sampling(self.subsampling_rate)
            self.word_dists = self.neg_distributes(self.vocab_distribute)

        if set_name == "train":
            self.product_query_idx = GlobalProdSearchData.read_arr_from_lines(
                     "{}/{}_query_idx.txt.gz".format(input_train_dir, set_name))
            self.review_info = global_data.train_review_info
            self.review_query_idx = global_data.train_query_idxs
        else:
            read_set_name = self.set_name
            if not self.args.has_valid: #if there is no validation set, use test as validation
                read_set_name = 'test'
            self.product_query_idx = GlobalProdSearchData.read_arr_from_lines(
                     "{}/test_query_idx.txt.gz".format(input_train_dir)) #validation and test have same set of queries
            self.review_info, self.review_query_idx = GlobalProdSearchData.read_review_id(
                    "{}/{}_id.txt.gz".format(input_train_dir, read_set_name),
                    global_data.line_review_id_map)

            franklist = '{}/{}.bias_product.ranklist'.format(input_train_dir, read_set_name)
            #franklist = '{}/test.bias_product.ranklist'.format(input_train_dir)
            if args.test_candi_size > 0 and os.path.exists(franklist): #otherwise use all the product ids
                self.uq_pids = self.read_ranklist(franklist, global_data.product_asin2ids)
            #if args.train_review_only:
        #self.u_reviews, self.p_reviews = self.get_u_i_reviews(
        self.u_reviews, self.p_reviews = self.get_u_i_reviews_set(
                self.user_size, self.product_size, global_data.train_review_info)

        if args.prod_freq_neg_sample:
            self.product_distribute = self.collect_product_distribute(global_data.train_review_info)
        else:
            self.product_distribute = np.ones(self.product_size)
        self.product_dists = self.neg_distributes(self.product_distribute)
        #print(self.product_dists)

        self.set_review_size = len(self.review_info)
            #u:reviews i:reviews

    def read_ranklist(self, fname, product_asin2ids):
        uq_pids = defaultdict(list)
        with open(fname, 'r') as fin:
            for line in fin:
                arr = line.strip().split(' ')
                uid, qid = arr[0].split('_')
                asin = arr[2]
                uq_pids[(uid, int(qid))].append(product_asin2ids[asin])
        return uq_pids

    def get_u_i_reviews(self, user_size, product_size, review_info):
        u_reviews = [[] for i in range(self.user_size)]
        p_reviews = [[] for i in range(self.product_size)]
        for _, u_idx, p_idx, r_idx in review_info:
            u_reviews[u_idx].append(r_idx)
            p_reviews[p_idx].append(r_idx)
        return u_reviews, p_reviews

    def get_u_i_reviews_set(self, user_size, product_size, review_info):
        u_reviews = [set() for i in range(self.user_size)]
        p_reviews = [set() for i in range(self.product_size)]
        for _, u_idx, p_idx, r_idx in review_info:
            u_reviews[u_idx].add(r_idx)
            p_reviews[p_idx].add(r_idx)
        return u_reviews, p_reviews

    def initialize_epoch(self):
        #self.neg_sample_products = np.random.randint(0, self.product_size, size = (self.set_review_size, self.neg_per_pos))
        #exlude padding idx
        if self.args.model_name == "item_transformer":
            return
        self.neg_sample_products = np.random.choice(self.product_size,
                size = (self.set_review_size, self.neg_per_pos), replace=True, p=self.product_dists)
        #do subsampling to self.global_data.review_words
        if self.args.do_subsample_mask:
            #self.global_data.set_padded_review_words(self.global_data.review_words)
            return

        rand_numbers = np.random.random(sum(self.global_data.review_length))
        updated_review_words = []
        entry_id = 0
        for review in self.global_data.review_words[:-1]:
            filtered_review = []
            for word_idx in review:
                if rand_numbers[entry_id] > self.sub_sampling_rate[word_idx]:
                    continue
                filtered_review.append(word_idx)
            updated_review_words.append(filtered_review)
        updated_review_words.append([self.global_data.word_pad_idx])
        updated_review_words = pad(updated_review_words,
                pad_id=self.global_data.word_pad_idx, width=self.args.review_word_limit)
        self.global_data.set_padded_review_words(updated_review_words)

    def collect_product_distribute(self, review_info):
        product_distribute = np.zeros(self.product_size)
        for _, uid, pid, _ in review_info:
            product_distribute[pid] += 1
        return product_distribute

    def read_reviews(self, fname):
        vocab_distribute = np.zeros(self.vocab_size)
        #review_info = []
        with gzip.open(fname, 'rt') as fin:
            for line in fin:
                arr = line.strip().split('\t')
                #review_info.append((int(arr[0]), int(arr[1]))) # (user_idx, product_idx)
                review_text = [int(i) for i in arr[2].split(' ')]
                for idx in review_text:
                    vocab_distribute[idx] += 1
        #return vocab_distribute, review_info
        return vocab_distribute

    def sub_sampling(self, subsample_threshold):
        self.sub_sampling_rate = np.asarray([1.0 for _ in range(self.vocab_size)])
        if subsample_threshold == 0.0:
            return
        threshold = sum(self.vocab_distribute) * subsample_threshold
        for i in range(self.vocab_size):
            #vocab_distribute[i] could be zero if the word does not appear in the training set
            if self.vocab_distribute[i] == 0:
                self.sub_sampling_rate[i] = 0
                #if this word does not appear in training set, set the rate to 0.
                continue
            self.sub_sampling_rate[i] = min(1.0, (np.sqrt(float(self.vocab_distribute[i]) / threshold) + 1) * threshold / float(self.vocab_distribute[i]))

        self.sample_count = sum([self.sub_sampling_rate[i] * self.vocab_distribute[i] for i in range(self.vocab_size)])
        self.sub_sampling_rate = np.asarray(self.sub_sampling_rate)
        logger.info("sample_count:{}".format(self.sample_count))

    def neg_distributes(self, weights, distortion = 0.75):
        #print weights
        weights = np.asarray(weights)
        #print weights.sum()
        wf = weights / weights.sum()
        wf = np.power(wf, distortion)
        wf = wf / wf.sum()
        return wf


class GlobalProdSearchData:
    def __init__(self, args, data_path, input_train_dir):

        self.product_ids = self.read_lines("{}/product.txt.gz".format(data_path))
        self.product_asin2ids = {x:i for i,x in enumerate(self.product_ids)}
        self.product_size = len(self.product_ids)
        self.user_ids = self.read_lines("{}/users.txt.gz".format(data_path))
        self.user_size = len(self.user_ids)
        self.words = self.read_lines("{}/vocab.txt.gz".format(data_path))
        self.vocab_size = len(self.words) + 1
        self.query_words = self.read_words_in_lines("{}/query.txt.gz".format(input_train_dir))
        self.word_pad_idx = self.vocab_size-1
        self.query_words = pad(self.query_words, pad_id=self.word_pad_idx)

        #review_word_limit = -1
        #if args.model_name == "review_transformer":
        #    self.review_word_limit = args.review_word_limit
        self.review_words = self.read_words_in_lines(
                "{}/review_text.txt.gz".format(data_path)) #, cutoff=review_word_limit)
        #when using average word embeddings to train, review_word_limit is set
        self.review_length = [len(x) for x in self.review_words]
        self.review_count = len(self.review_words) + 1
        if args.model_name == "review_transformer":
            self.review_words.append([self.word_pad_idx]) # * args.review_word_limit)
            #so that review_words[-1] = -1, ..., -1
            if args.do_subsample_mask:
                self.review_words = pad(self.review_words, pad_id=self.vocab_size-1, width=args.review_word_limit)
        #if args.do_seq_review_train or args.do_seq_review_test:
        self.u_r_seq = self.read_arr_from_lines("{}/u_r_seq.txt.gz".format(data_path)) #list of review ids
        self.i_r_seq = self.read_arr_from_lines("{}/p_r_seq.txt.gz".format(data_path)) #list of review ids
        self.review_loc_time = self.read_arr_from_lines("{}/review_uloc_ploc_and_time.txt.gz".format(data_path)) #(loc_in_u, loc_in_i, time) of each review

        self.line_review_id_map = self.read_review_id_line_map("{}/review_id.txt.gz".format(data_path))
        self.train_review_info, self.train_query_idxs = self.read_review_id(
                "{}/train_id.txt.gz".format(input_train_dir), self.line_review_id_map)
        self.review_u_p = self.read_arr_from_lines("{}/review_u_p.txt.gz".format(data_path)) #list of review ids

        logger.info("Data statistic: vocab %d, review %d, user %d, product %d" % (self.vocab_size,
                    self.review_count, self.user_size, self.product_size))
        self.padded_review_words = None

    def set_padded_review_words(self, review_words):
        self.padded_review_words = review_words
        #words after subsampling and cutoff and padding

    @staticmethod
    def read_review_id(fname, line_review_id_map):
        query_ids = []
        review_info = []
        with gzip.open(fname, 'rt', encoding='utf-8') as fin:
            line_no = 0
            for line in fin:
                arr = line.strip().split('\t')
                review_id = line_review_id_map[int(arr[2].split('_')[-1])]
                review_info.append((line_no, int(arr[0]), int(arr[1]), review_id))#(user_idx, product_idx)
                if arr[-1].isdigit():
                    query_ids.append(int(arr[-1]))
                line_no += 1
                #if there is no query idx afer review_id, query_ids will be illegal and not used
        return review_info, query_ids

    @staticmethod
    def read_review_id_line_map(fname):
        line_review_id_map = dict()
        with gzip.open(fname, 'rt') as fin:
            idx = 0
            for line in fin:
                ori_line_id = int(line.strip().split('_')[-1])
                line_review_id_map[ori_line_id] = idx
                idx += 1
        return line_review_id_map

    @staticmethod
    def read_arr_from_lines(fname):
        line_arr = []
        with gzip.open(fname, 'rt') as fin:
            for line in fin:
                arr = line.strip().split(' ')
                filter_arr = []
                for idx in arr:
                    if len(idx) < 1:
                        continue
                    filter_arr.append(int(idx))
                line_arr.append(filter_arr)
        return line_arr

    @staticmethod
    def read_lines(fname):
        arr = []
        with gzip.open(fname, 'rt') as fin:
            for line in fin:
                arr.append(line.strip())
        return arr

    @staticmethod
    def read_words_in_lines(fname, cutoff=-1):
        line_arr = []
        with gzip.open(fname, 'rt') as fin:
            for line in fin:
                words = [int(i) for i in line.strip().split(' ')]
                if cutoff < 0:
                    line_arr.append(words)
                else:
                    line_arr.append(words[:cutoff])
        return line_arr
