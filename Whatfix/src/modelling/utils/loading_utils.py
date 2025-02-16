import gzip
from src.modelling.utils.logging_utils import logger


def load_pretrained_embeddings(fname):
    embeddings = []
    word_index_dic = dict()
    with gzip.open(fname, 'rt') as fin:
        line_no = 0
        for line in fin:
            arr = line.strip(' ').split('\t')
            word_index_dic[arr[0]] = line_no
            line_no += 1
            vector = arr[1].split()
            vector = [float(x) for x in vector]
            embeddings.append(vector)
    logger.info("Loading {}".format(fname))
    logger.info("Count:{} Embeddings size:{}".format(len(embeddings), len(embeddings[0])))
    return word_index_dic, embeddings


def load_user_item_embeddings(fname):
    embeddings = []
    with open(fname, 'r') as fin:
        for line in fin:
            arr = line.strip().split(' ')
            vector = [float(x) for x in arr]
            embeddings.append(vector)
    logger.info("Loading {}".format(fname))
    logger.info("Count:{} Embeddings size:{}".format(len(embeddings), len(embeddings[0])))
    return embeddings
