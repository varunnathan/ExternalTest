import flair
from flair.models import SequenceTagger
import torch
from flair.data import Sentence
import copy
flair.device = torch.device('cpu')
import sys
import logging
from argparse import ArgumentParser
sys.path.append('../')
from constants import *
from preprocess_utils import _remove_non_ascii_characters
from utility import *


def initialize_logger():  # function to initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def parse_command_line_arguments():
    parser = ArgumentParser(description='Command line option parser for spacy inference')

    parser.add_argument('-t', '--text',
                        help='text sequence',
                        type=str,
                        required=True)

    args = parser.parse_args()

    return args


def preprocess(text):
    text_pre = _remove_non_ascii_characters(text)
    text_pre = preprocess_obj.fit_transform(pd.Series([text_pre])).values[0]
    return text_pre


def load_artifacts():
    model = SequenceTagger.load(os.path.join(FLAIR_MODEL_DIR, 'final-model.pt'))
    return model


def combine_contiguous_words(entity_n, label_k='pred_label', start_pos_k='start_pos',
                             end_pos_k='end_pos', text_k='text', score_k='pred_score'):
    out = []
    for i, ent in enumerate(entity_n[1:]):
        if i == 0:
            prev_ent = entity_n[0]
        else:
            if out:
                prev_ent = out[-1]
            else:
                prev_ent = entity_n[i-1]
        if ((prev_ent[label_k] == ent[label_k])
            and (prev_ent[end_pos_k] == ent[start_pos_k] - 1)):
            d = {text_k: prev_ent[text_k] + ' ' + ent[text_k],
                 start_pos_k: prev_ent[start_pos_k], end_pos_k: ent[end_pos_k],
                 label_k: prev_ent[label_k],
                 score_k: min([prev_ent[score_k], ent[score_k]])}
            if i == 0:
                out.append(d)
            else:
                out[-1] = d
        else:
            out.append(ent)
    return out


def inference(logger, model, text):

    logger.info('preprocessing\n')
    text_pre = preprocess(text)

    logger.info('prediction\n')
    sentence = Sentence(text_pre)
    model.predict(sentence)
    out = sentence.to_dict(tag_type='ner')
    entity = out.get('entities', [])

    logger.info('formatting the prediction output\n')
    entity_n = []
    for row in entity:
        row1 = copy.deepcopy(row)
        row1['pred_label'] = row['labels'][0]._value
        row1['pred_score'] = row['labels'][0]._score
        entity_n.append(row1)

    logger.info('combine contiguous parts\n')
    out = combine_contiguous_words(entity_n)

    return entity_n, out


if __name__ == '__main__':
    logger = initialize_logger()

    logger.info('parse args\n')
    args = parse_command_line_arguments()

    logger.info('load model artifacts\n')
    model = load_artifacts()

    logger.info('instantiate preprocess_obj\n')
    preprocess_obj = Text_Preprocessing(keep_eng=False, remove_nonalpha=False, lower_case=False,
                             remove_punkt=False, remove_stop=False, remove_numerals=False,
                             spell_check=False, contraction=True,
                             contraction_var=CONTRACTIONS, stem=False,
                             lem=False, filter_pos=False, pos_var=('N', 'J'),
                             tokenize=False, template_removal=False,
                             template_start_string='', regex_cleaning=False,
                             remove_ignore_words=False, ignore_words=IGNORE_WORDS,
                             custom_stoplist=[], word_size=2, word_size_filter=False)

    logger.info('inference\n')
    _, out = inference(logger, model, str(args.text))

    logger.info('Inference: {}'.format(str(out)))
