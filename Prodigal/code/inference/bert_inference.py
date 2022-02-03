from transformers import pipeline, TFAutoModelForTokenClassification, AutoTokenizer
import copy
import sys
import logging
import json
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
    model1 = TFAutoModelForTokenClassification.from_pretrained(BERT_MODEL_DIR_V2)
    tokenizer1 = AutoTokenizer.from_pretrained(BERT_MODEL_DIR_V2)
    ner_model = pipeline('ner', model=model1, tokenizer=tokenizer1)
    tag_index_inv = json.load(open(BERT_TAG_IDX_INV_FN, 'r'))
    return ner_model, tag_index_inv


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


def inference(logger, ner_model, tag_index_inv, text):

    logger.info('preprocess\n')
    text_pre = preprocess(text)

    logger.info('prediction\n')
    pred = ner_model(text_pre)

    logger.info('formatting the prediction\n')
    pred_n = []
    for row in pred:
        if row['entity'] in ['LABEL_'+str(x) for x in [1, 2, 4, 5]]:
            row1 = copy.deepcopy(row)
            row1['entity'] = tag_index_inv[row1['entity']]
            row1['start_pos'] = row1['end_pos'] = row1['index']
            pred_n.append(row1)

    logger.info('combining contiguous predictions\n')
    out = combine_contiguous_words(pred_n, label_k='entity', start_pos_k='start_pos',
                                   end_pos_k='end_pos', text_k='word', score_k='score')

    return pred_n, out


if __name__ == '__main__':
    logger = initialize_logger()

    logger.info('parse args\n')
    args = parse_command_line_arguments()

    logger.info('load model artifacts\n')
    ner_model, tag_index_inv = load_artifacts()

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
    _, out = inference(logger, ner_model, tag_index_inv, str(args.text))

    logger.info('Inference: {}'.format(str(out)))
