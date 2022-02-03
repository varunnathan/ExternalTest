import spacy
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
    nlp = spacy.load(SPACY_MODEL_DIR)
    return nlp


def inference(logger, nlp, text):
    logger.info('preprocessing\n')
    text_pre = preprocess(text)

    logger.info('inference\n')
    out = []
    for token in nlp(text).ents:
        out.append((token.text, token.label_))

    return out


if __name__ == '__main__':
    logger = initialize_logger()

    logger.info('parse args\n')
    args = parse_command_line_arguments()

    logger.info('load model artifacts\n')
    nlp = load_artifacts()

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
    out = inference(logger, nlp, str(args.text))

    logger.info('Inference: {}'.format(str(out)))
