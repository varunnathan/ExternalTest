import re


# util funcs for preprocessing
def _remove_non_ascii_characters(text):
    if (text):
        text = re.sub(r'[^\x00-\x7F]+',' ', text)
    return text


def _remove_special_characters(text):
    if (text):
        text = re.sub(r"[\>\->]", "", text)
    return text
