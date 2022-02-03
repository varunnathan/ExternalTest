import pandas as pd
import sys, unicodedata, re, string, nltk, os
import spacy
from tqdm import tqdm
#import psutil
import swifter
#import pyap
#from commonregex import CommonRegex
#reload(sys)
#sys.setdefaultencoding('utf-8')
# Nltk downloads
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('words')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import wordpunct_tokenize
#from textblob import TextBlob


CONTRACTIONS = {
"ain't": "am not / are not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is",
"i'd": "i had / i would",
"i'd've": "i would have",
"i'll": "i shall / i will",
"i'll've": "i shall have / i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"n't": "not",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}
MONTHS = ['january', 'jan', 'february', 'feb', 'march', 'mar', 'april', 'apr',
          'may', 'june', 'jun', 'july', 'jul', 'august', 'aug', 'september',
          'sep', 'sept', 'october', 'oct', 'november', 'nov', 'december',
          'dec']
DAYS = ['saturday', 'sat', 'sunday', 'sun', 'monday', 'mon', 'tuesday', 'tue',
        'tues', 'wednesday', 'wed', 'thursday', 'thur', 'thu', 'friday', 'fri']
LETTERS = list(string.ascii_lowercase)
IGNORE_WORDS = ['yes', 'would', 'bye', 'says', 'tomorrow', 'going', 'yesterday',
                'id','morning','say','three','hmm','ok','seven','guy','oh',
                'got','may','today','mhmm','goodbye','us','thank','ten','wanna',
                'nine','etc','get','four','go','might','mmmm','am','no','uhuh',
                'some','make','could','just','dot','sure','shall','pm','eight',
                'should','uhhuh','yep','na','okay','https','my','huh','like',
                'hello','com','five','must','also','one','eleven','six','two',
                'hi','see','put','evening','yeah','zero','hey','let',
                'afternoon','http']

IGNORE_WORDS = IGNORE_WORDS + MONTHS + DAYS + LETTERS
CONSIDER_WORDS = ['amount', 'call']
SPACY_NLP = spacy.load("en_core_web_lg", disable=["parser", "ner"])
#STOP_WORDS = SPACY_NLP.Defaults.stop_words.union(set(IGNORE_WORDS)) - set(CONSIDER_WORDS)
STOP_WORDS = SPACY_NLP.Defaults.stop_words


class TemplateCleaner:
    def __init__(self, text):
        self._cleaned_text = text

    def _remove_descr(self, stri):

        word_dump = []
        sentences = sent_tokenize(self._cleaned_text)
        for sent in sentences:
            if stri in sent:
                ss = re.split(re.escape(stri), sent)
                word_dump.append(ss[0])
            else:
                word_dump.append(sent)
        self._cleaned_text = ' '.join(word_dump)


class Text_Preprocessing(object):

    def __init__(self, keep_eng=True, remove_nonalpha=False, lower_case=False,
                 remove_punkt=False, remove_stop=False, remove_numerals=False,
                 spell_check=False, contraction=False,
                 contraction_var=None, stem=False,
                 lem=False, filter_pos=False, pos_var=('N', 'J'),
                 tokenize=False, template_removal=False,
                 template_start_string='', regex_cleaning=True,
                 remove_ignore_words=True, ignore_words=IGNORE_WORDS,
                 custom_stoplist=[], word_size=2, word_size_filter=False):
        self.keep_eng = keep_eng
        self.remove_nonalpha = remove_nonalpha
        self.lower_case = lower_case
        self.remove_punkt = remove_punkt
        self.remove_stop = remove_stop
        self.remove_numerals = remove_numerals
        self.spell_check = spell_check
        self.contraction = contraction
        self.contraction_var = contraction_var
        self.stem = stem
        self.lem = lem
        self.filter_pos = filter_pos
        self.pos_var = pos_var
        self.tokenize = tokenize
        self.template_removal = template_removal
        self.template_start_string = template_start_string
        self.regex_cleaning = regex_cleaning
        self.remove_ignore_words = remove_ignore_words
        self.ignore_words = ignore_words
        self.custom_stoplist = custom_stoplist
        self.word_size = word_size
        self.word_size_filter = word_size_filter

    def fit_transform(self, X):
        """
        X: Pandas series
        """
        Y = X.copy()
        if self.template_removal:
            print('template removal')
            Y = Y.apply(lambda x: self.remove_template_from_desc(
             str(x), self.template_start_string))
        if self.contraction:
            print('contraction')
            Y = Y.apply(lambda x: " ".join(
                 self.contraction_var[y.lower()].split('/')[0] if y.lower() in self.contraction_var else y for y in str(x).split()))
        if (self.lem) and (self.remove_stop):
            print('lemmatization and remove stop words')
            stop = STOP_WORDS
            if self.custom_stoplist:
                print('extend nltk list with custom stop words')
                stop = stop | set(self.custom_stoplist)
            Y = Y.swifter.apply(lambda x: " ".join([w.lemma_ for w in SPACY_NLP(x) if w.text not in stop]))
        if (self.lem) and (not self.remove_stop):
            print('lemmatization')
            """
            lem = WordNetLemmatizer()
            Y = Y.apply(lambda x: " ".join(lem.lemmatize(word) for word in x.split()))
            """
            #tqdm.pandas()
            """
            inp_lst = Y.tolist()
            out = []
            for doc in SPACY_NLP.pipe(inp_lst, batch_size=100, n_process=psutil.cpu_count()-1, disable=["parser", "ner"]):
                out.append(" ".join([w.lemma_ for w in doc]))
            Y = pd.Series(out)
            del out, inp_lst
            """
            Y = Y.swifter.apply(lambda x: " ".join([w.lemma_ for w in SPACY_NLP(x)]))
        if self.filter_pos:
            print('filter pos')
            Y = Y.apply(lambda x: " ".join(t[0] for t in nltk.pos_tag(
             x.split()) if t[1].startswith(self.pos_var)))
        if self.lower_case:
            print('lower case')
            Y = Y.apply(lambda x: str(x).strip().lower())
        if self.regex_cleaning:
            print('regex cleaning')
            Y = Y.apply(lambda x: self._regex_cleaning(str(x)))
        if self.remove_numerals:
            print('remove numerals')
            Y = Y.apply(lambda x: re.sub('[0-9]+', '', x))
        if self.remove_punkt:
            print('remove punctuation')
            Y = Y.str.replace('[^\w\s]','')
        if self.remove_ignore_words:
            print('remove ignore words')
            Y = Y.apply(lambda x: self._remove_ignore_words(str(x), self.ignore_words))
        if self.keep_eng:
            print('keep only eng words')
            printable = set(string.printable)
            Y = Y.apply(lambda x: filter(lambda y: y in printable, x))
        if self.remove_nonalpha:
            print('remove non-alphabets')
            Y = Y.apply(lambda x: " ".join(w for w in x.split() if w.isalpha()))
        if (self.remove_stop) and (not self.lem):
            print('remove stop words')
            stop = STOP_WORDS
            if self.custom_stoplist:
                print('extend nltk list with custom stop words')
                stop = stop | set(self.custom_stoplist)
            Y = Y.swifter.apply(lambda x: " ".join(y.text for y in SPACY_NLP.tokenizer(x) if
                                                   (y.text not in stop) and (y.text != " ")))
        if self.spell_check:
            print('spell correction')
            Y = Y.apply(lambda x: str(TextBlob(x).correct()))
        if self.word_size_filter:
            print('word size filter')
            Y = Y.apply(lambda x: " ".join(w for w in x.split()
                        if len(w) >= self.word_size))
        if self.stem:
            print('stemming')
            ps = PorterStemmer()
            Y = Y.apply(lambda x: " ".join(ps.stem(word) for word in x.split()))
        if self.tokenize:
            print('tokenization')
            #Y = Y.apply(nltk.word_tokenize)
            Y = Y.swifter.apply(lambda x: [x.text for x in SPACY_NLP.tokenizer(x) if x.text != " "])

        return Y

    def remove_template_from_desc(self, inp_text, start_string):
        """
        Removes every sentence after start_string from inp_text
        """
        temp_cleaner = TemplateCleaner(inp_text)
        if isinstance(start_string, list):
            for item in start_string:
                temp_cleaner._remove_descr(item)
        else:
            temp_cleaner._remove_descr(start_string)

        return temp_cleaner._cleaned_text
        '''
        self._remove_descr('os :')
        self._remove_descr('os:')
        self._remove_descr('esid: ')
        self._remove_descr('esid : ')
        return self
        '''

    def _regex_cleaning(self, text):
        text = text.lower()
        #text = re.sub(r"\'ll", " will", text)
        #text = re.sub(r"\'d", " would", text)
        #text = re.sub(r"\'ve", " have", text)
        #text = re.sub(r"\'re", " are", text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"www\S+\.com", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"x{2,}", "", text)
        text = re.sub(r"[-()._\#/@&%;*:<>{}+=~!|?,]", " ", text)
        text = re.sub(r"[\[\]]", "", text)
        text = re.sub(r"t1", "", text)
        text = re.sub(r"`", "", text)
        text = text.replace('"', '')
        text = text.replace("'", "")
        return text

    def _remove_ignore_words(self, text, lst):
        new_text = " ".join([x for x in text.split() if x not in lst])
        return new_text
