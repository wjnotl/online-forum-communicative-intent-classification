import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import re
import json
import string

import nltk
from nltk import download as nltk_download
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words, wordnet

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOPWORDS
from malaya.dictionary import is_malay

from phonenumbers import PhoneNumberMatcher
from emoji import demojize
from contractions import fix as fix_contractions_func

from joblib import load as joblib_load


def printmd(string_):
    """Print a plain-text message (was markdown rendering in the notebook)."""
    print(string_)

# 0. NLTK downloads (all corpora/models used anywhere in the pipeline below)
nltk_download('punkt', quiet=True)
nltk_download('punkt_tab', quiet=True)
nltk_download('averaged_perceptron_tagger', quiet=True)
nltk_download('averaged_perceptron_tagger_eng', quiet=True)
nltk_download('wordnet', quiet=True)
nltk_download('words', quiet=True)

target_cols = ['Inquiry', 'Complaint', 'Opinion', 'Information', 'Expressive', 'Spam']

# Step 2: Remove Lowyat Quotes, BBCode, Signatures, and HTML

def remove_quote_blocks(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\[quote(?:=[^\]]*)?\][\s\S]*?\[/quote\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?quote(?:=[^\]]*)?\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?quote\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'QUOTE\s*\([^\)]*?\)', ' ', text, flags=re.IGNORECASE)
    return text


def remove_spoilers(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'»\s*Click to show Spoiler.*?«', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[spoiler(?:=[^\]]*)?\][\s\S]*?\[/spoiler\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?spoiler(?:=[^\]]*)?\]', ' ', text, flags=re.IGNORECASE)
    return text


def remove_images_and_code(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'(?:CODE\s*)?\[?/?IMG\][\s\S]*?\[/?IMG\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[code(?:=[^\]]*)?\][\s\S]*?\[/code\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[?/?img\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?code(?:=[^\]]*)?\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bCODE\[', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bCODE(?=[^\s])', ' ', text)
    return text


def remove_media_bbcode(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\[?YOUTUBE(?:=[^\]\n]*)?\]?[\s\S]*?\[/?YOUTUBE\]?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[?VIDEO\s*(?:=[^\]\n]*)?\]?[\s\S]*?\[/?VIDEO\s*\]?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?(?:youtube|video)\s*(?:=[^\]\n]*)?\]?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?url(?:=[^\]\n]*)?\]?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?linkz(?:=[^\]\n]*)?\]?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[attachmentid\s*=\s*\d+\]?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?attachment(?:=[^\]\n]*)?\]?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'Attached\s+(?:thumbnail|image|file|picture)\(s\)', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?email(?:=[^\]\n]*)?\]?', ' ', text, flags=re.IGNORECASE)
    return text


def remove_formatting_tags(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\[/?(?:color|size|font)(?:=[^\]]*)?\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/(?:b|i|u|s)\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[(?:b|i|u)\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[s\](?![a-z])', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\][biuses]\]', ' ', text, flags=re.IGNORECASE)
    return text


def remove_signatures_and_redactions(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'This post has been edited by.*?(?=\n|$)', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<[a-zA-Z0-9\s_\-\/]+removed[a-zA-Z0-9\s_\-]*>', ' ', text, flags=re.IGNORECASE)
    return text


def remove_html_tags(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<(?:iframe|script|style)\b[^>]*>[\s\S]*?</(?:iframe|script|style)>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'</?(?:div|br|p|span|table|tbody|tr|td|th|thead|tfoot|ul|ol|li|img|a|strong|em|hr|blockquote|font|button|input|form)\b[^>]*>', ' ', text, flags=re.IGNORECASE)
    return text

# Step 3: Special Element Masking

def mask_emails(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\b[A-Za-z0-9._%+\-*]+@[A-Za-z0-9.\-*]+\.[A-Za-z]{2,}\b', ' EMAILTOKEN ', text, flags=re.IGNORECASE)
    return text


def mask_urls(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'(?:https?|hxxps?|ftp)://[^\s<>"]+', ' URLTOKEN ', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<![a-zA-Z0-9])www\.[^\s<>"]+', ' URLTOKEN ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:t\.me|goo\.gl|bit\.ly|tinyurl\.com|pictr\.com)(?:/[^\s<>"]*)?', ' URLTOKEN ', text, flags=re.IGNORECASE)

    label = r'[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?'

    my_compound = rf'(?<![@a-zA-Z0-9._-])\b(?:{label}\.)+(?:com|edu|gov|org|net)\.my(?:/[^\s<>"]*)?'
    text = re.sub(my_compound, ' URLTOKEN ', text, flags=re.IGNORECASE)

    gtlds = rf'(?<![@a-zA-Z0-9._-])\b(?:{label}\.)+(?:com|org|net|edu|gov|io|ai|app)(?:/[^\s<>"]*)?'
    text = re.sub(gtlds, ' URLTOKEN ', text, flags=re.IGNORECASE)

    generic_paths = rf'(?<![@a-zA-Z0-9._-])\b(?:{label}\.)+[a-zA-Z]{{2,}}/[^\s<>"]+'
    text = re.sub(generic_paths, ' URLTOKEN ', text, flags=re.IGNORECASE)

    return text


def mask_phone_numbers(text, regions=("MY", "SG", "US", "ID", "TH")):
    if not isinstance(text, str):
        return ""
    if not text.strip():
        return text

    for region in regions:
        try:
            matches = list(PhoneNumberMatcher(text, region))
            for match in reversed(matches):
                text = text[:match.start] + " PHONETOKEN " + text[match.end:]
        except Exception:
            pass

    text = re.sub(r'\b(?:\+?6?01)[0-46-9][-\s]?[0-9]{7,8}\b', ' PHONETOKEN ', text)
    text = re.sub(r'\b(?:\+?6?0[3-9])[-\s]?[0-9]{6,8}\b', ' PHONETOKEN ', text)
    text = re.sub(r'\+\d{1,3}[-\s]?\d{1,4}[-\s]?\d{4,8}\b', ' PHONETOKEN ', text)
    text = re.sub(r'\b(?:1[-.\s]?)?(?:800|888|877|866|855|844|833|1800|1300)[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}\b', ' PHONETOKEN ', text)
    text = re.sub(r'\b(?:1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b', ' PHONETOKEN ', text)
    text = re.sub(r'(?<!\w)(?:\+?60|0)[1-9][0-9X* -]{6,12}[0-9X*]', ' PHONETOKEN ', text, flags=re.IGNORECASE)

    return text


def mask_nric(text):
    if not isinstance(text, str):
        return ""
    if not text.strip():
        return text
    nric_pattern = r'\b[0-9X*]{6}[-\s][0-9X*]{2}[-\s][0-9X*]{4}\b'
    text = re.sub(nric_pattern, ' NRICTOKEN ', text, flags=re.IGNORECASE)
    return text

CURRENCY_PREFIX = r'(?:\b(?:RM|MYR|USD|SGD|AUD|RP)\.?|(?<!\w)(?:S\$|\$))'
CURRENCY_SUFFIX = r'(?:sen|cents?|ringgit|dollars?|myr|rm|usd|sgd|aud|rp)'
NUM_PATTERN = r'\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?'
PRICE_REGEX = re.compile(
    rf'(?:'
    rf'{CURRENCY_PREFIX}\s*(?:{NUM_PATTERN})(?:\s*k\b)?(?:\s*(?:[-–~]|to)\s*(?:{CURRENCY_PREFIX}\s*)?(?:{NUM_PATTERN})(?:\s*k\b)?)?|'
    rf'\b(?:{NUM_PATTERN})(?:\s*k\b)?(?:\s*(?:[-–~]|to)\s*(?:{NUM_PATTERN})(?:\s*k\b)?)?\s*{CURRENCY_SUFFIX}\b'
    rf')',
    re.IGNORECASE
)
MONETARY_K_REGEX = re.compile(
    r'\b(?:price|cost|budget|salary|gaji|deposit|refund|discount)\s*(?:is|of|around|about)?\s*(\d+(?:\.\d+)?)\s*k\b',
    re.IGNORECASE
)


def mask_prices(text):
    if not isinstance(text, str):
        return ""
    if not text.strip():
        return text
    text = PRICE_REGEX.sub(' PRICETOKEN ', text)
    text = MONETARY_K_REGEX.sub(' PRICETOKEN ', text)
    return text

MALAY_TIME_PREFIX = r'(?:pukul|jam|kul|kol|pkl)'
MALAY_TIME_OF_DAY = r'(?:pagi|tengah\s*hari|tengahari|tgh\s*hari|tghari|petang|ptg|malam|mlm)'
ENGLISH_TIME_OF_DAY = (
    r'(?:'
    r'(?:in\s+the\s+|at\s+)?(?:morning|afternoon|evening|night)|'
    r'tonight|noon|midnight'
    r')'
)
OCLOCK_PATTERN = r'(?:o\'?\s*clock)'
ALL_TIME_OF_DAY = rf'(?:{MALAY_TIME_OF_DAY}|{ENGLISH_TIME_OF_DAY}|{OCLOCK_PATTERN})'
TIME_REGEX = re.compile(
    r'\b(?:half|quarter|[1-5]?[0-9])\s+(?:past|to)\s+(?:1[0-2]|0?[1-9])(?:\s*' + ALL_TIME_OF_DAY + r')?\b|'
    r'\b(?:[01]?[0-9]|2[0-3])[:.][0-5][0-9](?:[:.][0-5][0-9])?(?:\s*(?:[AaPp]\.?[Mm]\.?|' + ALL_TIME_OF_DAY + r'))?\b|'
    r'\b(?:[01]?[0-9]|2[0-3])[0-5][0-9]\s*(?:[AaPp]\.?[Mm]\.?)\b|'
    r'\b(?:1[0-2]|0?[1-9])\s*(?:[AaPp]\.?[Mm]\.?|' + OCLOCK_PATTERN + r')\b|'
    r'\b(?:1[0-2]|0?[1-9])\s*(?:' + ALL_TIME_OF_DAY + r')\b|'
    rf'\b{MALAY_TIME_PREFIX}\s*(?:1[0-2]|0?[1-9]|2[0-3])(?::[0-5][0-9]|\.[0-5][0-9])?(?:\s*{ALL_TIME_OF_DAY})?\b|'
    r'\b(?:12\s+)?(?:noon|midnight)\b|'
    r'\b(?:[01][0-9]|2[0-3])[0-5][0-9]\s*(?:hrs?|hours?|h)\b',
    re.IGNORECASE
)


def mask_time(text):
    if not isinstance(text, str):
        return ""
    if not text.strip():
        return text
    return TIME_REGEX.sub(' TIMETOKEN ', text)

MONTH_NAMES = (
    r'(?:jan(?:uary)?|januari|'
    r'feb(?:ruary)?|februari|'
    r'mar(?:ch)?|mac|'
    r'apr(?:il)?|'
    r'may|mei|'
    r'jun(?:e)?|'
    r'jul(?:y)?|julai|'
    r'aug(?:ust)?|ogos|'
    r'sep(?:t(?:ember)?)?|'
    r'oct(?:ober)?|okt(?:ober)?|'
    r'nov(?:ember)?|'
    r'dec(?:ember)?|dis(?:ember)?)'
)
DATE_REGEX = re.compile(
    rf'\b(?:0[1-9]|[12][0-9]|3[01])[-/](?:0?[1-9]|1[0-2])(?:[-/](?:\d{{2}}|\d{{4}}))?\b|'
    rf'\b\d{{4}}[-/][0-1]?[0-9][-/][0-3]?[0-9]\b|'
    rf'\b[0-3]?[0-9](?:st|nd|rd|th)?[-/\s]*{MONTH_NAMES}(?:[-/\s]*(?:19\d{{2}}|20\d{{2}}|\d{{2}})(?!\s*[:.][0-9]|\s*[ap]m))?\b|'
    rf'\b{MONTH_NAMES}[-/\s]*[0-3]?[0-9](?:st|nd|rd|th)?(?:[-/\s]*(?:19\d{{2}}|20\d{{2}}|\d{{2}})(?!\s*[:.][0-9]|\s*[ap]m))?\b|'
    rf'\b{MONTH_NAMES}\s+(?:19\d{{2}}|20\d{{2}})\b|'
    rf'\b(?:year|tahun|since|sejak|dari|in|pada)\s+(?:19\d{{2}}|20\d{{2}})\b',
    re.IGNORECASE
)


def mask_dates(text):
    if not isinstance(text, str):
        return ""
    if not text.strip():
        return text
    return DATE_REGEX.sub(' DATETOKEN ', text)

# Step 4: Remove Elongated Content & Normalize Forum Laughter

def remove_elongated_content(text):
    if not isinstance(text, str) or not text.strip():
        return ""

    text = re.sub(r'\b(?=[a-z]*m)(?=[a-z]*u)(?=[a-z]*a)(?=[a-z]*h)[muah]{4,}\b', 'muahaha', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?=[a-z]*b)(?=[a-z]*a)(?=[a-z]*h)[bah]{4,}\b', 'bahaha', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?=[a-z]*w)(?=[a-z]*a)(?=[a-z]*k)[wak]{4,}\b', 'wakaka', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?=[a-z]*k)(?=[a-z]*a)(?=[a-z]*h)[kah]{4,}\b', 'kahkah', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?=[a-z]*w)(?=[a-z]*k)[wk]{3,}\b', 'wkwk', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?=[a-z]*a)(?=[a-z]*h)[ah]{3,}\b', 'haha', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?=[a-z]*e)(?=[a-z]*h)[eh]{3,}\b', 'hehe', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?=[a-z]*i)(?=[a-z]*h)[ih]{3,}\b', 'hihi', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?=[a-z]*u)(?=[a-z]*h)[uh]{3,}\b', 'huhu', text, flags=re.IGNORECASE)

    text = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', text, flags=re.IGNORECASE)
    text = re.sub(r'([a-zA-Z]{2,})\1{2,}', r'\1\1', text, flags=re.IGNORECASE)
    text = re.sub(r'([^\x00-\x7F])(?:\s*\1)+', r'\1', text)
    text = re.sub(r'([^\w\s])\1+', r'\1', text)

    return text

# Step 5: Convert Emojis to Text

def convert_emojis(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    converted = demojize(text, delimiters=("<emoji>", "</emoji>"))
    return re.sub(r'<emoji>(.*?)</emoji>', lambda m: f" {m.group(1).replace('_', ' ')} ", converted)

# Step 6: Lowercasing

def to_lowercase(text):
    return text.lower() if isinstance(text, str) else ""

# Step 7: Build Priority-Cascading Slang Dictionary & Normalize Slangs

def build_slang_dictionary(file_paths):
    combined_dict = {}
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                d = json.load(f)
                for k, v in d.items():
                    clean_k = str(k).strip().lower()
                    clean_v = str(v).strip().lower()
                    if clean_k and clean_k not in combined_dict:
                        combined_dict[clean_k] = clean_v
        except FileNotFoundError:
            pass
    return combined_dict

SLANG_FILE_PRIORITY = [
    'custom_malay_slang.json',
    'custom_english_slang.json',
    'malay_slangdict.json',
    'english_slangdict.json'
]
MASTER_SLANG_DICT = build_slang_dictionary(SLANG_FILE_PRIORITY)

MULTI_WORD_SLANG = {k: v for k, v in MASTER_SLANG_DICT.items() if ' ' in k or '-' in k}
SINGLE_WORD_SLANG = {k: v for k, v in MASTER_SLANG_DICT.items() if ' ' not in k and '-' not in k}

if MULTI_WORD_SLANG:
    phrase_keys = [re.escape(k) for k in sorted(MULTI_WORD_SLANG.keys(), key=len, reverse=True)]
    PHRASE_REGEX = re.compile(r'\b(' + '|'.join(phrase_keys) + r')\b', re.IGNORECASE)
else:
    PHRASE_REGEX = None


def normalize_slangs(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    if PHRASE_REGEX:
        text = PHRASE_REGEX.sub(lambda m: MULTI_WORD_SLANG[m.group(0).lower()], text)
    return re.sub(r'\b[a-zA-Z0-9]+\b', lambda m: SINGLE_WORD_SLANG.get(m.group(0).lower(), m.group(0)), text)

# Step 8: Fix Contractions

def fix_contractions(text):
    return fix_contractions_func(text) if isinstance(text, str) else ""

# Step 9: Remove All Punctuations

def remove_punctuations(text):
    if not isinstance(text, str):
        return ""
    punct_pattern = f"[{re.escape(string.punctuation)}]"
    return re.sub(punct_pattern, ' ', text)

# Step 10: Split Alphanumeric Tokens & Remove Standalone Numbers

def split_and_remove_numbers(text):
    if not isinstance(text, str):
        return ""
    if not text.strip():
        return text
    text = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)
    text = re.sub(r'\b\d+\b', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# Step 11: Remove Non-Latin Words / Characters

def remove_non_latin(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'[^\x00-\x7F]+', ' ', text)

# Step 12: Word Tokenization

def tokenize_words(text):
    if not isinstance(text, str) or not text.strip():
        return []
    return word_tokenize(text)

# Step 13: Language Classification and Token Tagging

NLTK_WORDS_SET = {w.lower() for w in words.words()}
MASKING_TAGS = {'emailtoken', 'urltoken', 'phonetoken', 'nrictoken', 'pricetoken', 'timetoken', 'datetoken'}

stemmer_factory = StemmerFactory()
malay_stemmer = stemmer_factory.create_stemmer()


def is_english_nltk(w):
    return w in NLTK_WORDS_SET or len(wordnet.synsets(w)) > 0


def tag_tokens(tokens):
    tagged = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower in MASKING_TAGS:
            tagged.append((token, "TAG"))
        elif is_english_nltk(token_lower):
            tagged.append((token, "ENGLISH"))
        elif is_malay(token_lower):
            tagged.append((token, "MALAY"))
        else:
            tagged.append((token, "UNKNOWN"))
    return tagged

# Step 14: Part-of-Speech (POS) Tagging for English Tokens

def get_simplified_pos(ptb_tag):
    if ptb_tag.startswith('J'):
        return 'ADJ'
    elif ptb_tag.startswith('V'):
        return 'VERB'
    elif ptb_tag.startswith('N'):
        return 'NOUN'
    elif ptb_tag.startswith('R'):
        return 'ADV'
    return 'NOUN'


def pos_tag_english_tokens(tagged_tokens):
    words_list = [t[0] for t in tagged_tokens]
    if not words_list:
        return []

    raw_pos = pos_tag(words_list)

    pos_tagged_list = []
    for (word, lang_tag), (_, ptb_tag) in zip(tagged_tokens, raw_pos):
        if lang_tag == "ENGLISH":
            pos_label = get_simplified_pos(ptb_tag)
            pos_tagged_list.append((word, lang_tag, pos_label))
        else:
            pos_tagged_list.append((word, lang_tag))

    return pos_tagged_list

# Step 15: Morphological Normalization (Lemmatization with POS & Stemming)

lemmatizer = WordNetLemmatizer()

POS_TO_WORDNET = {
    'ADJ': wordnet.ADJ,
    'VERB': wordnet.VERB,
    'NOUN': wordnet.NOUN,
    'ADV': wordnet.ADV
}


def lemmatize_and_stem(pos_tagged_tokens):
    processed = []
    for item in pos_tagged_tokens:
        word = item[0]
        lang_tag = item[1]

        if lang_tag == "ENGLISH":
            pos_label = item[2] if len(item) > 2 else 'NOUN'
            wn_pos = POS_TO_WORDNET.get(pos_label, wordnet.NOUN)
            processed.append((lemmatizer.lemmatize(word, pos=wn_pos), lang_tag))
        elif lang_tag == "MALAY":
            processed.append((malay_stemmer.stem(word), lang_tag))
        else:
            processed.append((word, lang_tag))

    return processed

# Step 16: Stop Word Removal

PRESERVED_WORDS = {'why', 'how', 'what', 'which', 'where', 'who', 'kenapa', 'mengapa', 'bagaimana'}
ENGLISH_STOPWORDS = set(SPACY_STOPWORDS) - PRESERVED_WORDS
sastrawi_stop_factory = StopWordRemoverFactory()
MALAY_STOPWORDS = set(sastrawi_stop_factory.get_stop_words()) - PRESERVED_WORDS


def remove_stopwords(morph_tokens):
    cleaned = []
    for word, tag in morph_tokens:
        word_lower = word.lower()

        if word_lower in ENGLISH_STOPWORDS or word_lower in MALAY_STOPWORDS:
            continue

        if word.isdigit():
            continue

        if len(word) > 1 or word_lower in MASKING_TAGS:
            cleaned.append((word, tag))

    return cleaned

# Step 27: Preprocessing Pipeline Encapsulation & Model Artifact Loading

def preprocess_pipeline(raw_text):
    # Step 2: Remove Lowyat Quotes, BBCode, Signatures, and HTML
    text = remove_quote_blocks(raw_text)
    text = remove_spoilers(text)
    text = remove_images_and_code(text)
    text = remove_media_bbcode(text)
    text = remove_formatting_tags(text)
    text = remove_signatures_and_redactions(text)
    text = remove_html_tags(text)

    # Step 3: Special Element Masking (Time first, then Date)
    text = mask_emails(text)
    text = mask_urls(text)
    text = mask_phone_numbers(text)
    text = mask_nric(text)
    text = mask_prices(text)
    text = mask_time(text)
    text = mask_dates(text)

    # Step 4: Remove Elongated Characters & Repeating Emojis
    text = remove_elongated_content(text)

    # Step 5: Convert Emojis to Text
    text = convert_emojis(text)

    # Step 6: Lowercasing
    text = to_lowercase(text)

    # Step 7: Slang Normalization (Hybrid)
    text = normalize_slangs(text)

    # Step 8: Fix Contractions
    text = fix_contractions(text)

    # Step 9: Remove All Punctuations
    text = remove_punctuations(text)

    # Step 10: Split Alphanumeric Tokens & Remove Numbers
    text = split_and_remove_numbers(text)

    # Step 11: Remove Non-Latin Characters
    text = remove_non_latin(text)

    # Step 12: Word Tokenization
    tokens = tokenize_words(text)

    # Step 13: Language Classification and Token Tagging
    tagged = tag_tokens(tokens)

    # Step 14: Part-of-Speech (POS) Tagging for English Tokens (on full sequences)
    pos_tagged = pos_tag_english_tokens(tagged)

    # Step 15: Morphological Normalization (Lemmatization with POS & Stemming)
    morph_tokens = lemmatize_and_stem(pos_tagged)

    # Step 16: Stop Word Removal (Applied on Lemmatized/Stemmed Tokens)
    filtered = remove_stopwords(morph_tokens)

    return ' '.join([t[0] for t in filtered])

loaded_tfidf = joblib_load("tfidf_vectorizer.joblib")
loaded_lr = joblib_load("logistic_regression_model.joblib")
loaded_svc = joblib_load("linear_svc_model.joblib")

# Step 28: Real-Time Prediction on Unseen Posts (Showing Both Models)

def format_ranked_badges(ranked_list, is_probability=True):
    if not ranked_list:
        return '<span style="color:#777; font-style:italic; font-size:12px;">Neutral / None</span>'

    badges = []
    for idx, (intent, score) in enumerate(ranked_list):
        score_text = f"{score * 100:.1f}%" if is_probability else f"score: {score:+.2f}"

        if idx == 0:
            badges.append(
                f'<span style="background-color:#2e7d32; color:#ffffff; padding: 3px 9px; '
                f'border-radius: 4px; font-weight: bold; font-size: 12px; margin-right: 4px;">'
                f'★ Best: {intent} ({score_text})</span>'
            )
        else:
            badges.append(
                f'<span style="background-color:#e0e0e0; color:#333333; padding: 3px 8px; '
                f'border-radius: 4px; font-weight: 500; font-size: 12px; margin-right: 4px;">'
                f'{intent} ({score_text})</span>'
            )
    return ''.join(badges)


def predict_unseen(raw_text):
    cleaned_input = preprocess_pipeline(raw_text)

    if not cleaned_input.strip():
        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned_input,
            "lr_ranked": [],
            "svc_ranked": []
        }

    vectorized_input = loaded_tfidf.transform([cleaned_input])

    lr_probs = loaded_lr.predict_proba(vectorized_input)[0]
    lr_preds = loaded_lr.predict(vectorized_input)[0]
    lr_positive = [(target_cols[i], lr_probs[i]) for i, val in enumerate(lr_preds) if val == 1]
    lr_ranked = sorted(lr_positive, key=lambda x: x[1], reverse=True)

    svc_scores = loaded_svc.decision_function(vectorized_input)[0]
    svc_preds = loaded_svc.predict(vectorized_input)[0]
    svc_positive = [(target_cols[i], svc_scores[i]) for i, val in enumerate(svc_preds) if val == 1]
    svc_ranked = sorted(svc_positive, key=lambda x: x[1], reverse=True)

    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned_input,
        "lr_ranked": lr_ranked,
        "svc_ranked": svc_ranked
    }


def format_ranked_plain(ranked_list, is_probability=True):
    """Plain-text version of format_ranked_badges for terminal output."""
    if not ranked_list:
        return "Neutral / None"

    parts = []
    for idx, (intent, score) in enumerate(ranked_list):
        score_text = f"{score * 100:.1f}%" if is_probability else f"score: {score:+.2f}"
        prefix = "* BEST: " if idx == 0 else "  "
        parts.append(f"{prefix}{intent} ({score_text})")
    return " | ".join(parts)


def interactive_intent_predictor():
    printmd("=== Forum Comment Intent Predictor ===")
    printmd("Type any forum comment below to test intent classification across both models (or enter 'q' to quit):")

    while True:
        user_post = input("\nYour comment (or 'q' to quit): ").strip()
        if user_post.lower() == 'q':
            printmd("Exiting session.")
            break
        if not user_post:
            print("Please enter a non-empty comment.")
            continue

        res = predict_unseen(user_post)
        print("-" * 60)
        print(f"Raw Input:     {res['raw_text']}")
        print(f"Cleaned Input: {res['cleaned_text']}")
        print(f"Logistic Regression: {format_ranked_plain(res['lr_ranked'], is_probability=True)}")
        print(f"Linear SVC:          {format_ranked_plain(res['svc_ranked'], is_probability=False)}")
        print("(Predicted intents are ordered by model confidence score; '* BEST' marks the primary intent.)")
        print("-" * 60)


if __name__ == "__main__":
    interactive_intent_predictor()