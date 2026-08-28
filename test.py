import os
import re
import json
import warnings
import pandas as pd
import numpy as np
import emoji
import phonenumbers
import contractions
import nltk
from price_parser.parser import CURRENCY_SYMBOLS
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
from malaya.dictionary import is_malay, is_english
from malaya.stem import sastrawi
from malaya.text.function import get_stopwords as malaya_stopwords

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# SECTION 1: NLTK SETUP & DOWNLOADS
# ==============================================================================
custom_nltk_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.insert(0, custom_nltk_dir)

# for resource in ["punkt", "punkt_tab", "wordnet", "omw-1.4", "stopwords"]:
    # try:
    #     nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    # except LookupError:
    #     # nltk.download(resource, download_dir=custom_nltk_dir, quiet=True)

# ==============================================================================
# SECTION 2: REGEX PATTERNS & CONSTANTS DEFINITIONS
# ==============================================================================
QUOTE_REGEX = re.compile(
    r"QUOTE\s*\([\s\S]*?\)|\[QUOTE[\s\S]*?\[/QUOTE\]", re.IGNORECASE
)
EDIT_SIG_REGEX = re.compile(r"This post has been edited by.*?(?=\n|$)", re.IGNORECASE)

URL_REGEX = re.compile(
    r"(https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9-]+\.[a-zA-Z]{2,24}[^\s]*)", re.IGNORECASE
)

all_symbols = set(CURRENCY_SYMBOLS)
all_symbols.update(["RM", "rm", "MYR", "myr", "S$", "A$", "HK$", "NT$"])
safe_symbols = {
    symbol for symbol in all_symbols if len(symbol) > 1 or not symbol.isalpha()
}
escaped_symbols = [
    re.escape(symbol) for symbol in sorted(safe_symbols, key=len, reverse=True)
]
DYNAMIC_CURRENCY_PATTERN = "|".join(escaped_symbols)
PRICE_REGEX = re.compile(
    rf"(?:(?:\b|\s)(?:{DYNAMIC_CURRENCY_PATTERN})\s*\d+(?:[\.,]\d+)?\b|\b\d+(?:[\.,]\d+)?\s*(?:{DYNAMIC_CURRENCY_PATTERN})\b)",
    re.IGNORECASE,
)

TIME_REGEX = re.compile(
    r"\b(?:1[0-2]|0?[1-9]):[0-5][0-9](?::[0-5][0-9])?\s?(?:[AaPp][Mm])?\b"
    r"|\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?\b"
    r"|\b(?:1[0-2]|0?[1-9])\.[0-5][0-9](?:\.[0-5][0-9])?\s?[AaPp][Mm]\b",
    re.IGNORECASE,
)

DAY_TWODIGIT = r"(?:0[1-9]|[12][0-9]|3[01])"
MONTH = r"(?:0?[1-9]|1[0-2])"
DATE_REGEX = re.compile(
    rf"\b{DAY_TWODIGIT}[-/]{MONTH}(?:[-/](?:\d{{2}}|\d{{4}}))?\b"
    r"|"
    r"\b[0-3]?[0-9]\s?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*(?:\s+\d{2,4})?\b"
    r"|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+[0-3]?[0-9](?:\s+\d{2,4})?\b"
    r"|"
    r"\b\d{4}[-/][0-1]?[0-9][-/][0-3]?[0-9]\b",
    re.IGNORECASE,
)

SMART_APOSTROPHE_REGEX = re.compile(r"['\u2018\u2019\u201a\u201b\u2032\u2035`]")
NON_LATIN_REGEX = re.compile(r"[^\x00-\x7F]+")
REPEAT_CHAR_REGEX = re.compile(r"(.)\1{2,}", re.IGNORECASE)
REPEAT_PUNCT_REGEX = re.compile(r"([!?.])\1+")
EDGE_PUNCT_REGEX = re.compile(r"^([^\w]*)(.*?)([^\w]*)$")

KEEP_PUNCTUATION = {"?", "!"}
ENTITY_TOKENS = {"urltoken", "pricetoken", "phonetoken", "timetoken", "datetoken"}

# ==============================================================================
# SECTION 3: SLANG DICTIONARIES SETUP
# ==============================================================================
# English Slang Setup
with open("slangdict.json", "r", encoding="utf-8") as file:
    english_slangdict = json.load(file)

EN_SLANG_MAP = {
    str(key).lower(): str(value) for key, value in english_slangdict.items() if key
}
en_multi_word_keys = [key for key in EN_SLANG_MAP if " " in key]
en_multi_word_keys.sort(key=len, reverse=True)

EN_MULTI_WORD_REGEX = None
if en_multi_word_keys:
    EN_MULTI_WORD_REGEX = re.compile(
        r"\b(" + "|".join(re.escape(key) for key in en_multi_word_keys) + r")\b",
        flags=re.IGNORECASE,
    )

# Malay Slang Setup
with open("malayslangdict.json", "r", encoding="utf-8") as file:
    malay_slang_1 = json.load(file)

with open("custom_malay_slang.json", "r", encoding="utf-8") as file:
    malay_slang_2 = json.load(file)

malay_slangdict = {**malay_slang_1, **malay_slang_2}

MY_SLANG_MAP = {
    str(key).lower(): str(value)
    for key, value in malay_slangdict.items()
    if key and str(key).lower() != str(value).lower()
}
my_multi_word_keys = [key for key in MY_SLANG_MAP if " " in key]
my_multi_word_keys.sort(key=len, reverse=True)

MY_MULTI_WORD_REGEX = None
if my_multi_word_keys:
    MY_MULTI_WORD_REGEX = re.compile(
        r"\b(" + "|".join(re.escape(key) for key in my_multi_word_keys) + r")\b",
        flags=re.IGNORECASE,
    )

# ==============================================================================
# SECTION 4: STEMMERS, LEMMATIZERS & STOPWORDS SETUP
# ==============================================================================
sastrawi_stemmer = sastrawi()
wordnet_lemmatizer = WordNetLemmatizer()

english_stops = {word.lower() for word in nltk_stopwords.words("english")}
malay_stops = {word.lower() for word in malaya_stopwords()}

PROTECTED_STOPWORDS = {
    "no", "not", "never", "why", "what", "how", "when", "where", "who", "which",
    "tak", "tidak", "bukan", "jangan", "takde", "tiada", "apa", "kenapa",
    "mengapa", "bagaimana", "bila", "mana", "siapa",
}

# ==============================================================================
# SECTION 5: FUNCTION DEFINITIONS
# ==============================================================================
def remove_quotes(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    return QUOTE_REGEX.sub("", text)


def remove_lowyat_edit_signatures(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    return EDIT_SIG_REGEX.sub("", text)


def mask_urls(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    return URL_REGEX.sub(" URLTOKEN ", text)


def mask_phone_numbers(text, default_region="MY"):
    if not isinstance(text, str) or not text.strip():
        return ""
    matches = list(phonenumbers.PhoneNumberMatcher(text, default_region))
    for match in reversed(matches):
        text = text[: match.start] + " PHONETOKEN " + text[match.end :]
    return text


def mask_prices(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    return PRICE_REGEX.sub(" PRICETOKEN ", text)


def mask_times(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    return TIME_REGEX.sub(" TIMETOKEN ", text)


def mask_dates(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    return DATE_REGEX.sub(" DATETOKEN ", text)


def convert_emojis(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    return emoji.demojize(text).replace(":", " ").replace("_", " ")


def to_lower(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    return text.lower()


def normalize_apostrophes(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    return SMART_APOSTROPHE_REGEX.sub("'", text)


def strip_non_latin(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    return NON_LATIN_REGEX.sub("", text)


def reduce_elongated(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = REPEAT_CHAR_REGEX.sub(r"\1\1", text)
    text = REPEAT_PUNCT_REGEX.sub(r"\1", text)
    return text.strip()


def tokenize_sentences(text):
    if not isinstance(text, str) or not text.strip():
        return []
    return sent_tokenize(text)


def normalize_slang_bulletproof(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    if EN_MULTI_WORD_REGEX:
        text = EN_MULTI_WORD_REGEX.sub(
            lambda match: EN_SLANG_MAP.get(match.group(0).lower(), match.group(0)), text
        )
    tokens = text.split()
    normalized_tokens = []
    for token in tokens:
        lower_token = token.lower()
        if lower_token in EN_SLANG_MAP:
            normalized_tokens.append(EN_SLANG_MAP[lower_token])
            continue
        match = EDGE_PUNCT_REGEX.match(token)
        if match:
            print(token + " matches")
            prefix, core, suffix = match.groups()
            lower_core = core.lower()
            if lower_core in EN_SLANG_MAP:
                normalized_tokens.append(f"{prefix}{EN_SLANG_MAP[lower_core]}{suffix}")
                continue
        normalized_tokens.append(token)
    return " ".join(normalized_tokens)


def normalize_malay_slang_bulletproof(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    if MY_MULTI_WORD_REGEX:
        text = MY_MULTI_WORD_REGEX.sub(
            lambda match: MY_SLANG_MAP.get(match.group(0).lower(), match.group(0)), text
        )
    tokens = text.split()
    normalized_tokens = []
    for token in tokens:
        lower_token = token.lower()
        if lower_token in MY_SLANG_MAP:
            normalized_tokens.append(MY_SLANG_MAP[lower_token])
            continue
        match = EDGE_PUNCT_REGEX.match(token)
        if match:
            prefix, core, suffix = match.groups()
            lower_core = core.lower()
            if lower_core in MY_SLANG_MAP:
                normalized_tokens.append(f"{prefix}{MY_SLANG_MAP[lower_core]}{suffix}")
                continue
        normalized_tokens.append(token)
    return " ".join(normalized_tokens)


def normalize_contractions(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    return contractions.fix(text)


def tokenize_words(sentence_text):
    if not isinstance(sentence_text, str) or not sentence_text.strip():
        return []
    return word_tokenize(sentence_text)


def filter_tokens(post_sentences):
    cleaned_post = []
    for sentence in post_sentences:
        cleaned_sentence = []
        for token in sentence:
            token_str = token.strip().lower()
            if token_str in ENTITY_TOKENS:
                cleaned_sentence.append(token_str)
            elif token_str.isalnum():
                cleaned_sentence.append(token_str)
            elif token_str in KEEP_PUNCTUATION:
                cleaned_sentence.append(token_str)
        if cleaned_sentence:
            cleaned_post.append(cleaned_sentence)
    return cleaned_post


def classify_word_lang(token):
    token = token.lower()
    if token in ENTITY_TOKENS:
        return "TAG"
    if token in {"?", "!"}:
        return "PUNCT"
    if is_malay(token):
        return "MALAY"
    if is_english(token):
        return "ENGLISH"
    return "UNKNOWN"


def classify_post_lang(post):
    return [
        [(token, classify_word_lang(token)) for token in sentence] for sentence in post
    ]


def lemmatize_english_token(token: str) -> str:
    """Lemmatizes an English token by checking verb form first, then noun form."""
    token_lower = token.lower()
    lemma = wordnet_lemmatizer.lemmatize(token_lower, pos="v")
    lemma = wordnet_lemmatizer.lemmatize(lemma, pos="n")
    return lemma


def stem_and_lemmatize_token(token, lang_label):
    if lang_label == "MALAY":
        stemmed = sastrawi_stemmer.stem(token)
        return stemmed if stemmed else token
    elif lang_label == "ENGLISH":
        return lemmatize_english_token(token)
    return token


def process_sentence_stem_lemma(sentence_tuples):
    return [
        (stem_and_lemmatize_token(token, lang), lang) for token, lang in sentence_tuples
    ]


def process_post_stem_lemma(post_tuples):
    return [process_sentence_stem_lemma(sentence) for sentence in post_tuples]


def is_stopword_token(token, lang_label):
    if lang_label == "TAG":
        return False
    if lang_label == "PUNCT":
        return False
    return token.lower() in COMBINED_STOPWORDS


def filter_sentence_stopwords(sentence_tuples):
    return [
        (token, lang)
        for token, lang in sentence_tuples
        if not is_stopword_token(token, lang)
    ]


def filter_post_stopwords(post_tuples):
    cleaned_sentences = [
        filter_sentence_stopwords(sentence) for sentence in post_tuples
    ]
    return [sentence for sentence in cleaned_sentences if sentence]


def flatten_post_to_tokens(post_tuples):
    tokens = []
    for sentence in post_tuples:
        for token, _ in sentence:
            tokens.append(str(token))
    return tokens


# ==============================================================================
# SECTION 6: NORMALIZE STOPWORDS (Using defined functions)
# ==============================================================================
normalized_english_stopwords = set()
for stopword in english_stops:
    expanded = normalize_contractions(stopword)
    tokens = word_tokenize(expanded)
    for token in tokens:
        lemma = lemmatize_english_token(token)
        normalized_english_stopwords.add(lemma)

normalized_malay_stopwords = set()
for stopword in malay_stops:
    stemmed = sastrawi_stemmer.stem(stopword)
    normalized_malay_stopwords.add(stemmed if stemmed else stopword)

COMBINED_STOPWORDS = normalized_english_stopwords.union(normalized_malay_stopwords)
COMBINED_STOPWORDS -= PROTECTED_STOPWORDS


# ==============================================================================
# SECTION 7: STEP-BY-STEP PREPROCESSING APPLICATION ON SAMPLE DATA
# ==============================================================================
sample_posts = [
    "saya tak nak beli ni"
]

data = pd.DataFrame({"text": sample_posts})

print("Original Sample Data:")
for i, t in enumerate(data["text"]):
    print(f"[{i}]: {t}")

# 1. Structural cleaning & entity masking
print("Structural cleaning & entity masking:")
data["text"] = data["text"].apply(remove_quotes)
print(data["text"])
data["text"] = data["text"].apply(remove_lowyat_edit_signatures)
print(data["text"])
data["text"] = data["text"].apply(mask_urls)
print(data["text"])
data["text"] = data["text"].apply(mask_phone_numbers)
print(data["text"])
data["text"] = data["text"].apply(mask_prices)
print(data["text"])
data["text"] = data["text"].apply(mask_times)
print(data["text"])
data["text"] = data["text"].apply(mask_dates)
print(data["text"])
data["text"] = data["text"].apply(convert_emojis)
print(data["text"])
data["text"] = data["text"].apply(to_lower)
print(data["text"])
data["text"] = data["text"].apply(normalize_apostrophes)
print(data["text"])
data["text"] = data["text"].apply(strip_non_latin)
print(data["text"])
data["text"] = data["text"].apply(reduce_elongated)
print(data["text"])

# 2. Sentence tokenization & slang/contraction normalization
print("Sentence tokenization & slang/contraction normalization:")
tokenized_sentences = data["text"].apply(tokenize_sentences)
print(tokenized_sentences)
normalized_english_sentences = tokenized_sentences.apply(
    lambda sentence_list: [
        normalize_slang_bulletproof(sentence) for sentence in sentence_list
    ]
)
print(normalized_english_sentences)
normalized_malay_sentences = normalized_english_sentences.apply(
    lambda sentence_list: [
        normalize_malay_slang_bulletproof(sentence) for sentence in sentence_list
    ]
)
print(normalized_malay_sentences)
normalized_contraction_sentences = normalized_malay_sentences.apply(
    lambda sentence_list: [
        normalize_contractions(sentence) for sentence in sentence_list
    ]
)
print(normalized_contraction_sentences)

# 3. Word tokenization & punctuation filtering
print("Word tokenization & punctuation filtering:")
word_tokenized_posts = normalized_contraction_sentences.apply(
    lambda sentence_list: [tokenize_words(sentence) for sentence in sentence_list]
)
print(word_tokenized_posts)
filtered_punct_posts = word_tokenized_posts.apply(filter_tokens)
print(filtered_punct_posts)

# 4. Language identification, Stemming/Lemmatization & Stopword filtering
print("Language identification, Stemming/Lemmatization & Stopword filtering:")
classified_lang_posts = filtered_punct_posts.apply(classify_post_lang)
print(classified_lang_posts)
stemmed_lemmatized_posts = classified_lang_posts.apply(process_post_stem_lemma)
print(stemmed_lemmatized_posts)
filtered_stopwords_posts = stemmed_lemmatized_posts.apply(filter_post_stopwords)
print(filtered_stopwords_posts)

# 5. Flatten to list of tokens
preprocessed_tokens = filtered_stopwords_posts.apply(flatten_post_to_tokens)
print(preprocessed_tokens)