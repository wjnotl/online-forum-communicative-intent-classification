# Step 1: Load Dataset & Exploratory Data Analysis (EDA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("dataset.csv")

# 2. Display basic counts and top 5 rows
print(f"Dataset summary: {len(df):,} total rows, {df.shape[1]} columns")
print(f"\nPreview of first 5 records:")
print(df.head())

# Target intent columns
target_cols = ['Inquiry', 'Complaint', 'Opinion', 'Information', 'Expressive', 'Spam']

# 3. Calculate and display intent distribution
intent_counts = df[target_cols].sum().sort_values(ascending=False)
print("\n1. Target Intent Distribution:")
print(pd.DataFrame({"Total Posts": intent_counts, "Percentage (%)": (intent_counts / len(df) * 100).round(2)}))

# 4. Multi-Label Intent Cardinality (How many intents per post)
df['num_intents'] = df[target_cols].sum(axis=1)
intent_num_dist = df['num_intents'].value_counts().sort_index()
print("\n2. Multi-Label Intent Cardinality (Number of Intents per Post):")
print(pd.DataFrame({
    "Intents per Post": intent_num_dist.index.map(lambda n: f"{n} Intent{'s' if n != 1 else ''}"),
    "Total Posts": intent_num_dist.values,
    "Percentage (%)": (intent_num_dist.values / len(df) * 100).round(2)
}))

# 5. Top Co-occurring Intent Combinations
df['intent_combination'] = df[target_cols].apply(
    lambda row: ' + '.join([col for col in target_cols if row[col] == 1]) if row.sum() > 0 else 'None (0 Intents)',
    axis=1
)
top_combinations = df['intent_combination'].value_counts().head(8)
print("\n3. Top 8 Most Common Intent Combinations:")
print(pd.DataFrame({
    "Intent Combination": top_combinations.index,
    "Total Posts": top_combinations.values,
    "Percentage (%)": (top_combinations.values / len(df) * 100).round(2)
}))

# 6. Calculate word count per post
df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))
df['char_length'] = df['text'].astype(str).apply(len)

avg_word_counts = {}
for col in target_cols:
    avg_word_counts[col] = df[df[col] == 1]['word_count'].mean()
avg_words_series = pd.Series(avg_word_counts).sort_values(ascending=False)

print(f"\n4. Average post length (word count) per intent category:")
print(avg_words_series.round(1))

# 7. Plot 3-Panel Side-by-Side EDA Graphs
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Subplot 1: Intent Distribution
sns.barplot(x=intent_counts.index, y=intent_counts.values, ax=axes[0], palette="viridis")
axes[0].set_title("1. Individual Intent Distribution", fontsize=11, fontweight='bold')
axes[0].set_ylabel("Total Post Count")
axes[0].tick_params(axis='x', rotation=30)
for i, v in enumerate(intent_counts.values):
    axes[0].text(i, v + 500, f"{v:,}", ha='center', fontsize=8)

# Subplot 2: Multi-Label Intent Cardinality (Number of Intents per Post)
sns.barplot(x=[f"{n} Intent{'s' if n != 1 else ''}" for n in intent_num_dist.index], 
            y=intent_num_dist.values, ax=axes[1], palette="magma")
axes[1].set_title("2. Intents Assigned per Post", fontsize=11, fontweight='bold')
axes[1].set_ylabel("Total Post Count")
axes[1].tick_params(axis='x', rotation=15)
for i, v in enumerate(intent_num_dist.values):
    axes[1].text(i, v + 500, f"{v:,}", ha='center', fontsize=8)

# Subplot 3: Average Post Length (Word Count) vs Intent Category
sns.barplot(x=avg_words_series.index, y=avg_words_series.values, ax=axes[2], palette="mako")
axes[2].set_title("3. Average Word Count vs Intent", fontsize=11, fontweight='bold')
axes[2].set_ylabel("Average Words per Post")
axes[2].tick_params(axis='x', rotation=30)
for i, v in enumerate(avg_words_series.values):
    axes[2].text(i, v + 0.5, f"{v:.1f}", ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.show()



import re
import emoji

# Step 2: Remove Lowyat quotes, BBCode, signatures, and strict HTML
def remove_lowyat_signatures(text):
    if not isinstance(text, str):
        return ""
    # BBCode Quote blocks & quote headers
    text = re.sub(r'\[quote(?:=[^\]]*)?\][\s\S]*?\[/quote\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?quote(?:=[^\]]*)?\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'QUOTE\s*\([^\)]*?\)', ' ', text, flags=re.IGNORECASE)
    
    # Spoilers
    text = re.sub(r'»\s*Click to show Spoiler.*?«', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[spoiler(?:=[^\]]*)?\][\s\S]*?\[/spoiler\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?spoiler(?:=[^\]]*)?\]', ' ', text, flags=re.IGNORECASE)
    
    # Images & Code blocks
    text = re.sub(r'(?:CODE\s*)?\[IMG\][\s\S]*?\[/IMG\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?img\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[code(?:=[^\]]*)?\][\s\S]*?\[/code\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?code(?:=[^\]]*)?\]', ' ', text, flags=re.IGNORECASE)
    
    # URLs, YouTube, Videos, Emails & Attachments
    text = re.sub(r'\[url(?:=[^\]]*)?\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/url\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?linkz(?:=[^\]]*)?\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[youtube(?:=[^\]]*)?\][\s\S]*?\[/youtube\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?youtube\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[video(?:=[^\]]*)?\][\s\S]*?\[/video\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?video\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[attachmentid=\d+\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?attachment(?:=[^\]]*)?\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'Attached (?:thumbnail|image|file|picture)\(s\)', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[email(?:=[^\]]*)?\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/email\]', ' ', text, flags=re.IGNORECASE)
    
    # Formatting: Bold, Italic, Underline, Strike, Color, Size, Font
    text = re.sub(r'\[/?b\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?i\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?u\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?s\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?color(?:=[^\]]*)?\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?size(?:=[^\]]*)?\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?font(?:=[^\]]*)?\]', ' ', text, flags=re.IGNORECASE)
    
    # Edit signatures & Moderator redactions
    text = re.sub(r'This post has been edited by.*?(?=\n|$)', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<[a-zA-Z0-9\s_\-\/]+removed[a-zA-Z0-9\s_\-]*>', ' ', text, flags=re.IGNORECASE)
    
    # Strict HTML tags
    strict_html = r'</?(?:a|abbr|b|br|button|div|em|font|h[1-6]|hr|i|iframe|img|li|ol|p|pre|s|small|span|strong|sub|sup|table|tbody|td|th|thead|tr|u|ul)(?:\s+[^>]*?)?>'
    text = re.sub(strict_html, ' ', text, flags=re.IGNORECASE)
    return text

import phonenumbers

# Currency patterns (explicit prefixes and suffixes)
CURRENCY_PREFIX = r'(?:RM|rm|MYR|myr|USD|usd|SGD|sgd|AUD|aud|RP|Rp|S\$|\$)'
CURRENCY_SUFFIX = r'(?:sen|cents?|ringgit|dollars?|myr|rm|usd|sgd|aud|rp)'

PRICE_REGEX = re.compile(
    rf'(?:'
    rf'\b{CURRENCY_PREFIX}\s*\d+(?:[\.,]\d+)?(?:\s*k\b)?|'
    rf'\b\d+(?:[\.,]\d+)?\s*{CURRENCY_SUFFIX}\b'
    rf')',
    re.IGNORECASE
)
MONETARY_K_REGEX = re.compile(
    r'\b(?:price|cost|budget|salary|gaji|deposit|refund|discount)\s*(?:is|of|around|about)?\s*(\d+(?:\.\d+)?)\s*k\b',
    re.IGNORECASE
)

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
    rf'\b[0-3]?[0-9]\s+{MONTH_NAMES}(?:\s+\d{{2,4}})?\b|'
    rf'\b{MONTH_NAMES}\s+[0-3]?[0-9](?:\s+\d{{2,4}})?\b|'
    r'\b\d{4}[-/][0-1]?[0-9][-/][0-3]?[0-9]\b',
    re.IGNORECASE
)

# Step 3: Special element masking
def mask_elements(text):
    if not isinstance(text, str):
        return ""
    if not text.strip():
        return text
    # URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' URLTOKEN ', text, flags=re.IGNORECASE)
    # Phone numbers via phonenumbers + strict Malaysian/international prefixes
    try:
        matches = list(phonenumbers.PhoneNumberMatcher(text, "MY"))
        for match in reversed(matches):
            text = text[:match.start] + " PHONETOKEN " + text[match.end:]
    except Exception:
        pass
    text = re.sub(r'\b(?:\+?6?01)[0-46-9][-\s]?[0-9]{7,8}\b', ' PHONETOKEN ', text)
    text = re.sub(r'\b(?:\+?6?0[3-9])[-\s]?[0-9]{6,8}\b', ' PHONETOKEN ', text)
    text = re.sub(r'\+\d{1,3}[-\s]?\d{1,4}[-\s]?\d{4,8}\b', ' PHONETOKEN ', text)
    
    # Prices & Currencies
    text = PRICE_REGEX.sub(' PRICETOKEN ', text)
    text = MONETARY_K_REGEX.sub(' PRICETOKEN ', text)
    
    # Dates
    text = DATE_REGEX.sub(' DATETOKEN ', text)
    # Time
    text = re.sub(r'\b(?:1[0-2]|0?[1-9]):[0-5][0-9](?::[0-5][0-9])?\s?(?:[AaPp][Mm])?\b|\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?\b|\b(?:1[0-2]|0?[1-9])\.[0-5][0-9](?:\.[0-5][0-9])?\s?[AaPp][Mm]\b', ' TIMETOKEN ', text, flags=re.IGNORECASE)
    return text

# Step 4: Convert Emojis to text (natural separated words)
def convert_emojis(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    demojized = emoji.demojize(text, delimiters=(" ", " "))
    return demojized.replace("_", " ")

# Step 5: Lowercasing
def to_lowercase(text):
    return text.lower() if isinstance(text, str) else ""

# Step 6: Remove elongated characters, repeated words & repeated phrases
def remove_elongated_content(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    # 1. Reduce elongated characters (>2 repetitions to 2)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # 2. Reduce repeated consecutive words (e.g. "haha haha haha haha" -> "haha haha")
    text = re.sub(r'\b(\w+)(?:\s+\1\b){2,}', r'\1 \1', text, flags=re.IGNORECASE)
    # 3. Reduce repeated multi-word phrases (e.g. "rolling on the floor laughing" x 6 -> x 2)
    text = re.sub(r'(\b(?:\w+\s+){1,5}\w+)(?:\s+\1){2,}', r'\1 \1', text, flags=re.IGNORECASE)
    return text


import json
import string
import contractions

# Step 7: Build priority-cascading slang dictionary (Hybrid Solution)
# Sources:
# 1. custom_malay_slang.json: Custom-curated Lowyat Kopitiam / Malaysian SMS shortforms
# 2. custom_english_slang.json: Custom-curated internet & forum abbreviations
# 3. malay_slangdict.json: Mendeley Data Malay Slang Dataset (https://data.mendeley.com/datasets/mgv2n2vcb9/3/files/a7b86a2f-1175-4ff0-b813-d95218534cd4)
# 4. english_slangdict.json: Ekphrasis / NoSlang Internet Slang Dictionary (https://github.com/cbaziotis/ekphrasis/blob/master/ekphrasis/dicts/noslang/slangdict.py)

def build_slang_dictionary(file_paths):
    combined_dict = {}
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                d = json.load(f)
                for k, v in d.items():
                    clean_k = str(k).strip().lower()
                    clean_v = str(v).strip().lower()
                    # Only add if key doesn't exist (preserves priority order)
                    if clean_k and clean_k not in combined_dict:
                        combined_dict[clean_k] = clean_v
        except FileNotFoundError:
            pass
    return combined_dict

slang_file_priority = [
    'custom_malay_slang.json',
    'custom_english_slang.json',
    'malay_slangdict.json',
    'english_slangdict.json'
]
MASTER_SLANG_DICT = build_slang_dictionary(slang_file_priority)

# Separate into multi-word phrases and single-word slangs for hybrid processing
MULTI_WORD_SLANG = {k: v for k, v in MASTER_SLANG_DICT.items() if ' ' in k or '-' in k}
SINGLE_WORD_SLANG = {k: v for k, v in MASTER_SLANG_DICT.items() if ' ' not in k and '-' not in k}
print(f"Compiled master slang dictionary: {len(MASTER_SLANG_DICT):,} entries (Single words: {len(SINGLE_WORD_SLANG):,}, Phrases: {len(MULTI_WORD_SLANG):,})")

if MULTI_WORD_SLANG:
    phrase_keys = [re.escape(k) for k in sorted(MULTI_WORD_SLANG.keys(), key=len, reverse=True)]
    PHRASE_REGEX = re.compile(r'\b(' + '|'.join(phrase_keys) + r')\b', re.IGNORECASE)
else:
    PHRASE_REGEX = None

def normalize_slangs(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    # 1. Multi-word phrase replacement
    if PHRASE_REGEX:
        text = PHRASE_REGEX.sub(lambda m: MULTI_WORD_SLANG[m.group(0).lower()], text)
    # 2. Fast O(1) dictionary word lookup
    words = text.split()
    return ' '.join([SINGLE_WORD_SLANG.get(w.lower(), w) for w in words])

# Step 8: Fix contractions
def fix_contractions(text):
    return contractions.fix(text)

# Step 9: Remove punctuations and replace with space
def remove_punctuations(text):
    # Strip punctuation and ensure spacing
    punct_pattern = f"[{re.escape(string.punctuation)}]"
    return re.sub(punct_pattern, ' ', text)

# Step 10: Remove non-Latin words/characters
def remove_non_latin(text):
    return re.sub(r'[^\x00-\x7F]+', ' ', text)


import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words as nltk_english_words
import spacy
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('words', quiet=True)

# English vocabulary & tools
ENGLISH_VOCAB = set(w.lower() for w in nltk_english_words.words())
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOPWORDS
from malaya.dictionary import is_malay, is_english
from malaya.text.function import stopwords

lemmatizer = WordNetLemmatizer()
MALAYA_STOPWORDS = set(stopwords)

stemmer_factory = StemmerFactory()
malay_stemmer = stemmer_factory.create_stemmer()

MASKING_TAGS = {'urltoken', 'phonetoken', 'pricetoken', 'timetoken', 'datetoken'}

# Step 11: Word tokenization
def tokenize_words(text):
    if not isinstance(text, str) or not text.strip():
        return []
    return word_tokenize(text)

print("Step 11: Tokenizing text...")
df['tokens'] = df['clean_text'].apply(tokenize_words)

# Step 12: Language classification and tagging (Malaya Lexicon)
def tag_tokens(tokens):
    tagged = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower in MASKING_TAGS:
            tagged.append((token, "TAG"))
        elif is_malay(token_lower):
            tagged.append((token, "MALAY"))
        elif is_english(token_lower):
            tagged.append((token, "ENGLISH"))
        else:
            tagged.append((token, "UNKNOWN"))
    return tagged

print("Step 12: Tagging tokens via Malaya...")
df['tagged_tokens'] = df['tokens'].apply(tag_tokens)

# Step 13: Remove stop words
def remove_stopwords(tagged_tokens):
    cleaned = []
    for word, tag in tagged_tokens:
        word_lower = word.lower()
        if tag == "ENGLISH" and word_lower in SPACY_STOPWORDS:
            continue
        elif tag == "MALAY" and word_lower in MALAYA_STOPWORDS:
            continue
        if len(word) > 1 or word.isdigit() or word_lower in MASKING_TAGS:
            cleaned.append((word, tag))
    return cleaned

print("Step 13: Filtering stopwords...")
print(f"  Active English Stopwords: {len(SPACY_STOPWORDS)} words loaded via SpaCy")
print(f"  Active Malay Stopwords: {len(MALAYA_STOPWORDS)} words loaded via Malaya")
df['filtered_tokens'] = df['tagged_tokens'].apply(remove_stopwords)

# Step 14: Lemmatization for English & Stemming for Malay
def lemmatize_and_stem(tagged_tokens):
    processed = []
    for word, tag in tagged_tokens:
        if tag == "ENGLISH":
            processed.append((lemmatizer.lemmatize(word), tag))
        elif tag == "MALAY":
            processed.append((malay_stemmer.stem(word), tag))
        else:
            processed.append((word, tag))
    return processed

# Step 14: Lemmatizing and stemming tokens...
print("Step 14: Lemmatizing and stemming tokens...")
df['morph_tokens'] = df['filtered_tokens'].apply(lemmatize_and_stem)
df['clean_text'] = df['morph_tokens'].apply(lambda t_list: ' '.join([t[0] for t in t_list]))

# Step 14.1: Filter Out Empty & 0-Token Posts
initial_total_rows = len(df)
df = df[df['clean_text'].astype(str).str.strip().str.len() > 0].reset_index(drop=True)
dropped_rows = initial_total_rows - len(df)
print(f"\nStep 14.1: Filtered out {dropped_rows:,} empty/0-token posts ({dropped_rows/initial_total_rows*100:.2f}% of total).")
print(f"Cleaned Dataset for Model Training: {len(df):,} valid informative posts remaining.\n")

import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

# Step 15: 70/30 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df[target_cols], test_size=0.30, random_state=42
)

# Step 16: Fit TfidfVectorizer & Save
tfidf = TfidfVectorizer(min_df=3, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

# Step 17: Train One-vs-Rest Logistic Regression & Save
lr_model = OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
lr_model.fit(X_train_tfidf, y_train)
joblib.dump(lr_model, "logistic_regression_model.pkl")

# Step 18: Show top feature words per intent category
feature_names = tfidf.get_feature_names_out()
top_words_lr = {}
for i, label in enumerate(target_cols):
    coefs = lr_model.estimators_[i].coef_[0]
    top_indices = np.argsort(coefs)[-8:][::-1]
    top_words_lr[label] = [f"{feature_names[idx]} ({coefs[idx]:.2f})" for idx in top_indices]

print(f"\nTop 8 predictive features per intent class (Logistic Regression):")
print(pd.DataFrame(top_words_lr))

# Step 19: Logistic Regression Equations & Evaluation Metrics
print(f"\nLogistic Regression linear decision boundary equations:")
for i, label in enumerate(target_cols):
    coefs = lr_model.estimators_[i].coef_[0]
    intercept = lr_model.estimators_[i].intercept_[0]
    top_3_idx = np.argsort(coefs)[-3:][::-1]
    terms = " + ".join([f"({coefs[idx]:.2f} * {feature_names[idx]})" for idx in top_3_idx])
    print(f"[{label:<11}] z = {intercept:.2f} + {terms} + ...")

y_pred_lr = lr_model.predict(X_test_tfidf)
print(f"\nClassification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_lr, target_names=target_cols))


from sklearn.ensemble import RandomForestClassifier

# Step 20: Train One-vs-Rest Random Forest & Save
rf_model = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1))
rf_model.fit(X_train_tfidf, y_train)
joblib.dump(rf_model, "random_forest_model.pkl")

# Step 21: Display Random Forest Feature Importances & Evaluation
top_words_rf = {}
for i, label in enumerate(target_cols):
    importances = rf_model.estimators_[i].feature_importances_
    top_indices = np.argsort(importances)[-8:][::-1]
    top_words_rf[label] = [f"{feature_names[idx]} ({importances[idx]:.4f})" for idx in top_indices]

print(f"\nRandom Forest top Gini feature importances:")
print(pd.DataFrame(top_words_rf))

y_pred_rf = rf_model.predict(X_test_tfidf)
print(f"\nClassification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf, target_names=target_cols))

# Step 22: Consolidated All-Metrics Comparison Table
lr_accuracies = [(y_test[col] == y_pred_lr[:, idx]).mean() for idx, col in enumerate(target_cols)]
rf_accuracies = [(y_test[col] == y_pred_rf[:, idx]).mean() for idx, col in enumerate(target_cols)]
lr_precisions = [precision_score(y_test[col], y_pred_lr[:, idx], zero_division=0) for idx, col in enumerate(target_cols)]
rf_precisions = [precision_score(y_test[col], y_pred_rf[:, idx], zero_division=0) for idx, col in enumerate(target_cols)]
lr_recalls = [recall_score(y_test[col], y_pred_lr[:, idx], zero_division=0) for idx, col in enumerate(target_cols)]
rf_recalls = [recall_score(y_test[col], y_pred_rf[:, idx], zero_division=0) for idx, col in enumerate(target_cols)]
lr_f1_scores = [f1_score(y_test[col], y_pred_lr[:, idx], zero_division=0) for idx, col in enumerate(target_cols)]
rf_f1_scores = [f1_score(y_test[col], y_pred_rf[:, idx], zero_division=0) for idx, col in enumerate(target_cols)]

comp_table = pd.DataFrame({
    'Category': target_cols,
    'LR Accuracy': lr_accuracies,
    'RF Accuracy': rf_accuracies,
    'LR Precision': lr_precisions,
    'RF Precision': rf_precisions,
    'LR Recall': lr_recalls,
    'RF Recall': rf_recalls,
    'LR F1-Score': lr_f1_scores,
    'RF F1-Score': rf_f1_scores,
})

print(f"\nOverall Model Performance Comparison (Logistic Regression vs Random Forest):")
print(comp_table.to_string(index=False))


# Step 23: Preprocessing Pipeline Encapsulation & Model Artifacts Loading
def preprocess_pipeline(raw_text):
    text = remove_lowyat_signatures(raw_text)
    text = mask_elements(text)
    text = convert_emojis(text)
    text = to_lowercase(text)
    text = remove_elongated_content(text)
    text = normalize_slangs(text)
    text = fix_contractions(text)
    text = remove_punctuations(text)
    text = remove_non_latin(text)
    
    tokens = tokenize_words(text)
    tagged = tag_tokens(tokens)
    filtered = remove_stopwords(tagged)
    morph_tokens = lemmatize_and_stem(filtered)
    
    return ' '.join([t[0] for t in morph_tokens])

loaded_tfidf = joblib.load("tfidf_vectorizer.pkl")
loaded_lr = joblib.load("logistic_regression_model.pkl")
loaded_rf = joblib.load("random_forest_model.pkl")

# Step 24: Prediction function for unseen data & Interactive User Input
def predict_unseen(raw_text, model_type="logistic_regression"):
    cleaned_input = preprocess_pipeline(raw_text)
    
    if not cleaned_input.strip():
        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned_input,
            "predicted_intents": ["Neutral/None"]
        }
    
    vectorized_input = loaded_tfidf.transform([cleaned_input])
    
    if model_type == "logistic_regression":
        pred_array = loaded_lr.predict(vectorized_input)[0]
    elif model_type == "random_forest":
        pred_array = loaded_rf.predict(vectorized_input)[0]
    else:
        raise ValueError("Invalid model_type. Choose 'logistic_regression' or 'random_forest'.")
    
    predicted_intents = [target_cols[i] for i, val in enumerate(pred_array) if val == 1]
    
    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned_input,
        "predicted_intents": predicted_intents if predicted_intents else ["Neutral/None"]
    }

# Automated verification
test_samples = [
    "QUOTE(seller @ 10am) barang rosak teruk refund RM50 https://shop.com/scam please help!!!",
    "bila tarikh release movie baru tu? nak book ticket kat shopee",
    "i",
    "may i know which course you chosing"
]

print(f"\nEvaluating sample test posts:")
for s in test_samples:
    res = predict_unseen(s, model_type="logistic_regression")
    print(f"Post  : {res['raw_text']}")
    print(f"Clean : {res['cleaned_text']}")
    print(f"Intent: {res['predicted_intents']}\n")

# Interactive user input prompt (loops until typing 'q')
def interactive_prompt():
    print(f"\nInteractive Forum Comment Intent Predictor")
    print(f"Enter any test comment (or enter 'q' to quit):")
    while True:
        user_post = input("\nYour comment (or 'q' to quit): ").strip()
        if user_post.lower() == 'q':
            print("Exiting interactive session.")
            break
        if not user_post:
            print("Please enter a non-empty comment.")
            continue
            
        choice = input("Select model (lr / rf, default: lr): ").strip().lower()
        m_type = "random_forest" if choice == "rf" else "logistic_regression"
        res = predict_unseen(user_post, model_type=m_type)
        print(f"\nPrediction output:")
        print(f"Model used        : {'Logistic Regression' if m_type == 'logistic_regression' else 'Random Forest'}")
        print(f"Cleaned input     : {res['cleaned_text']}")
        print(f"Predicted intent  : {', '.join(res['predicted_intents'])}")

interactive_prompt()