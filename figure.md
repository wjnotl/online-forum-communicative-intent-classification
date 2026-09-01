# BMCS2203 Assignment Figure Guide and Implementation

### Figure 3.2

* **Exact Word Document Placement:** In Section 3.2.2, replace `[Insert Figure 3.X here. A screenshot of the actual dataframe head from your notebook, showing a handful of real rows with all eight columns visible.]`[cite: 1]. Place this directly below **Table 3.1 (Data Dictionary)** and directly above the paragraph beginning with *"Each of the six category columns is an independent binary flag..."*[cite: 1].
* **Caption:** `Figure 3.2: Sample preview of the raw forum dataset showing post identifiers text entries and multi-label intent flags`
* **Notebook Code:**
```python
printmd("#### Figure 3.2: Preview of First 5 Dataset Records")
display(df.head(5))
```
* **Notebook Placement:** In Section 1.1 (`Load Dataset & Preview Records`), run directly inside or below Code Cell 3[cite: 2].

---

### Figure 3.3

* **Exact Word Document Placement:** In Section 3.2.2, replace `[Insert Figure 3.X here. Please pull one real Complaint labelled row from your own dataset here, since I should not invent an example for this category.]`[cite: 1]. Place this directly below the **Complaint** description paragraph and directly above the **Opinion** subsection heading[cite: 1].
* **Caption:** `Figure 3.3: Representative forum post categorized under Complaint expressing backward looking consumer dissatisfaction`
* **Notebook Code:**
```python
printmd("#### Figure 3.3: Complaint Labeled Post Example")
sample_complaint = df[(df['Complaint'] == 1) & (df['text'].str.len() > 20)][['id', 'text', 'Complaint', 'Opinion', 'Inquiry']].head(1)
display(sample_complaint)
```
* **Notebook Placement:** In Section 1.1 (`Load Dataset & Preview Records`), create a new code cell directly below Code Cell 3[cite: 2].

---

### Figure 3.4

* **Exact Word Document Placement:** In Section 3.2.2, replace `[Insert Figure 3.X here. Please pull one real Opinion labelled row from your own dataset, ideally one that is not also labelled Complaint, so the distinction is visible in a genuine example.]`[cite: 1]. Place this directly below the **Opinion** description paragraph and directly above the **Information** subsection heading[cite: 1].
* **Caption:** `Figure 3.4: Representative forum post categorized exclusively under Opinion expressing subjective viewpoints without complaint grievances`
* **Notebook Code:**
```python
printmd("#### Figure 3.4: Pure Opinion Labeled Post Example")
sample_opinion = df[(df['Opinion'] == 1) & (df['Complaint'] == 0) & (df['text'].str.len() > 20)][['id', 'text', 'Opinion', 'Complaint', 'Information']].head(1)
display(sample_opinion)
```
* **Notebook Placement:** In Section 1.1 (`Load Dataset & Preview Records`), paste directly below the Figure 3.3 code cell.

---

### Figure 3.5

* **Exact Word Document Placement:** In Section 3.2.3, replace `[Insert Figure 3.X here. A short screenshot showing two or three real raw posts from your dataset that clearly display this messiness, ideally one with a quote block visible, one with an emoji, and one that mixes English and Malay in the same sentence.]`[cite: 1]. Place this directly below the main paragraph of Section 3.2.3 and directly above the Section 3.2.4 heading[cite: 1].
* **Caption:** `Figure 3.5: Raw forum comment samples illustrating BBCode quotation artefacts emoji strings and multilingual code switching`
* **Notebook Code:**
```python
printmd("#### Figure 3.5: Raw Forum Post Samples with Noise and Code-Switching")
sample_quote = df[df['text'].str.contains(r'QUOTE\(', regex=True, na=False)].head(1)
sample_emoji = df[df['text'].str.contains(r'[\U00010000-\U0010ffff]', regex=True, na=False)].head(1)
sample_mixed = df[df['text'].str.contains(r'\b(padu|betul|keta|dlm|bunuh)\b', regex=True, na=False)].head(1)

noise_samples_df = pd.concat([sample_quote, sample_emoji, sample_mixed]).drop_duplicates()
display(noise_samples_df[['id', 'text']])
```
* **Notebook Placement:** In Section 1.1 (`Load Dataset & Preview Records`), paste directly below the Figure 3.4 code cell.

---

### Figure 3.6

* **Exact Word Document Placement:** In Section 3.2.4, replace `[Insert Figure 3.X here. Before and after example of a real post that contained a quote block or forum signature.]`[cite: 1]. Place this directly below the **Step 1 (Removal of forum platform markup)** paragraph and directly above the **Step 2** paragraph[cite: 1].
* **Caption:** `Figure 3.6: Preprocessing Step 1 elimination of BBCode quotation blocks HTML tags and automated edit signatures`
* **Notebook Code:**
```python
raw_sample_step1 = df[df['text'].str.contains(r'QUOTE\(', regex=True, na=False)]['text'].iloc[0]
cleaned_sample_step1 = remove_forum_markup(raw_sample_step1)

show_step_diff(
    step_num=1,
    step_name="Removal of Forum Platform Markup",
    before_text=raw_sample_step1,
    after_text=cleaned_sample_step1
)
```
* **Notebook Placement:** In Section 2 (Preprocessing Pipeline), paste into a code cell directly below your Step 1 markup cleaning function[cite: 1].

---

### Figure 3.7

* **Exact Word Document Placement:** In Section 3.2.4, replace `[Insert Figure 3.X here. Before and after example of a real post containing a URL, price, or phone number being replaced with its placeholder token.]`[cite: 1]. Place this directly below the **Step 2 (Masking of variable entities...)** paragraph and directly above the **Step 3** paragraph[cite: 1].
* **Caption:** `Figure 3.7: Preprocessing Step 2 entity masking replacing unbounded URLs price amounts and timestamps with standardized tokens`
* **Notebook Code:**
```python
raw_sample_step2 = df[df['text'].str.contains(r'https?://|www\.', regex=True, na=False)]['text'].iloc[0]
cleaned_sample_step2 = mask_entities(raw_sample_step2)

show_step_diff(
    step_num=2,
    step_name="Masking of Variable Entities to Tokens",
    before_text=raw_sample_step2,
    after_text=cleaned_sample_step2
)
```
* **Notebook Placement:** In Section 2 (Preprocessing Pipeline), paste into a code cell directly below your Step 2 entity masking function[cite: 1].

---

### Figure 3.8

* **Exact Word Document Placement:** In Section 3.2.4, replace `[Insert Figure 3.X here. Before and after example of a real post with an elongated word or laughter pattern being normalised.]`[cite: 1]. Place this directly below the **Step 3 (Normalisation of elongated words...)** paragraph and directly above the **Step 4** paragraph[cite: 1].
* **Caption:** `Figure 3.8: Preprocessing Step 3 character elongation compression and laughter sequence standardization`
* **Notebook Code:**
```python
raw_sample_step3 = df[df['text'].str.contains(r'(.)\1{3,}|hahaha|wkwk', case=False, regex=True, na=False)]['text'].iloc[0]
cleaned_sample_step3 = normalize_elongation_and_laughter(raw_sample_step3)

show_step_diff(
    step_num=3,
    step_name="Normalisation of Elongated Words and Laughter",
    before_text=raw_sample_step3,
    after_text=cleaned_sample_step3
)
```
* **Notebook Placement:** In Section 2 (Preprocessing Pipeline), paste into a code cell directly below your Step 3 elongation normalization function[cite: 1].

---

### Figure 3.9

* **Exact Word Document Placement:** In Section 3.2.4, replace `[Insert Figure 3.X here. Before and after example of a real post containing an emoji being converted to its text description.]`[cite: 1]. Place this directly below the **Step 4 (Conversion of emojis into text)** paragraph and directly above the **Step 5** paragraph[cite: 1].
* **Caption:** `Figure 3.9: Preprocessing Step 4 emoji conversion mapping graphical unicode symbols into textual semantic descriptors`
* **Notebook Code:**
```python
raw_sample_step4 = df[df['text'].str.contains(r'[\U00010000-\U0010ffff]', regex=True, na=False)]['text'].iloc[0]
cleaned_sample_step4 = convert_emojis_to_text(raw_sample_step4)

show_step_diff(
    step_num=4,
    step_name="Conversion of Emojis to Text",
    before_text=raw_sample_step4,
    after_text=cleaned_sample_step4
)
```
* **Notebook Placement:** In Section 2 (Preprocessing Pipeline), paste into a code cell directly below your Step 4 emoji conversion function[cite: 1].

---

### Figure 3.10

* **Exact Word Document Placement:** In Section 3.2.4, replace `[Insert Figure 3.X here. Before and after example of a real post containing Malay or English slang being normalised. A screenshot of a handful of rows from your slang dictionary itself would also work well here.]`[cite: 1]. Place this directly below the **Step 6 (Slang normalisation)** paragraph and directly above the **Step 7** paragraph[cite: 1].
* **Caption:** `Figure 3.10: Preprocessing Step 6 hybrid slang normalization resolving Malay and English internet abbreviations`
* **Notebook Code:**
```python
raw_sample_step6 = df[df['text'].str.contains(r'\b(dlm|keta|bkn|sbb|idk|tak|yg)\b', case=False, regex=True, na=False)]['text'].iloc[0]
cleaned_sample_step6 = normalize_slang(raw_sample_step6)

show_step_diff(
    step_num=6,
    step_name="Slang Normalisation via Bilingual Dictionary",
    before_text=raw_sample_step6,
    after_text=cleaned_sample_step6
)
```
* **Notebook Placement:** In Section 2 (Preprocessing Pipeline), paste into a code cell directly below your Step 6 slang normalization function[cite: 1].

---

### Figure 3.11

* **Exact Word Document Placement:** In Section 3.2.4, replace `[Insert Figure 3.X here. Before and after example of a real post with a contraction, ideally one without an apostrophe, being expanded.]`[cite: 1]. Place this directly below the **Step 7 (Expansion of contractions)** paragraph and directly above the **Step 8** paragraph[cite: 1].
* **Caption:** `Figure 3.11: Preprocessing Step 7 grammatical contraction expansion across standard and apostrophe free informal forms`
* **Notebook Code:**
```python
raw_sample_step7 = df[df['text'].str.contains(r"\b(dont|can't|won't|isn't|didn't|tat's)\b", case=False, regex=True, na=False)]['text'].iloc[0]
cleaned_sample_step7 = expand_contractions(raw_sample_step7)

show_step_diff(
    step_num=7,
    step_name="Expansion of Contractions",
    before_text=raw_sample_step7,
    after_text=cleaned_sample_step7
)
```
* **Notebook Placement:** In Section 2 (Preprocessing Pipeline), paste into a code cell directly below your Step 7 contraction expansion function[cite: 1].

---

### Figure 3.12

* **Exact Word Document Placement:** In Section 3.2.4, replace `[Insert Figure 3.X here. Before and after example of a real post with a glued alphanumeric word being split.]`[cite: 1]. Place this directly below the **Step 9 (Splitting of glued alphanumeric text...)** paragraph and directly above the **Step 10** paragraph[cite: 1].
* **Caption:** `Figure 3.12: Preprocessing Step 9 boundary splitting of glued alphanumeric tokens and stripping of contextual standalone numbers`
* **Notebook Code:**
```python
raw_sample_step9 = df[df['text'].str.contains(r'[a-zA-Z]+\d+|\d+[a-zA-Z]+', regex=True, na=False)]['text'].iloc[0]
cleaned_sample_step9 = split_alphanumeric_and_remove_numbers(raw_sample_step9)

show_step_diff(
    step_num=9,
    step_name="Splitting Glued Alphanumeric and Number Removal",
    before_text=raw_sample_step9,
    after_text=cleaned_sample_step9
)
```
* **Notebook Placement:** In Section 2 (Preprocessing Pipeline), paste into a code cell directly below your Step 9 alphanumeric splitting function[cite: 1].

---

### Figure 3.13

* **Exact Word Document Placement:** In Section 3.2.4, replace `[Insert Figure 3.X here. A screenshot showing one real cleaned sentence before tokenisation next to its tokenised list form.]`[cite: 1]. Place this directly below the **Step 11 (Word tokenisation)** paragraph and directly above the **Step 12** paragraph[cite: 1].
* **Caption:** `Figure 3.13: Preprocessing Step 11 word tokenization transforming continuous strings into discrete token lists`
* **Notebook Code:**
```python
raw_sample_step11 = df.loc[0, 'text']
tokenized_sample_step11 = word_tokenize(raw_sample_step11)

show_step_diff(
    step_num=11,
    step_name="Word Tokenisation",
    before_text=raw_sample_step11,
    after_text=str(tokenized_sample_step11)
)
```
* **Notebook Placement:** In Section 2 (Preprocessing Pipeline), paste into a code cell directly below your Step 11 word tokenization execution[cite: 1].

---

### Figure 3.14

* **Exact Word Document Placement:** In Section 3.2.4, replace `[Insert Figure 3.X here. A screenshot of one real code switched post with each token labelled by its assigned tag, English, Malay, or unknown.]`[cite: 1]. Place this directly below the **Step 12 (Language and entity tagging)** paragraph and directly above the **Step 13** paragraph[cite: 1].
* **Caption:** `Figure 3.14: Preprocessing Step 12 bilingual language classification and placeholder entity tag routing per token`
* **Notebook Code:**
```python
sample_tokens_step12 = ['situasi', 'padu', 'URLTOKEN', 'player', 'health']
tagged_output_step12 = tag_language_and_entities(sample_tokens_step12)

printmd("#### Figure 3.14: Language and Entity Tagging Breakdown")
display(pd.DataFrame(tagged_output_step12, columns=['Token', 'Assigned Tag']))
```
* **Notebook Placement:** In Section 2 (Preprocessing Pipeline), paste into a code cell directly below your Step 12 language tagging function[cite: 1].

---

### Figure 3.15

* **Exact Word Document Placement:** In Section 3.2.4, replace `[Insert Figure 3.X here. Before and after example showing one real English word being lemmatised and one real Malay word being stemmed.]`[cite: 1]. Place this directly below the **Step 14 (Lemmatisation of English tokens and stemming of Malay tokens)** paragraph and directly above the **Step 15** paragraph[cite: 1].
* **Caption:** `Figure 3.15: Preprocessing Step 14 morphological root reduction comparing English POS lemmatization and Malay affix stemming`
* **Notebook Code:**
```python
eng_before = "playing (Verb)"
eng_after = lemmatize_english_token("playing", pos='v')

malay_before = "memandukan (Affixed)"
malay_after = stem_malay_token("memandukan")

printmd("#### Figure 3.15: English Lemmatization vs Malay Stemming")
display(pd.DataFrame([
    {'Language Task': 'English (WordNet POS Lemmatization)', 'Input Token': eng_before, 'Root Output': eng_after},
    {'Language Task': 'Malay (Sastrawi Stemming)', 'Input Token': malay_before, 'Root Output': malay_after}
]))
```
* **Notebook Placement:** In Section 2 (Preprocessing Pipeline), paste into a code cell directly below your Step 14 root reduction execution[cite: 1].

---

### Figure 3.16

* **Exact Word Document Placement:** In Section 3.2.4, replace `[Insert Figure 3.X here. Before and after example of a real Inquiry post showing how words like “what” or “why” survive stopword removal while ordinary stopwords around them are stripped.]`[cite: 1]. Place this directly below the **Step 15 (Stopword removal with a preserved exception list)** paragraph and directly above the **Step 16** paragraph[cite: 1].
* **Caption:** `Figure 3.16: Preprocessing Step 15 stopword filtering showing retention of diagnostic Inquiry interrogatives`
* **Notebook Code:**
```python
sample_inquiry_tokens = ["currently", "he", "is", "playing", "for", "india", "why", "how", "what"]
filtered_inquiry_tokens = remove_stopwords_with_exceptions(sample_inquiry_tokens)

show_step_diff(
    step_num=15,
    step_name="Stopword Removal with Protected Interrogatives",
    before_text=str(sample_inquiry_tokens),
    after_text=str(filtered_inquiry_tokens)
)
```
* **Notebook Placement:** In Section 2 (Preprocessing Pipeline), paste into a code cell directly below your Step 15 stopword removal function[cite: 1].

---

### Figure 3.17

* **Exact Word Document Placement:** In Section 3.2.5, replace `[Insert Figure 3.X here. A screenshot from your own notebook of a handful of real posts alongside their top weighted TF-IDF terms.]`[cite: 1]. Place this directly below the final paragraph of Section 3.2.5 ending with *"...produced a vocabulary of 72,943 unique features."*[cite: 1].
* **Caption:** `Figure 3.17: Vectorized document representation displaying learned unigram and bigram TF-IDF numerical weights`
* **Notebook Code:**
```python
printmd("#### Figure 3.17: Sample Document TF-IDF Feature Weights")
sample_tfidf_df = pd.DataFrame(
    X_train_tfidf[:3].toarray(),
    columns=tfidf_vectorizer.get_feature_names_out(),
    index=['Post ID 1', 'Post ID 2', 'Post ID 3']
)
display(sample_tfidf_df.iloc[:, :7])
```
* **Notebook Placement:** In Section 3.2.5 (TF-IDF Vectorization), paste into a code cell directly below your training TF-IDF vectorizer fitting cell[cite: 1].

---

### Figure 3.18

* **Exact Word Document Placement:** In Section 3.3.3, replace `[Insert Figure 3.X here. A simple diagram showing the dataset splitting into a training portion and a testing portion, with only the training portion feeding into the model during learning.]`[cite: 1]. Place this directly below the final paragraph of Section 3.3.3[cite: 1].
* **Caption:** `Figure 3.18: Dataset partitioning workflow demonstrating isolated TF-IDF vocabulary fitting on the training split`
* **Notebook Code:**
```python
fig, ax = plt.subplots(figsize=(10, 2.8))
ax.axis('off')

bbox_clean = dict(boxstyle='round,pad=0.5', facecolor='#e2e8f0', edgecolor='#475569', lw=1.2)
bbox_train = dict(boxstyle='round,pad=0.5', facecolor='#e2e8f0', edgecolor='#475569', lw=1.2)
bbox_test = dict(boxstyle='round,pad=0.5', facecolor='#e2e8f0', edgecolor='#475569', lw=1.2)

ax.text(0.12, 0.5, "Cleaned Dataset\n(98,492 Posts)", ha='center', va='center', bbox=bbox_clean, fontsize=9.5, fontweight='bold')
ax.annotate("", xy=(0.34, 0.72), xytext=(0.24, 0.55), arrowprops=dict(arrowstyle="->", lw=1.5, color='#334155'))
ax.annotate("", xy=(0.34, 0.28), xytext=(0.24, 0.45), arrowprops=dict(arrowstyle="->", lw=1.5, color='#334155'))

ax.text(0.58, 0.75, "Training Set (80%)\nfit_transform(X_train)\nVocabulary: 72,943 Features", ha='center', va='center', bbox=bbox_train, fontsize=9)
ax.text(0.58, 0.25, "Testing Set (20%)\ntransform(X_test)\n(No Data Leakage)", ha='center', va='center', bbox=bbox_test, fontsize=9)

plt.title("Figure 3.18: Dataset Train and Test Split Architecture", fontsize=11, fontweight='bold', pad=10)
plt.tight_layout()
plt.show()
```
* **Notebook Placement:** In Section 3.3.3 (Train/Test Split), paste into a code cell directly below `train_test_split(X, y, test_size=0.2)`[cite: 1].

---

### Figure 3.19

* **Exact Word Document Placement:** In Section 3.3.5, replace `[Insert Figure 3.X here. Log loss formula, showing negative y times the log of p, plus one minus y times the log of one minus p, where y is the true label and p is the model's predicted probability.]`[cite: 1]. Place this directly below the sentence ending with *"...This is measured using log loss."* and above *"This formula splits into two simple cases..."*[cite: 1].
* **Caption:** `Figure 3.19: Binary cross entropy log loss optimization objective minimized across training posts in Logistic Regression`
* **Notebook Code:**
```python
printmd("#### Figure 3.19: Log Loss Binary Cross-Entropy Formula")
printmd(r"$$\mathcal{L}_{\log}(y, p) = -\left[ y \ln(p) + (1 - y) \ln(1 - p) \right]$$")
```
* **Notebook Placement:** In Section 3.3.5 (Logistic Regression), paste into a markdown or code cell directly above model training[cite: 1].

---

### Figure 3.20

* **Exact Word Document Placement:** In Section 3.3.5, replace `[Insert Figure 3.X here. Sigmoid probability curve for one of the six categories, showing how a raw score is converted into a probability between 0 and 1, with the 0.5 decision threshold marked.]`[cite: 1]. Place this directly below the paragraph ending with *"...process is called gradient descent..."* and above the paragraph starting with *"The specific words each category's classifier learned to weight..."*[cite: 1].
* **Caption:** `Figure 3.20: Sigmoid activation curve mapping linear decision scores into calibrated probabilities with the 0.5 classification threshold`
* **Notebook Code:**
```python
z_vals = np.linspace(-7, 7, 250)
sig_vals = 1 / (1 + np.exp(-z_vals))

plt.figure(figsize=(7, 4))
plt.plot(z_vals, sig_vals, color='#1e3a8a', lw=2, label=r'$\sigma(z) = \frac{1}{1 + e^{-z}}$')
plt.axvline(0, color='#b91c1c', linestyle='--', lw=1.2, label='Threshold (z=0, p=0.5)')
plt.axhline(0.5, color='#94a3b8', linestyle=':')
plt.title("Figure 3.20: Sigmoid Activation Function and Probability Mapping", fontsize=11, fontweight='bold')
plt.xlabel("Raw Linear Score (z)", fontsize=9.5, fontweight='bold')
plt.ylabel("Calibrated Probability (p)", fontsize=9.5, fontweight='bold')
plt.legend(frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```
* **Notebook Placement:** In Section 3.3.5 (Logistic Regression), paste directly below the Figure 3.19 formula cell[cite: 1].

---

### Figure 3.21

* **Exact Word Document Placement:** In Section 3.3.6, replace `[Insert Figure 3.X here. Linear SVC objective formula, showing one half times the magnitude of the coefficient vector squared, plus C times the total hinge loss summed across all training posts.]`[cite: 1]. Place this directly below the paragraph ending with *"...and make as few classification mistakes as possible."* and above the paragraph starting with *"The first part of this formula..."*[cite: 1].
* **Caption:** `Figure 3.21: Soft margin Linear Support Vector Classifier mathematical objective balancing margin width and cumulative hinge loss`
* **Notebook Code:**
```python
printmd("#### Figure 3.21: Linear SVC Optimization Objective Formula")
printmd(r"$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{N} \max\left(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b)\right)$$")
```
* **Notebook Placement:** In Section 3.3.6 (Linear SVC), paste into a markdown or code cell directly above the Linear SVC setup[cite: 1].

---

### Figure 3.22

* **Exact Word Document Placement:** In Section 3.3.6, replace `[Insert Figure 3.X here. Hinge loss formula, showing the larger of zero or one minus, y times the raw score, where y is the true label encoded as positive one or negative one.]`[cite: 1]. Place this directly below the sentence *"The second part is hinge loss."* and above the paragraph starting with *"If a post is correctly classified..."*[cite: 1].
* **Caption:** `Figure 3.22: Piecewise linear hinge loss penalty function evaluated for binary margin constraints`
* **Notebook Code:**
```python
printmd("#### Figure 3.22: Hinge Loss Formula")
printmd(r"$$\mathcal{L}_{\text{hinge}}(y, \hat{y}) = \max(0, 1 - y \cdot \hat{y})$$")
```
* **Notebook Placement:** In Section 3.3.6 (Linear SVC), paste directly below the Figure 3.21 formula cell[cite: 1].

---

### Figure 3.23

* **Exact Word Document Placement:** In Section 3.3.6, replace `[Insert Figure 3.X here. Simplified two feature illustration of the Linear SVC decision boundary, the margin around it, and the support vectors sitting on the edge of the margin. Caption should note this is a simplified illustration for conceptual purposes, since the actual model operates across 72,943 features.]`[cite: 1]. Place this directly below the paragraph ending with *"...and was never calibrated to behave like one."* and above the paragraph starting with *"The specific words each category's classifier learned..."*[cite: 1].
* **Caption:** `Figure 3.23: Simplified two-feature geometric diagram of the Linear Support Vector Classifier decision boundary margins and support vectors`
* **Notebook Code:**
```python
plt.figure(figsize=(7, 4.2))
plt.plot([-3, 3], [-2, 2], color='#0f172a', lw=2, label='Decision Boundary ($w^T x + b = 0$)')
plt.plot([-3, 3], [-1, 3], color='#1e3a8a', linestyle='--', label='Positive Margin ($w^T x + b = +1$)')
plt.plot([-3, 3], [-3, 1], color='#b91c1c', linestyle='--', label='Negative Margin ($w^T x + b = -1$)')

plt.scatter([0.5, -0.5], [1.75, -1.75], s=120, facecolors='none', edgecolors='#ca8a04', lw=2, label='Support Vectors', zorder=5)
plt.scatter([1, 1.6, 2.2], [2.4, 3.1, 2.7], color='#1e3a8a', s=50, label='Class +1')
plt.scatter([-1, -1.6, -2.2], [-2.4, -3.1, -2.7], color='#b91c1c', s=50, label='Class -1')

plt.title("Figure 3.23: Linear SVC Margin and Hyperplane (Conceptual 2D)", fontsize=11, fontweight='bold')
plt.xlabel("Feature Dimension 1", fontsize=9.5, fontweight='bold')
plt.ylabel("Feature Dimension 2", fontsize=9.5, fontweight='bold')
plt.legend(loc='upper left', fontsize=8.5, frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```
* **Notebook Placement:** In Section 3.3.6 (Linear SVC), paste directly below the Figure 3.22 formula cell[cite: 1].

---

### Figure 3.24

* **Exact Word Document Placement:** In Section 3.4, replace `[Insert Figure 3.X here. A labelled diagram of a confusion matrix, empty template only, no real numbers, since this section only describes the metrics rather than reporting outcomes.]`[cite: 1]. Place this directly below the paragraph ending with *"...that truly belonged to the category but was not flagged."* and above the **Accuracy** subsection heading[cite: 1].
* **Caption:** `Figure 3.24: Binary classification confusion matrix quadrant framework defining positive and negative prediction states`
* **Notebook Code:**
```python
matrix_layout = [["True Positive (TP)", "False Negative (FN)"],
                 ["False Positive (FP)", "True Negative (TN)"]]

plt.figure(figsize=(5.5, 3.8))
sns.heatmap([[1, 0], [0, 1]], annot=matrix_layout, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Predicted Positive", "Predicted Negative"],
            yticklabels=["Actual Positive", "Actual Negative"],
            annot_kws={'fontsize': 10, 'fontweight': 'bold'})
plt.title("Figure 3.24: Binary Confusion Matrix Template Layout", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()
```
* **Notebook Placement:** In Section 3.4 (Evaluation Metrics), paste into a code cell directly above metric definitions[cite: 1].

---

### Figure 3.25

* **Exact Word Document Placement:** In Section 3.4, replace the four separate placeholder tags under the metric definitions (`[Insert Figure 3.X here. Accuracy formula...]`, `[Insert Figure 3.X here. Precision formula...]`, `[Insert Figure 3.X here. Recall formula...]`, and `[Insert Figure 3.X here. F1 Score formula...]`) with this combined formula representation[cite: 1].
* **Caption:** `Figure 3.25: Mathematical evaluation formulas for per-category classification performance metrics`
* **Notebook Code:**
```python
printmd("#### Figure 3.25: Mathematical Formulas for Evaluation Metrics")
printmd(r"""
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}, \quad \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
""")
```
* **Notebook Placement:** In Section 3.4 (Evaluation Metrics), paste directly below the Figure 3.24 confusion matrix template[cite: 1].

---

### Figure 4.1

* **Exact Word Document Placement:** In Section 4.1.1, replace `[Insert Figure 4.1 here. Screenshot of the overall performance comparison table, showing Accuracy, Precision, Recall, and F1 Score for both models across all six categories plus the macro average row, produced under Step 26.]`[cite: 1]. Place this directly below the **4.1.1 Overall Performance Comparison** heading and above the paragraph starting with *"This table reports accuracy, precision..."*[cite: 1].
* **Caption:** `Figure 4.1: Overall performance metrics comparison table for Logistic Regression and Linear SVC across communicative categories and Macro Average`
* **Notebook Code:**
```python
printmd("#### Figure 4.1: Overall Performance Comparison Table")
display(df_model_comparison)
```
* **Notebook Placement:** In Step 26 (Model Comparison), run directly inside the Step 26 summary table cell[cite: 1].

---

### Figure 4.2

* **Exact Word Document Placement:** In Section 4.1.2, replace `[Insert Figure 4.2 here. Screenshot of the six confusion matrix heatmaps produced for Logistic Regression, produced under Step 22.2.]`[cite: 1]. Place this directly below the **4.1.2 Confusion Matrices** heading and directly above Figure 4.3[cite: 1].
* **Caption:** `Figure 4.2: Confusion matrix heatmaps for the six One-vs-Rest binary Logistic Regression classifiers`
* **Notebook Code:**
```python
plot_logistic_regression_confusion_matrices(y_test, y_pred_lr, target_cols)
```
* **Notebook Placement:** In Step 22.2 (Logistic Regression Evaluation), run inside the Step 22.2 plotting cell[cite: 1].

---

### Figure 4.3

* **Exact Word Document Placement:** In Section 4.1.2, replace `[Insert Figure 4.3 here. Screenshot of the six confusion matrix heatmaps produced for Linear SVC, produced under Step 25.1.]`[cite: 1]. Place this directly below Figure 4.2 and above the paragraph starting with *"These two figures show the true positive..."*[cite: 1].
* **Caption:** `Figure 4.3: Confusion matrix heatmaps for the six One-vs-Rest binary Linear Support Vector Classifiers`
* **Notebook Code:**
```python
plot_linear_svc_confusion_matrices(y_test, y_pred_svc, target_cols)
```
* **Notebook Placement:** In Step 25.1 (Linear SVC Evaluation), run inside the Step 25.1 plotting cell[cite: 1].

---

### Figure 4.4

* **Exact Word Document Placement:** In Section 4.1.3, replace `[Insert Figure 4.4 here. Screenshot of the 2 by 2 grid of bar charts comparing Accuracy, Precision, Recall, and F1 Score per category between the two models, produced under Step 26.]`[cite: 1]. Place this directly below the **4.1.3 Per-Category Metric Comparison** heading and above the paragraph starting with *"This figure places both models side by side..."*[cite: 1].
* **Caption:** `Figure 4.4: Comparative bar charts showing Accuracy Precision Recall and F1 Score across communicative categories for both models`
* **Notebook Code:**
```python
plot_model_comparison_bars(df_model_comparison)
```
* **Notebook Placement:** In Step 26 (Model Comparison), run inside the 2x2 comparison plotting cell[cite: 1].

---

### Figure 4.5

* **Exact Word Document Placement:** In Section 4.1.4, replace `[Insert Figure 4.5 here. Screenshot of the six sigmoid probability curve plots, one per category, produced under Step 21.3.]`[cite: 1]. Place this directly below the **4.1.4 Sigmoid Probability Curves** heading and above the paragraph starting with *"This figure applies only to Logistic Regression..."*[cite: 1].
* **Caption:** `Figure 4.5: Calibrated sigmoid probability curves across all six categories for Logistic Regression`
* **Notebook Code:**
```python
plot_sigmoid_probability_curves(lr_models, X_test_tfidf, target_cols)
```
* **Notebook Placement:** In Step 21.3 (Logistic Regression Curve Visualization), run inside the Step 21.3 subplot plotting cell[cite: 1].

---

### Figure 4.6

* **Exact Word Document Placement:** In Section 4.1.5, replace `[Insert Figure 4.6 here. Screenshot of the top 15 predictive words table per category for Logistic Regression, produced under Step 21.1.]`[cite: 1]. Place this directly below the **4.1.5 Top Predictive Words per Category** heading and above Figure 4.7[cite: 1].
* **Caption:** `Figure 4.6: Top positive predictive features per category ranked by coefficient magnitude for Logistic Regression`
* **Notebook Code:**
```python
printmd("#### Figure 4.6: Top 15 Predictive Features per Category (Logistic Regression)")
display(df_top_words_lr)
```
* **Notebook Placement:** In Step 21.1 (Logistic Regression Feature Extraction), paste into the display cell under Step 21.1[cite: 1].

---

### Figure 4.7

* **Exact Word Document Placement:** In Section 4.1.5, replace `[Insert Figure 4.7 here. Screenshot of the top 15 predictive words table per category for Linear SVC, produced under Step 24.1.]`[cite: 1]. Place this directly below Figure 4.6 and above the paragraph starting with *"These two tables list the words and short phrases..."*[cite: 1].
* **Caption:** `Figure 4.7: Top positive predictive features per category ranked by hyperplane weight for Linear Support Vector Classifier`
* **Notebook Code:**
```python
printmd("#### Figure 4.7: Top 15 Predictive Features per Category (Linear SVC)")
display(df_top_words_svc)
```
* **Notebook Placement:** In Step 24.1 (Linear SVC Feature Extraction), paste into the display cell under Step 24.1[cite: 1].

---

### Figure 4.8

* **Exact Word Document Placement:** In Section 4.1.6, replace `[Insert Figure 4.8 here. Screenshot of the printed decision boundary equations for Logistic Regression, produced under Step 21.2.]`[cite: 1]. Place this directly below the **4.1.6 Learned Decision Boundary Equations** heading and above Figure 4.9[cite: 1].
* **Caption:** `Figure 4.8: Learned decision boundary linear equations with model intercepts and dominant feature weights for Logistic Regression`
* **Notebook Code:**
```python
printmd("#### Figure 4.8: Learned Decision Boundary Equations (Logistic Regression)")
display_decision_boundary_equations_lr(lr_models, feature_names, target_cols)
```
* **Notebook Placement:** In Step 21.2 (Logistic Regression Decision Equations), run inside the Step 21.2 equation display cell[cite: 1].

---

### Figure 4.9

* **Exact Word Document Placement:** In Section 4.1.6, replace `[Insert Figure 4.9 here. Screenshot of the printed decision boundary equations for Linear SVC, produced under Step 24.2.]`[cite: 1]. Place this directly below Figure 4.8 and above the paragraph starting with *"This shows, in equation form, exactly how the raw score..."*[cite: 1].
* **Caption:** `Figure 4.9: Learned decision boundary linear equations with model intercepts and dominant hyperplane weights for Linear Support Vector Classifier`
* **Notebook Code:**
```python
printmd("#### Figure 4.9: Learned Decision Boundary Equations (Linear SVC)")
display_decision_boundary_equations_svc(svc_models, feature_names, target_cols)
```
* **Notebook Placement:** In Step 24.2 (Linear SVC Decision Equations), run inside the Step 24.2 equation display cell[cite: 1].