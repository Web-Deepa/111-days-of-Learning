# Natural Language Processing(NLP)
import numpy as np
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # Fixed Tfidf typo
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

# 1. Text cleaning
print("Text cleaning-----")
def clean_text(text):
    text = text.lower() # lowercase
    text = re.sub(r'[^a-z\s]', '', text) # remove punctuation/number
    text = re.sub(r'\s+', ' ', text).strip() # Fixed: Replace spaces with ' ', not empty ''
    return text

samples = [
    "Hello ! I am learning Machine Learnig",
    "Python is easy and more efficcient",
    "The students are running on ground---",
]
print(f"{'Original':<40} {'Cleaned'}")
print("-" * 50)
for s in samples:
    print(f"{s:<40} {clean_text(s)}")

# 2. Tokenization and stop
print("\nTokenization and Stopword removal---")
# Fixed: Missing closing quote on 'the'
stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'i', 'we', 'you', 'he', 'she', 'it', 'they', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}

def tokenize(text):
    return clean_text(text).split()

def remove_stopwords(tokens):
    return [t for t in tokens if t not in stopwords]

sen = "The Students are running on  ground"
tokens = tokenize(sen)
filtered = remove_stopwords(tokens)
print(f"Original  : {sen}")
print(f"Tokens    : {tokens}")
print(f"Filtered  : {filtered}")

# 3. Stemming and Lemmatization
print("\nStemming and Lemmatization--")
try:
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer() # Fixed: Lammatizer -> Lemmatizer
    words = ['running', 'students', 'better', 'education', 'playing', 'game']
    
    # Fixed: Repaired broken block indentations below
    print(f"{'Word':<15} {'Stemmed':<15} {'Lemmatized'}")
    print("-" * 40)
    for w in words:
        print(f"{w:<15} {stemmer.stem(w):<15} {lemmatizer.lemmatize(w)}")
except ImportError: # Fixed: exept -> except
    print("Install nltk: pip install nltk")
    print("Simple manual stem demo:")
    words = ['running', 'studentss', 'better', 'education', 'playing', 'game']
    for w in words:
        stem = w.rstrip('ing').rstrip('s').rstrip('ies') or w
        print(f"  {w} → {stem}")

# 4. CountVectorizer
print("\nCountVectorizer-Bag of words--")
corpus = [
    'I am learning ML',
    'Fond of Python',
    'Let change by coding'
]
cv = CountVectorizer() # Fixed: CountVcetorizer -> CountVectorizer
bow = cv.fit_transform(corpus)
print(f"Vocabulary : {cv.get_feature_names_out().tolist()}")
print(f"Shape      : {bow.shape}  (documents × words)")
print(f"\nBoW matrix:")
print(bow.toarray())

# 5. TF-IDF Vectorizer
print("\nTF-IDF Vectorizer----")
tfidf = TfidfVectorizer() # Fixed: TfidVectorizer -> TfidfVectorizer
tfidf_mat = tfidf.fit_transform(corpus)
print("TF-IDF scores (higher = more important word in that doc):")
df_tfidf = pd.DataFrame(
    tfidf_mat.toarray().round(3),
    columns=tfidf.get_feature_names_out()
)
print(df_tfidf)
