import pandas as pd
import numpy as np
import string
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# 0. Download NLTK Data (run once)
# -------------------------------
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv('./data/nyc_311_shuffled_sample.csv')

# -------------------------------
# 2. Combine Text Columns
# -------------------------------
df['text'] = (
    df['descriptor'].fillna('') + " " +
    df['resolution_description'].fillna('')
)

# Keep only needed columns
df = df[['text', 'complaint_type']].dropna()

# Remove empty text rows
df = df[df['text'].str.strip() != ""]

# -------------------------------
# 3. Remove Rare Classes (VERY IMPORTANT)
# -------------------------------
class_counts = df['complaint_type'].value_counts()

# Keep classes with at least 2 samples
valid_classes = class_counts[class_counts >= 2].index
df = df[df['complaint_type'].isin(valid_classes)]

print("Remaining classes:", len(valid_classes))

# -------------------------------
# 4. Preprocessing (NLTK)
# -------------------------------
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(f'[{re.escape(string.punctuation)}0-9]', '', text)
    
    words = word_tokenize(text)
    
    cleaned = [
        w for w in words
        if w.isalpha() and w not in stop_words
    ]
    
    return ' '.join(cleaned)

df['clean_text'] = df['text'].apply(preprocess_text)

# Remove empty processed rows
df = df[df['clean_text'].str.strip() != ""]

# -------------------------------
# 5. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'],
    df['complaint_type'],
    test_size=0.2,
    random_state=42,
    stratify=df['complaint_type']
)

# -------------------------------
# 6. TF-IDF Vectorization (Improved)
# -------------------------------
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),     # 🔥 big improvement
    min_df=2,               # remove rare words
    max_df=0.95             # remove very common words
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -------------------------------
# 7. Model Training (SVM)
# -------------------------------
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,        # number of trees
    max_depth=None,          # allow full growth
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,               # use all CPU cores
    random_state=42
)

model.fit(X_train_tfidf, y_train)

# -------------------------------
# 8. Evaluation
# -------------------------------
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# 9. Save Model + Vectorizer
# -------------------------------
joblib.dump(model, "random_forest.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\n✅ Model and vectorizer saved successfully!")