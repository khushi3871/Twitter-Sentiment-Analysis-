import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -------------------------------
# Load dataset
# -------------------------------
file_path = r'D:\twitterSentimentAnalysis\training.1600000.processed.noemoticon.csv'

df = pd.read_csv(
    file_path,
    encoding='latin-1',
    header=None,
    names=['sentiment', 'id', 'date', 'query', 'user', 'text']
)

df = df[['sentiment', 'text']]
df['sentiment'] = df['sentiment'].replace({0: 'negative', 4: 'positive'})

# Sample for faster training
df = df.sample(100000, random_state=42)

# -------------------------------
# Text Cleaning with Negation Handling
# -------------------------------
def clean_tweet(tweet):
    tweet = str(tweet).lower()

    tweet = re.sub(r"http\S+|www\S+", "", tweet)
    tweet = re.sub(r"@\w+", "", tweet)
    tweet = tweet.replace("#", "")

    tokens = tweet.split()
    negation_words = {"not", "no", "never", "dont", "didnt", "cant", "cannot"}

    result = []
    negate = False

    for word in tokens:
        word = re.sub(r"[^a-z0-9]", "", word)

        if word in negation_words:
            negate = True
            continue

        if negate:
            result.append("NEG_" + word)
            negate = False
        else:
            result.append(word)

    return " ".join(result)

df['clean_text'] = df['text'].apply(clean_tweet)

# -------------------------------
# Encode labels
# -------------------------------
le = LabelEncoder()
df['label'] = le.fit_transform(df['sentiment'])

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'],
    df['label'],
    test_size=0.2,
    random_state=42
)

# -------------------------------
# Pipeline
# -------------------------------
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=2000))
])

# -------------------------------
# Hyperparameter Grid
# -------------------------------
param_grid = {
    'tfidf__max_features': [10000, 15000],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'tfidf__sublinear_tf': [True],
    'clf__C': [0.5, 1, 2],
    'clf__class_weight': ['balanced']
}

# -------------------------------
# Grid Search
# -------------------------------
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)

# -------------------------------
# Evaluation
# -------------------------------
y_pred = best_model.predict(X_test)

print("\nModel Performance:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# Save Model Components
# -------------------------------
joblib.dump(best_model.named_steps['clf'], "twitter_sentiment_model.pkl")
joblib.dump(best_model.named_steps['tfidf'], "tfidf_vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\nBest model saved successfully!")
