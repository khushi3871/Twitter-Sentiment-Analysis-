import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# -------------------------------
# Load dataset
# -------------------------------
file_path = r'D:\twitterSentimentAnalysis\training.1600000.processed.noemoticon.csv'
df = pd.read_csv(file_path, encoding='latin-1', header=None,
                 names=['sentiment', 'id', 'date', 'query', 'user', 'text'])
df = df[['sentiment', 'text']]
df['sentiment'] = df['sentiment'].replace({0: 'negative', 4: 'positive'})

# Optional: sample for faster training
df_sample = df.sample(100000, random_state=42)

# -------------------------------
# Preprocess tweets
# -------------------------------
def clean_tweet(tweet):
    tweet = str(tweet).lower()

    # Better negation handling
    negation_patterns = [
        r"not\s+(\w+)",
        r"never\s+(\w+)",
        r"no\s+(\w+)",
        r"don't\s+(\w+)",
        r"didn't\s+(\w+)",
        r"doesn't\s+(\w+)",
        r"cant\s+(\w+)",
        r"cannot\s+(\w+)"
    ]

    for pattern in negation_patterns:
        tweet = re.sub(pattern, r"not_\1", tweet)

    # Convert emoticons to words
    tweet = tweet.replace(" :(", " sad")
    tweet = tweet.replace(" :-(", " sad")
    tweet = tweet.replace(" :/", " bad")
    tweet = tweet.replace(" :|", " neutral")

    # remove urls
    tweet = re.sub(r"http\S+|www\S+", "", tweet)

    # remove mentions
    tweet = re.sub(r"@\w+", "", tweet)

    # remove hashtags
    tweet = tweet.replace("#", "")

    # remove punctuation
    tweet = re.sub(r"[^A-Za-z0-9\s]", "", tweet)

    # remove extra spaces
    tweet = re.sub(r"\s+", " ", tweet).strip()

    return tweet


df_sample['clean_text'] = df_sample['text'].apply(clean_tweet)
import matplotlib.pyplot as plt
import seaborn as sns

df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(8,5))
sns.histplot(df['text_length'], bins=50)
plt.title("Tweet Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()
plt.figure(figsize=(7,5))
sns.boxplot(x=df['sentiment'], y=df['text_length'])
plt.title("Tweet Length by Sentiment")
plt.show()
from collections import Counter
import itertools

words = list(itertools.chain(*df_sample['clean_text'].str.split()))
word_counts = Counter(words).most_common(30)

wc_df = pd.DataFrame(word_counts, columns=['word','count'])

plt.figure(figsize=(10,5))
sns.barplot(data=wc_df, x='count', y='word')
plt.title("Top 30 Most Common Words")
plt.show()

# -------------------------------
# Encode labels
# -------------------------------
le = LabelEncoder()
df_sample['label'] = le.fit_transform(df_sample['sentiment'])

# -------------------------------
# Split data
# -------------------------------
X = df_sample['clean_text']
y = df_sample['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# TF-IDF + Logistic Regression
# -------------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# -------------------------------
# Save model components
# -------------------------------
joblib.dump(model, "twitter_sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model, vectorizer, and label encoder saved successfully!")
