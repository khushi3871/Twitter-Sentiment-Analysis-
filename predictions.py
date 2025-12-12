import re
import joblib

# -------------------------------
# Load model, vectorizer, label encoder
# -------------------------------
model = joblib.load('twitter_sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

# -------------------------------
# Tweet cleaning function
# -------------------------------
def clean_tweet(tweet):
    tweet = str(tweet).lower()
    tweet = re.sub(r'http\S+|www\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#','', tweet)
    tweet = re.sub(r'[^A-Za-z0-9\s]','', tweet)
    tweet = re.sub(r'\s+',' ', tweet).strip()
    return tweet

# -------------------------------
# Predict function
# -------------------------------
def predict_sentiment(tweet):
    tweet_clean = clean_tweet(tweet)
    vector = vectorizer.transform([tweet_clean])
    pred = model.predict(vector)
    return le.inverse_transform(pred)[0]

# -------------------------------
# Example
# -------------------------------
print(predict_sentiment("I love this new phone!"))
print(predict_sentiment("This is the worst experience ever."))
print(predict_sentiment("I am not happy."))
