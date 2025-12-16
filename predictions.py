import re
import joblib

# -------------------------------
# Load saved components
# -------------------------------
model = joblib.load('twitter_sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

# -------------------------------
# SAME cleaning logic (VERY IMPORTANT)
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

# -------------------------------
# Predict Function
# -------------------------------
def predict_sentiment(tweet):
    cleaned = clean_tweet(tweet)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)
    return le.inverse_transform(pred)[0]

# -------------------------------
# Examples
# -------------------------------
print(predict_sentiment("I love this new phone"))
print(predict_sentiment("This is the worst experience ever"))
print(predict_sentiment("I am not happy"))
print(predict_sentiment("not good at all"))
