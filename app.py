from flask import Flask, request, jsonify, render_template
import joblib
import re

app = Flask(__name__)

# Load components
model = joblib.load('twitter_sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

# EXACT SAME CLEANING AS train.py
def clean_tweet(tweet):
    tweet = str(tweet).lower()

    # Negation handling (VERY IMPORTANT)
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

    # Emoticons
    tweet = tweet.replace(":(", " sad")
    tweet = tweet.replace(":-(", " sad")
    tweet = tweet.replace(":)", " happy")
    tweet = tweet.replace(":|", " neutral")

    # Remove noise
    tweet = re.sub(r"http\S+|www\S+", "", tweet)
    tweet = re.sub(r"@\w+", "", tweet)
    tweet = tweet.replace("#", "")
    tweet = re.sub(r"[^A-Za-z0-9_\s]", "", tweet)
    tweet = re.sub(r"\s+", " ", tweet).strip()

    return tweet

def predict_sentiment(tweet):
    cleaned = clean_tweet(tweet)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)
    return le.inverse_transform(pred)[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    tweet = data['tweet']
    result = predict_sentiment(tweet)
    return jsonify({'sentiment': result})

if __name__ == "__main__":
    app.run(debug=True)
