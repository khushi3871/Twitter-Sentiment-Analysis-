from flask import Flask, request, jsonify, render_template
import joblib
import re

app = Flask(__name__)

# Load components
model = joblib.load('twitter_sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

# Cleaning function
def clean_tweet(tweet):
    tweet = str(tweet).lower()
    tweet = re.sub(r"http\S+|www\S+", "", tweet)
    tweet = re.sub(r"@\w+", "", tweet)
    tweet = re.sub(r"#","", tweet)
    tweet = re.sub(r"[^A-Za-z0-9\s]","", tweet)
    tweet = re.sub(r"\s+"," ", tweet).strip()
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
