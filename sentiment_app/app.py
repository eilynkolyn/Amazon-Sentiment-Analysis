import nltk
import re
import pickle
import tensorflow as tf

from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords




# Load model and tokenizer (you should save/load tokenizer via pickle ideally)
model = load_model("../model/best_model.h5")

# Reinitializing tokenizer to match training
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

app = Flask(__name__)

# Text cleaning function (same as used during training)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    cleaned = clean_text(review)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
    prediction = model.predict(padded)[0][0]
    sentiment = "Positive Review" if prediction > 0.5 else "Negative Review"
    confidence = f"{prediction:.2f}"
    return render_template('index.html', review=review, sentiment=sentiment, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
