from flask import Flask, render_template, request
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords (runs only once)
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load trained model and TF-IDF vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
print(type(model))
print(model)

# Initialize stopwords and stemmer
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


# Text preprocessing function
def clean_text(text):
    text = text.lower()

    # Remove text inside square brackets
    text = re.sub(r'\[.*?\]', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    # Remove numbers
    text = re.sub(r'\w*\d\w*', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]

    # Stemming
    words = [stemmer.stem(word) for word in words]

    return " ".join(words)


# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    # Preprocess text
    cleaned_news = clean_text(news)
    print("=" * 60)
    print("Original News:")
    print(news)

    print("\nCleaned News:")
    print(cleaned_news)

    # Convert text into TF-IDF features
    news_vector = vectorizer.transform([cleaned_news])
    print("Vector Shape:", news_vector.shape)

    # Predict
    prediction = model.predict(news_vector)
    print("Prediction:", prediction)
    print("=" * 60)
    print("Prediction:", prediction)

    if prediction[0] == 0:
       result = "❌ FAKE NEWS DETECTED"
    else:
       result = "✅ VERIFIED REAL NEWS"

    return render_template(
    "index.html",
    prediction=result,
    news=news
)

if __name__ == "__main__":
    app.run(debug=True)