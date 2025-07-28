from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import cohere
import joblib
import pickle
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pymongo import MongoClient
from collections import Counter


nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fallback_secret')


mongo_uri = os.getenv('MONGO_URI')
MONGODB_AVAILABLE = False
try:
    if mongo_uri:
        client = MongoClient(mongo_uri)
        db = client["mental_health_db"]
        results_collection = db["survey_results"]
        MONGODB_AVAILABLE = True
    else:
        print("⚠️ MONGO_URI not set in environment.")
except Exception as e:
    print(f"MongoDB connection error: {e}")


cohere_api_key = os.getenv('COHERE_API_KEY')
if not cohere_api_key:
    raise ValueError("⚠️ COHERE_API_KEY is not set.")
co = cohere.Client(cohere_api_key)


try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not installed; skipping tf_model loading.")


tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

questions = [
    {"type": "word", "text": "Alone →", "placeholder": "e.g., isolated, peaceful, scared"},
    {"type": "word", "text": "Failure →", "placeholder": "e.g., shame, learning, inevitable"},
    {"type": "word", "text": "Joy →", "placeholder": "e.g., laughter, rare, music"},
    {"type": "scenario", "text": "Describe your ideal day in 3 words →", "placeholder": "e.g., sunny, relaxed, fulfilled"},
    {"type": "scenario", "text": "Complete: 'When I wake up, I feel...' →", "placeholder": "e.g., tired but hopeful"},
    {"type": "scenario", "text": "What's something you've given up on? →", "placeholder": "e.g., trusting people"},
    {"type": "scenario", "text": "If emotions were weather, yours would be... →", "placeholder": "e.g., cloudy with lightning"},
    {"type": "scenario", "text": "What comforts you when you're low? →", "placeholder": "e.g., music, solitude"},
    {"type": "scenario", "text": "'I wish people knew...' →", "placeholder": "e.g., how hard I try"},
    {"type": "scenario", "text": "What does 'okay' look like for you? →", "placeholder": "e.g., getting out of bed"},
    {"type": "behavioral", "text": "Rate your energy today (1-10) →", "placeholder": "e.g., 3"},
    {"type": "behavioral", "text": "Do you cancel plans often? (Y/N) →", "placeholder": "e.g., Y"},
    {"type": "behavioral", "text": "Do you still enjoy old hobbies? (Y/N) →", "placeholder": "e.g., N"},
    {"type": "behavioral", "text": "When stressed, you: (1) isolate or (2) seek people →", "placeholder": "e.g., 1"},
    {"type": "behavioral", "text": "Do you have someone to talk to? (Y/N) →", "placeholder": "e.g., Y"}
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-input')
def text_input():
    return render_template('text_input.html')

@app.route('/predict-text', methods=['POST'])
def predict_text():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"})
    
    predictions = predict_all_models(text)
    final_result = majority_vote(predictions)
    
    return jsonify({
        "predictions": predictions,
        "final_result": final_result,
        "input_text": text
    })

@app.route('/questions')
def get_questions():
    return jsonify(questions)

@app.route('/submit', methods=['POST'])
def submit():
    responses = request.get_json()
    full_analysis = analyze_responses(responses)

    paragraphs = full_analysis.split('\n\n')
    second_paragraph = paragraphs[1] if len(paragraphs) > 1 else paragraphs[0]
    compressed_summary = compress_summary(second_paragraph)
    predictions = predict_all_models(compressed_summary)
    final_result = majority_vote(predictions)

    if MONGODB_AVAILABLE:
        try:
            results_collection.insert_one({
                "responses": responses,
                "analysis": full_analysis,
                "compressed_summary": compressed_summary,
                "predictions": predictions,
                "final_result": final_result
            })
        except Exception as e:
            print(f"MongoDB insert failed: {e}")

    session['analysis'] = full_analysis
    session['second_paragraph'] = compressed_summary
    session['predictions'] = predictions
    session['final_result'] = final_result

    return jsonify({"success": True})

@app.route('/results')
def results():
    analysis = session.get('analysis')
    second_paragraph = session.get('second_paragraph')
    predictions = session.get('predictions')
    final_result = session.get('final_result')

    if not analysis or not predictions:
        return redirect(url_for('home'))

    return render_template('results.html', analysis=analysis, second_paragraph=second_paragraph, predictions=predictions, final_result=final_result)



def analyze_responses(responses):
    try:
        prompt = f"""
        Analyze these survey responses from a user:
        {responses}

        Provide only a **concise psychological summary** of the user's 
        emotional and mental state in **one short paragraph** 
        (no headings, no bullet points, no numbered lists). The summary 
        should reflect emotional patterns, mindset, strengths, and concerns
        — all merged naturally in a narrative form.

        Then in a second paragraph, describe what kind of mental health condition the person may be showing signs of.
        """
        response = co.chat(
            model="command-r",
            message=prompt,
            temperature=0.7,
        )
        return response.text.strip()
    except Exception as e:
        print(f"Cohere API error: {e}")
        return "Unable to analyze responses at this time."

def compress_summary(paragraph):
    try:
        prompt = f"Compress this paragraph into a summary of 50-60 words:\n{paragraph}"
        response = co.chat(
            model="command-r",
            message=prompt,
            temperature=0.7,
        )
        return response.text.strip()
    except Exception as e:
        print(f"Cohere compression error: {e}")
        return paragraph[:100] + "..." if len(paragraph) > 100 else paragraph

def majority_vote(preds):
    labels = [v.lower().strip() for v in preds.values()]
    count = Counter(labels)
    if count["depression"] > count["non-depression"]:
        return "Depressed"
    return "Not Depressed"

def clean_text(text):
    text = str(text).lower()
    words = tokenizer.tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [ps.stem(word) for word in words]
    return ' '.join(words)

def predict_all_models(text):
    predictions = {}
    cleaned = clean_text(text)

    try:
        vectorizer = joblib.load('ss/tfidf.pkl')
        transformed = vectorizer.transform([cleaned]).toarray()

        label_encoder = joblib.load('ss/label_encoder.pkl')
        def decode_label(label):
            return label_encoder.inverse_transform([label])[0]

        models = {
            "Naive Bayes": joblib.load("ss/naive_bayes.pkl"),
            "Decision Tree": joblib.load("ss/decision_tree.pkl"),
            "Random Forest": joblib.load("ss/random_forest.pkl"),
            "XGBoost": joblib.load("ss/xgboost.pkl"),
            "Logistic Regression": joblib.load("ss/logistic_regression.pkl"),
        }

        for name, model in models.items():
            pred = model.predict(transformed)[0]
            predictions[name] = decode_label(pred)

        with open("ss/lightgbm.pkl", "rb") as f:
            lgbm_model = pickle.load(f)
        pred = lgbm_model.predict(transformed)[0]
        predictions["LightGBM"] = decode_label(pred)

        if TENSORFLOW_AVAILABLE:
            tf_model = load_model("ss/tf_model.keras")
            tf_pred = tf_model.predict(transformed)
            tf_label = np.argmax(tf_pred, axis=1)[0]
            predictions["TensorFlow"] = decode_label(tf_label)
        else:
            predictions["TensorFlow"] = "Not available"

    except Exception as e:
        print(f"Model prediction error: {e}")
    
    return predictions



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
