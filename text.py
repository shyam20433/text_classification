import joblib
import pickle
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model
from collections import Counter

nltk.download('stopwords')

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = str(text).lower()
    words = tokenizer.tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [ps.stem(word) for word in words]
    return ' '.join(words)

def predict_all_models(text):
    predictions = {}

    cleaned = clean_text(text)
    
    vectorizer = joblib.load('ss/tfidf.pkl')
    transformed = vectorizer.transform([cleaned]).toarray()

    label_encoder = joblib.load('ss/label_encoder.pkl')
    def decode_label(label):
        return label_encoder.inverse_transform([label])[0]

    nb_model = joblib.load("ss/naive_bayes.pkl")
    pred = nb_model.predict(transformed)[0]
    predictions["Naive Bayes"] = decode_label(pred)

    dt_model = joblib.load("ss/decision_tree.pkl")
    pred = dt_model.predict(transformed)[0]
    predictions["Decision Tree"] = decode_label(pred)

    rf_model = joblib.load("ss/random_forest.pkl")
    pred = rf_model.predict(transformed)[0]
    predictions["Random Forest"] = decode_label(pred)

    xgb_model = joblib.load("ss/xgboost.pkl")
    pred = xgb_model.predict(transformed)[0]
    predictions["XGBoost"] = decode_label(pred)

    lr_model = joblib.load("ss/logistic_regression.pkl")
    pred = lr_model.predict(transformed)[0]
    predictions["Logistic Regression"] = decode_label(pred)

    with open("ss/lightgbm.pkl", "rb") as f:
        lgbm_model = pickle.load(f)
    pred = lgbm_model.predict(transformed)[0]
    predictions["LightGBM"] = decode_label(pred)

    tf_model = load_model("ss/tf_model.keras")
    tf_pred = tf_model.predict(transformed)
    tf_pred_label = np.argmax(tf_pred, axis=1)[0]
    predictions["TensorFlow"] = decode_label(tf_pred_label)

    return predictions

def majority_vote(preds):
    labels = [v.lower().strip() for v in preds.values()]
    count = Counter(labels)
    print(f"Individual model predictions: {preds}")
    print(f"Labels after processing: {labels}")
    print(f"Count of each label: {count}")
    if count["depression"] > count["non-depression"]:
        result = "Depressed"
    else:
        result = "Not Depressed"
    print(f"Final result: {result}")
    return result

if __name__ == "__main__":
    print("Enter text to check for depression (Ctrl+C to quit):")
    while True:
        try:
            input_text = input("\nText:\n")
            results = predict_all_models(input_text)
            print("\nPredictions from all models:\n")
            for model, result in results.items():
                print(f"{model}: {result}")
            
            final_result = majority_vote(results)
            print(f"\n🎯 Final Result: {final_result}")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
