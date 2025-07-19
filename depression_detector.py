import numpy as np
import pandas as pd
import joblib
import pickle
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from lightgbm import LGBMClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

nltk.download('stopwords')


df = pd.read_csv("Suicide_Detection.csv")
df['class'] = df['class'].replace({'non-suicide': 'non-depression', 'suicide': 'depression'})

if len(df) > 60000:
    df = df.sample(n=60000, random_state=42)
    print("Sampled 60000 rows")
else:
    df = df.sample(frac=1.0, random_state=42)
    print(f"Using full dataset of {len(df)} rows")

if 'Unnamed: 0' in df.columns:
    df.drop(columns='Unnamed: 0', inplace=True)


tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = str(text).lower()
    words = tokenizer.tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [ps.stem(word) for word in words]
    return ' '.join(words)

df['text'] = df['text'].apply(clean_text)


texts = df['text']
y = df['class']
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, 'label_encoder.pkl')

X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, y_encoded, test_size=0.20, random_state=42
)

vectorizer = TfidfVectorizer(min_df=50, max_features=5000)
X_train = vectorizer.fit_transform(X_train_text).toarray()
X_test = vectorizer.transform(X_test_text).toarray()
joblib.dump(vectorizer, 'tfidf.pkl')


results = {}

nb = GaussianNB()
nb2 = BernoulliNB()
nb3 = MultinomialNB()
NaiveBayes = VotingClassifier(
    estimators=[('GaussianNB', nb), ('BernoulliNB', nb2), ('MultinomialNB', nb3)],
    voting='soft'
)
NaiveBayes.fit(X_train, y_train)
joblib.dump(NaiveBayes, "naive_bayes.pkl")
print("Saved Naive Bayes")

DecisionTree = DecisionTreeClassifier(max_depth=4, random_state=42)
DecisionTree.fit(X_train, y_train)
joblib.dump(DecisionTree, "decision_tree.pkl")
print("Saved Decision Tree")

RandomForest = RandomizedSearchCV(
    RandomForestClassifier(),
    {
        'n_estimators': [4, 5],
        'criterion': ['entropy'],
        'max_depth': range(1, 4),
        'min_samples_split': range(2, 5)
    },
    random_state=12
)
RandomForest.fit(X_train, y_train)
joblib.dump(RandomForest, "random_forest.pkl")
print("Saved Random Forest")


import xgboost as xgb
xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, "xgboost.pkl")
print("Saved XGBoost")


lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
joblib.dump(lr_model, "logistic_regression.pkl")
results["LogisticRegression"] = lr_model.score(X_test, y_test)
print("Logistic Regression Accuracy:", results["LogisticRegression"])

lgbm_model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    class_weight='balanced',
    random_state=42
)
lgbm_model.fit(X_train, y_train)
results["LightGBM"] = lgbm_model.score(X_test, y_test)
with open("lightgbm.pkl", "wb") as f:
    pickle.dump(lgbm_model, f)
print("LightGBM Accuracy:", results["LightGBM"])


y_train_keras = to_categorical(y_train)
y_test_keras = to_categorical(y_test)

tf_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y_train)), activation='softmax')
])
tf_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5)

history = tf_model.fit(
    X_train, y_train_keras,
    validation_data=(X_test, y_test_keras),
    epochs=20,
    batch_size=256,
    callbacks=[early_stop, reducelr]
)
tf_model.save("tf_model.keras")
print("Saved TensorFlow model")

print("Training complete. All models and vectorizer saved.")
