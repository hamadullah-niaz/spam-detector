
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
import joblib


        #### Load dataset

df = pd.read_csv(r"C:\Users\shubi\Desktop\myeea\spam.csv.csv", encoding='latin_1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']


        #### Clean

df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df.dropna(inplace=True)


        #### Train/Test split

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)


        #### Vectorize (TF-IDF)

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)  # fit_transform here
X_test_vec = vectorizer.transform(X_test)        # only transform here


       #### Naive Bayes

nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_pred = nb_model.predict(X_test_vec)


        #### Logistic Regression

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
    )
lr_model.fit(X_train_vec, y_train)
lr_pred = lr_model.predict(X_test_vec)


        #### Evaluate

print("=== Naive Bayes ===")
print(f"Accuracy: {accuracy_score(y_test, nb_pred):.4f}")
print(classification_report(y_test, nb_pred, target_names=['Ham', 'Spam']))

print("=== Logistic Regression ===")
print(f"Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
print(classification_report(y_test, lr_pred, target_names=['Ham', 'Spam']))


        #### Save model  #Model.pkl  # Vectorizer.pkl 

joblib.dump(nb_model, r'C:\Users\shubi\Desktop\myeea\kmdir spam-detector\model.pkl')
joblib.dump(vectorizer, r'C:\Users\shubi\Desktop\myeea\kmdir spam-detector\vectorizer.pkl')
print("\nModel saved!")



#         #### App

from flask import Flask, render_template, request
import joblib
import re, string


app = Flask(__name__)

model = joblib.load(r'C:\Users\shubi\Desktop\myeea\kmdir spam-detector\model.pkl')
vectorizer = joblib.load(r'C:\Users\shubi\Desktop\myeea\kmdir spam-detector\vectorizer.pkl')


def clean_text(text):
    text =text.lower()
    text =re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = clean_text(request.form['message'])

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    prob = model.predict_proba(text_vec)[0][1]

    result = "Spam" if prediction == 1 else "Not Spam"
    

    return render_template(
        'index.html',
        prediction_text = result,
        probability= round(prob*100, 2)
        )


if __name__ =="__main__":
    app.run(debug=True)


