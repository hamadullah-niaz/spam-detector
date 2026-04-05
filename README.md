# Spam Detector Project

Detects SMS messages as Spam or Not Spam using Machine Learning (Logistic Regression / Naive Bayes) with a Flask web app.

---

## 📊 Accuracy

[Accuracy Screenshot](accuracy.png)

> Screenshot of model accuracy on test set

---

## 🛠 Technologies Used

- Python
- Pandas
- scikit-learn
- Flask
- Joblib (for saving model)
- HTML/CSS (templates)

---

## 📁 Project Structure


spam-detector/
│
├── train.py # ML training + model save
├── app.py # Flask web app
├── model.pkl # Saved model
├── vectorizer.pkl # Saved vectorizer
└── templates/
└── index.html # Front-end HTML
