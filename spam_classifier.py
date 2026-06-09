# ============================================================
# EMAIL SPAM CLASSIFIER
# Uses NLP + Machine Learning to detect spam emails
# Algorithms: Naive Bayes, Logistic Regression, SVM
# ============================================================

# Step 1: Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import re

print("=" * 50)
print("   EMAIL SPAM CLASSIFIER")
print("=" * 50)

# Step 2: Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

print(f"\n✅ Dataset loaded: {len(df)} messages")

# Step 3: Clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["clean_message"] = df["message"].apply(clean_text)
df["label_num"] = df["label"].map({"spam": 1, "ham": 0})

# Step 4: Split dataset
X = df["clean_message"]
y = df["label_num"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Convert text to numbers
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train 3 models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

results = {}
print("\n" + "=" * 50)
print("   MODEL RESULTS")
print("=" * 50)

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {"model": model, "accuracy": acc}
    print(f"\n📊 {name}: {acc * 100:.2f}% accuracy")

# Step 7: Best model
best_name = max(results, key=lambda x: results[x]["accuracy"])
best_model = results[best_name]["model"]
print(f"\n🏆 Best Model: {best_name}")

# Step 8: Test messages
def predict_message(message):
    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    prediction = best_model.predict(vectorized)[0]
    return "🚨 SPAM" if prediction == 1 else "✅ HAM (Not Spam)"

print("\n--- Sample Tests ---")
test_messages = [
    "Congratulations! You won a FREE iPhone!",
    "Hey, are we meeting tomorrow?",
    "URGENT: Your account is suspended!"
]
for msg in test_messages:
    print(f"\nMessage: {msg}")
    print(f"Result : {predict_message(msg)}")

# Step 9: Try your own message
print("\n" + "=" * 50)
while True:
    user_input = input("\nEnter a message (or 'quit' to exit):\n> ")
    if user_input.lower() == "quit":
        print("Goodbye! 👋")
        break
    print(f"Prediction: {predict_message(user_input)}")
