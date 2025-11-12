# model.py
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

print("📂 Loading dataset...")
df = pd.read_csv("data/Original_Dataset.csv", encoding="utf-8")

# Combine symptoms columns -> single text field (space-separated, like training)
df["combined_symptoms"] = df.iloc[:, 1:].apply(
    lambda x: " ".join(x.dropna().astype(str)).strip().lower(),
    axis=1
)

vectorizer = CountVectorizer()
label_encoder = LabelEncoder()

X = vectorizer.fit_transform(df["combined_symptoms"]).toarray()
y = label_encoder.fit_transform(df["Disease"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("⚙️ Training Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Model trained successfully — Accuracy: {acc*100:.2f}%")

os.makedirs("model", exist_ok=True)
with open("model/disease_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/symptom_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("💾 Saved: model/disease_model.pkl, model/symptom_vectorizer.pkl, model/label_encoder.pkl")
