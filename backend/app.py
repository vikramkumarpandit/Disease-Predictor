# app.py
from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import traceback
import re
import difflib
import unicodedata
import os

app = Flask(__name__)

# ------------------------------------------------------------
# Load model and data
# ------------------------------------------------------------
print("📦 Loading model and data...")

# Load trained model files
model = pickle.load(open("model/disease_model.pkl", "rb"))
vectorizer = pickle.load(open("model/symptom_vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))

# -------------------------
# Load CSV files
# -------------------------
def load_csv_with_header_fallback(path, default_cols, encoding="utf-8"):
    """Loads a CSV file, adding default column headers if none exist."""
    try:
        df = pd.read_csv(path, encoding=encoding)
        # If 'Disease' not in columns → assume header missing
        if not any("disease" in c.lower() for c in df.columns):
            print(f"⚠️ No valid header detected in {os.path.basename(path)} — adding default columns {default_cols}")
            df = pd.read_csv(path, names=default_cols, encoding=encoding)
        return df
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return pd.DataFrame(columns=default_cols)

# Load disease description and doctor mapping CSVs
disease_desc = load_csv_with_header_fallback("data/Disease_Description.csv", ["Disease", "Description"], encoding="utf-8")
doctor_vs_disease = load_csv_with_header_fallback("data/Doctor_Versus_Disease.csv", ["Disease", "Doctor"], encoding="latin1")

# Clean column headers (strip spaces, remove BOMs)
disease_desc.columns = disease_desc.columns.str.strip().str.replace('\ufeff', '', regex=False)
doctor_vs_disease.columns = doctor_vs_disease.columns.str.strip().str.replace('\ufeff', '', regex=False)

# Confirm which columns are being used
desc_disease_col = next((c for c in disease_desc.columns if "disease" in c.lower()), "Disease")
desc_text_col = next((c for c in disease_desc.columns if "desc" in c.lower()), "Description")
doc_disease_col = next((c for c in doctor_vs_disease.columns if "disease" in c.lower()), "Disease")
doc_specialist_col = next((c for c in doctor_vs_disease.columns if "doctor" in c.lower() or "specialist" in c.lower()), "Doctor")

print(f"✅ Using columns — Description: ({desc_disease_col}, {desc_text_col}) | Doctor Mapping: ({doc_disease_col}, {doc_specialist_col})")

# ------------------------------------------------------------
# Normalization and Helper Functions
# ------------------------------------------------------------
def normalize_text(s: str) -> str:
    """Cleans and normalizes text for matching."""
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Build normalized lookup dictionaries
desc_map = {
    normalize_text(row[desc_disease_col]): str(row[desc_text_col]).strip()
    for _, row in disease_desc.iterrows()
    if isinstance(row[desc_disease_col], str)
}

spec_map = {
    normalize_text(row[doc_disease_col]): str(row[doc_specialist_col]).strip()
    for _, row in doctor_vs_disease.iterrows()
    if isinstance(row[doc_disease_col], str)
}

_spec_keys = list(spec_map.keys())
_desc_keys = list(desc_map.keys())

# Fuzzy match lookup
def fuzzy_lookup(name: str, mapping: dict, keys: list, cutoff=0.8):
    key = normalize_text(name)
    if key in mapping:
        return mapping[key]
    matches = difflib.get_close_matches(key, keys, n=1, cutoff=cutoff)
    if matches:
        return mapping.get(matches[0])
    return None

def lookup_specialist(disease_name: str) -> str:
    val = fuzzy_lookup(disease_name, spec_map, _spec_keys)
    return val if val and val.strip() else "No specialist found."

def lookup_description(disease_name: str) -> str:
    val = fuzzy_lookup(disease_name, desc_map, _desc_keys)
    return val if val and val.strip() else "No description available."

print("✅ Model and CSV data loaded successfully!")

# ------------------------------------------------------------
# Flask Routes
# ------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        symptoms = data.get("symptoms", [])
        print(f"📩 Received symptoms: {symptoms}")

        if not symptoms or not isinstance(symptoms, list):
            return jsonify({"error": "Invalid symptoms input."}), 400

        # Join and vectorize
        input_text = " ".join([s.strip().lower() for s in symptoms if s.strip()])
        if not input_text:
            return jsonify({"error": "No symptoms provided."}), 400

        X = vectorizer.transform([input_text]).toarray()

        # Predict
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
        else:
            pred_idx = model.predict(X)[0]
            probs = np.zeros(len(label_encoder.classes_))
            probs[pred_idx] = 1.0

        # Top 3 predictions
        top_idx = np.argsort(probs)[::-1][:3]
        result_rows = []

        for idx in top_idx:
            disease_name = label_encoder.inverse_transform([idx])[0]
            chance = round(float(probs[idx]) * 100.0, 2)
            specialist = lookup_specialist(disease_name)
            description = lookup_description(disease_name)

            print(f"🩺 {disease_name:<30} | Specialist: {specialist}")

            result_rows.append({
                "Disease": disease_name,
                "Chances": chance,
                "Specialist": specialist,
                "Description": description
            })

        print("🎯 Predictions:", result_rows)
        return jsonify({"predictions": result_rows})

    except Exception as e:
        print("❌ Prediction Error:", e)
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("🚀 Running Flask app at http://127.0.0.1:5000")
    app.run(debug=True)
