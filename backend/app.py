# app.py
from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import traceback

app = Flask(__name__)

# -------------------------
# Load model + data safely
# -------------------------
print("📦 Loading model + data...")

model = pickle.load(open("model/disease_model.pkl", "rb"))
vectorizer = pickle.load(open("model/symptom_vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))

# CSV encodings (latin1 for doctor map due to common Windows-1252 saves)
disease_desc = pd.read_csv("data/Disease_Description.csv", encoding="utf-8")
doctor_vs_disease = pd.read_csv("data/Doctor_Versus_Disease.csv", encoding="latin1")

# Clean column headers (strip spaces) and auto-detect relevant columns
disease_desc.columns = disease_desc.columns.str.strip()
doctor_vs_disease.columns = doctor_vs_disease.columns.str.strip()

# For Disease_Description.csv we expect: "Disease", "Description"
desc_disease_col = None
desc_text_col = None
for c in disease_desc.columns:
    low = c.lower()
    if desc_disease_col is None and "disease" in low:
        desc_disease_col = c
    if desc_text_col is None and ("description" in low or "desc" in low):
        desc_text_col = c
# Fallbacks
desc_disease_col = desc_disease_col or "Disease"
desc_text_col = desc_text_col or "Description"

# For Doctor_Versus_Disease.csv we expect: "Disease", "Doctor"/"Specialist"
doc_disease_col = None
doc_specialist_col = None
for c in doctor_vs_disease.columns:
    low = c.lower()
    if doc_disease_col is None and "disease" in low:
        doc_disease_col = c
    if doc_specialist_col is None and ("doctor" in low or "specialist" in low):
        doc_specialist_col = c
# Fallbacks
doc_disease_col = doc_disease_col or "Disease"
doc_specialist_col = doc_specialist_col or "Doctor"

print(f"✅ Using columns — Desc: ({desc_disease_col}, {desc_text_col}) | DoctorMap: ({doc_disease_col}, {doc_specialist_col})")
print("✅ Model + data loaded!")

# -------------------------
# Helpers
# -------------------------
def lookup_description(disease_name: str) -> str:
    try:
        m = disease_desc[ disease_desc[desc_disease_col].str.lower() == disease_name.lower() ]
        if not m.empty:
            return str(m.iloc[0][desc_text_col])
    except Exception:
        pass
    return "No description available."

def lookup_specialist(disease_name: str) -> str:
    try:
        m = doctor_vs_disease[ doctor_vs_disease[doc_disease_col].str.lower() == disease_name.lower() ]
        if not m.empty:
            return str(m.iloc[0][doc_specialist_col])
    except Exception:
        pass
    return "No specialist found."

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        symptoms = data.get("symptoms", [])
        print("📩 Received symptoms:", symptoms)

        if not symptoms or not isinstance(symptoms, list):
            return jsonify({"error": "Invalid symptoms input."}), 400

        # Join and clean symptoms into the exact format the vectorizer expects
        input_text = " ".join([s.strip().lower() for s in symptoms if s.strip()])
        if not input_text:
            return jsonify({"error": "No symptoms provided."}), 400

        # Vectorize and get probabilities
        X = vectorizer.transform([input_text]).toarray()
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
        else:
            # fallback: 1.0 for predicted class, 0 for others (NB has predict_proba though)
            pred_idx = model.predict(X)[0]
            probs = np.zeros(len(label_encoder.classes_), dtype=float)
            probs[pred_idx] = 1.0

        # Top-N predictions
        top_n = 3
        top_idx = np.argsort(probs)[::-1][:top_n]

        result_rows = []
        for idx in top_idx:
            disease_name = label_encoder.inverse_transform([idx])[0]
            chance = round(float(probs[idx]) * 100.0, 6)  # high precision, you can round later in UI
            specialist = lookup_specialist(disease_name)
            description = lookup_description(disease_name)
            result_rows.append({
                "Disease": disease_name,
                "Chances": chance,
                "Specialist": specialist,
                "Description": description
            })

        print("🎯 Top predictions:", result_rows)
        return jsonify({"predictions": result_rows})

    except Exception as e:
        print("❌ Prediction Error:", e)
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("🚀 Running on http://127.0.0.1:5000")
    app.run(debug=True)
