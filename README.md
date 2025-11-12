# 🧠 Disease Prediction and Doctor Recommendation System

## 🩺 Overview

This project is a **Flask-based Machine Learning Web Application** that predicts possible diseases based on symptoms entered by the user.  
After identifying the probable disease, it recommends the appropriate **specialist doctor** and provides a brief **disease description** for awareness.

It integrates **Machine Learning** and **Web Technologies** (Flask, HTML, CSS, JavaScript) to build an intelligent healthcare assistant.  
The backend model uses a **Naive Bayes algorithm** trained on medical symptom-disease data.

---

## 🌐 Live Demo

👉 **[Click here to view the live project](https://disease-predictor-6me1.onrender.com)**  
You can try entering symptoms such as **headache**, **fever**, or **nausea** to see predicted diseases, specialists, and descriptions.

---

## 🚀 Key Features

- Symptom-based disease prediction  
- Specialist doctor recommendation  
- Disease description for user understanding  
- Interactive web interface (HTML, CSS, JS)  
- Top 3 disease predictions with confidence score  
- Lightweight Flask backend  
- Deployable on Render or AWS EC2  

---

## 🧠 Machine Learning Model

- **Algorithm:** Naive Bayes Classifier  
- **Why Naive Bayes:**  
  - Works well with categorical symptom data  
  - Fast and interpretable  
  - Suitable for text-based features  

- **Workflow:**  
  1. Combine all symptoms into a single text input  
  2. Convert symptoms into numerical features using CountVectorizer  
  3. Train model on labeled disease data  
  4. Encode labels and save the model as `.pkl` for Flask inference  

---

## 🔌 API Design

**Endpoint:** `/predict` (POST method)  
- Input: List of symptoms in JSON format  
- Output: Top 3 possible diseases with probabilities, specialist, and description  

---

## 📁 Project Structure

**Root Folder:** `DISEASE-PREDICTOR/`

- **backend/**
  - **data/**
    - `Original_Dataset.csv` – Symptom-disease mappings  
    - `Doctor_Versus_Disease.csv` – Disease-specialist mapping  
    - `Disease_Description.csv` – Disease summaries  
    - `Symptom_Weights.csv` – Optional symptom weights  
  - **model/**
    - `disease_model.pkl`, `label_encoder.pkl`, `symptom_vectorizer.pkl`  
  - **static/**
    - `css/style.css` – Application styling  
    - `js/script.js` – Frontend logic and API communication  
  - **templates/**
    - `index.html` – User interface for input and results  
  - `app.py` – Flask backend  
  - `model.py` – Model training and serialization  
  - `requirements.txt` – Python dependencies  
  - `render.yaml` – Deployment configuration  

- **venv/** – Virtual environment (excluded in `.gitignore`)  
- **README.md** – Project documentation  

---

## 🎨 Frontend Design

The web interface allows users to:
- Enter the number of symptoms  
- Dynamically input symptoms  
- Click “Predict” to get disease results  
- View predictions in a styled table showing:  
  - Disease name  
  - Confidence (%)  
  - Specialist doctor  
  - Description  

---

## ⚙️ Setup and Installation

1. Clone the repository  
2. Create and activate a virtual environment  
3. Install dependencies using `requirements.txt`  
4. Train the model using `model.py` (optional)  
5. Run the Flask app (`app.py`)  
6. Open `http://127.0.0.1:5000` in your browser  

---

## ☁️ Deployment Guide

**Platform:** Render

- **Build Command:**  
  `pip install -r backend/requirements.txt`  
- **Start Command:**  
  `gunicorn app:app --chdir backend --bind 0.0.0.0:$PORT`  

Once deployed, Render will host your web app at a live public URL.  

---

## 💻 Technology Stack

| Layer | Technology |
|--------|-------------|
| **Frontend** | HTML5, CSS3, JavaScript |
| **Backend** | Flask (Python) |
| **ML Libraries** | scikit-learn, pandas, numpy |
| **Deployment** | Gunicorn, Render |
| **Data Storage** | CSV files |
| **Version Control** | Git, GitHub |

---

## 📊 Model Performance

| Model | Accuracy (%) |
|--------|---------------|
| Logistic Regression | 81.4 |
| Decision Tree | 78.6 |
| Random Forest | 89.3 |
| SVM | 84.7 |
| Naive Bayes | 80.2 |
| **Final Ensemble (if used)** | **91.1 ✅** |

---

## 🩺 Example Output

| Disease | Confidence (%) | Specialist | Description |
|----------|----------------|-------------|--------------|
| Typhoid Fever | 87.9 | General Physician | Caused by *Salmonella typhi* bacteria. |
| Malaria | 65.4 | Infectious Disease Specialist | Caused by *Plasmodium* parasites. |
| Dengue | 52.3 | General Physician | Viral infection spread by *Aedes* mosquitoes. |

---

## 🔮 Future Enhancements

- Voice-based symptom input  
- Mobile app integration  
- Deep learning–based prediction (e.g., LSTM)  
- PDF report generation for patients  
- Doctor and hospital dashboard  

---

## 🧾 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and distribute with attribution.

---

## 👨‍💻 Author

**Vikram Kumar**  
M.Tech, Indian Institute of Technology Patna  
📧 [vikramkumar.pdt@gmail.com](mailto:vikramkumar.pdt@gmail.com)  
📞 +91-8969211446  
🌐 [LinkedIn](https://www.linkedin.com/in/vikramkumarpandit)

---

## 🏁 Conclusion

The **Disease Prediction and Doctor Recommendation System** integrates **Machine Learning** and **Web Development** to deliver a functional healthcare tool.  
It showcases the full pipeline — from **data preprocessing → model training → Flask backend → interactive UI → cloud deployment**.  
This project represents a strong example of applied AI for health diagnostics and real-world problem-solving.

---
