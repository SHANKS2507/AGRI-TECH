# 🌿 Smart Agriculture AI

An AI-powered crop recommendation system using Random Forest ML — built with Flask, Python, and a premium dark organic UI.

---

## Project Structure

```
smart_agriculture/
├── app.py                        ← Flask app (routes + profit logic)
├── model.py                      ← ML training & prediction
├── Crop_recommendation.csv       ← Dataset (you provide this)
├── crop_model.pkl                ← Auto-generated after first run
├── label_encoder.pkl             ← Auto-generated after first run
├── requirements.txt
├── templates/
│   ├── index.html                ← Input form page
│   └── result.html               ← Prediction results page
└── static/
    └── css/
        └── style.css             ← Full custom styling
```

---

## Setup & Run

### 1. Get the Dataset

Download `Crop_recommendation.csv` from Kaggle:
👉 https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

Place it in the project root (same folder as `app.py`).

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

> The model will train automatically on first run and save `crop_model.pkl`.

---

## Re-train Model

Visit `http://localhost:5000/train` to retrain and see accuracy + classification report.

---

## Features

- **RandomForestClassifier** (200 estimators) with ~99% accuracy on test set
- **Top 3 crop predictions** with probability scores
- **Profit analysis** using market price × yield − production cost
- **Most profitable crop** recommendation among top 3
- **Interactive sliders** with live number sync
- **Form validation** + loading spinner
- **Dark organic UI** — Playfair Display + DM Sans typography

---

## Dataset Features

| Feature     | Description                  | Range     |
|-------------|------------------------------|-----------|
| N           | Nitrogen content (mg/kg)     | 0–140     |
| P           | Phosphorus content (mg/kg)   | 5–145     |
| K           | Potassium content (mg/kg)    | 5–205     |
| temperature | Temperature (°C)             | 0–50      |
| humidity    | Relative humidity (%)        | 14–100    |
| ph          | Soil pH                      | 3.5–9.5   |
| rainfall    | Annual rainfall (mm)         | 20–300    |

---

## Supported Crops

Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee.
