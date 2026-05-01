"""
app.py - Smart Agriculture AI
Flask application: routing, form handling, prediction orchestration.
"""

from flask import Flask, render_template, request, redirect, url_for, session
import os
from model import predict_crop

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ── Crop profit dictionary ────────────────────────────────────────────────────
# Price in INR per 100 kg (quintal), estimated market values (1 USD ≈ 83 INR)
CROP_PRICES = {
    "rice":        2320,
    "maize":       1494,
    "chickpea":    4565,
    "kidneybeans": 4980,
    "pigeonpeas":  3984,
    "mothbeans":   4316,
    "mungbean":    4814,
    "blackgram":   4482,
    "lentil":      5395,
    "pomegranate": 7470,
    "banana":      2905,
    "mango":       6640,
    "grapes":      7885,
    "watermelon":  1660,
    "muskmelon":   2075,
    "apple":       9130,
    "orange":      3735,
    "papaya":      2490,
    "coconut":     6225,
    "cotton":      5395,
    "jute":        1826,
    "coffee":      16600,
}

# Assumed yield (quintals per hectare) per crop
CROP_YIELD = {
    "rice":        30.0,
    "maize":       35.0,
    "chickpea":    12.0,
    "kidneybeans": 10.0,
    "pigeonpeas":  10.0,
    "mothbeans":    8.0,
    "mungbean":     9.0,
    "blackgram":    8.0,
    "lentil":      11.0,
    "pomegranate": 18.0,
    "banana":      200.0,
    "mango":       20.0,
    "grapes":      25.0,
    "watermelon":  200.0,
    "muskmelon":   100.0,
    "apple":       25.0,
    "orange":      30.0,
    "papaya":      100.0,
    "coconut":     80.0,
    "cotton":      15.0,
    "jute":        25.0,
    "coffee":       8.0,
}

# Assumed production cost (INR/hectare) - 1 USD ≈ 83 INR
PRODUCTION_COST = {
    "rice":        33200,
    "maize":       29050,
    "chickpea":    20750,
    "kidneybeans": 23240,
    "pigeonpeas":  19090,
    "mothbeans":   16600,
    "mungbean":    18260,
    "blackgram":   17430,
    "lentil":      19920,
    "pomegranate": 49800,
    "banana":      66400,
    "mango":       41500,
    "grapes":      74700,
    "watermelon":  58100,
    "muskmelon":   49800,
    "apple":       83000,
    "orange":      45650,
    "papaya":      53950,
    "coconut":     33200,
    "cotton":      37350,
    "jute":        23240,
    "coffee":      99600,
}

CROP_ICONS = {
    "rice":        "🌾",
    "maize":       "🌽",
    "chickpea":    "🫘",
    "kidneybeans": "🫘",
    "pigeonpeas":  "🫛",
    "mothbeans":   "🌱",
    "mungbean":    "🌱",
    "blackgram":   "🌱",
    "lentil":      "🌿",
    "pomegranate": "🍎",
    "banana":      "🍌",
    "mango":       "🥭",
    "grapes":      "🍇",
    "watermelon":  "🍉",
    "muskmelon":   "🍈",
    "apple":       "🍎",
    "orange":      "🍊",
    "papaya":      "🍑",
    "coconut":     "🥥",
    "cotton":      "🌸",
    "jute":        "🌿",
    "coffee":      "☕",
}


def calculate_profit(crop_name: str) -> float:
    """Return estimated net profit in USD per hectare."""
    crop = crop_name.lower()
    price = CROP_PRICES.get(crop, 40.0)
    yld   = CROP_YIELD.get(crop, 15.0)
    cost  = PRODUCTION_COST.get(crop, 300)
    return round(price * yld - cost, 2)


def enrich_predictions(top_3: list) -> list:
    """
    Attach profit, icon, probability % to each top-3 prediction.
    Returns list of dicts sorted by probability (already sorted).
    """
    enriched = []
    for crop, prob in top_3:
        enriched.append({
            "name":     crop.capitalize(),
            "raw_name": crop.lower(),
            "prob":     round(prob * 100, 2),
            "profit":   calculate_profit(crop),
            "icon":     CROP_ICONS.get(crop.lower(), "🌱"),
        })
    return enriched


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = {
            "N":           float(request.form["N"]),
            "P":           float(request.form["P"]),
            "K":           float(request.form["K"]),
            "temperature": float(request.form["temperature"]),
            "humidity":    float(request.form["humidity"]),
            "ph":          float(request.form["ph"]),
            "rainfall":    float(request.form["rainfall"]),
        }
    except (KeyError, ValueError) as e:
        return render_template("index.html", error=f"Invalid input: {e}")

    result   = predict_crop(features)
    top_3    = enrich_predictions(result["top_3"])

    # Best crop = highest confidence
    best_crop = top_3[0]

    # Most profitable among top 3
    profitable_crop = max(top_3, key=lambda x: x["profit"])

    return render_template(
        "result.html",
        features=features,
        top_3=top_3,
        best_crop=best_crop,
        profitable_crop=profitable_crop,
    )


@app.route("/train", methods=["GET"])
def train():
    """Trigger model re-training (admin convenience route)."""
    from model import train_model
    try:
        info = train_model()
        return f"<pre>Training complete!\nAccuracy: {info['accuracy']*100:.2f}%\n\n{info['report']}</pre>"
    except FileNotFoundError as e:
        return str(e), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
