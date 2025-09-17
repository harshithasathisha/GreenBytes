import os
import json
import re
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from flask import Flask, jsonify, request, render_template, redirect, url_for
import random

# -------------------- Paths -------------------- #
APP_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(APP_DIR, "templates")
STATIC_DIR = os.path.join(APP_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")   # FIX: uploads folder path
DATA_FILE = os.path.join(APP_DIR, "data", "crops.json")
FAV_FILE = os.path.join(APP_DIR, "data", "favorites.json")

# Make sure uploads folder exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATE_DIR)

# Placeholder soil model (replace with real model later)
soil_model = None  

# -------------------- Soil Type → Crops -------------------- #
SOIL_CROPS = {
    "Laterite": ["Tea", "Coffee", "Cashew", "Coconut", "Rubber", "Arecanut", "Tapioca", "Spices", "Pineapple", "Jackfruit"],
    "Alluvial": ["Rice", "Wheat", "Sugarcane", "Jute", "Maize", "Barley", "Pulses", "Oilseeds", "Fruits", "Vegetables"],
    "Black": ["Cotton", "Soybean", "Sunflower", "Groundnut", "Tobacco", "Millets", "Citrus Fruits", "Pomegranate"],
    "Red": ["Groundnut","Millets","Cotton","Wheat","Pulses","Potato","Oilseeds","Onion","Tomato","Chillies"],  
    "Desert": ["Bajra","Barley","Guar","Mustard","Cumin","Dates","Castor","Fodder Crops","Jowar","Moth Beans"],
    "Mountain": ["Apple","Peach","Plum","Maize","Barley","Tea","Walnut","Almond","Pear","Medicinal Herbs"]
}

# -------------------- Utility Functions -------------------- #
def load_crops():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_favs():
    if not os.path.exists(FAV_FILE):
        with open(FAV_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
    with open(FAV_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_favs(favs):
    with open(FAV_FILE, "w", encoding="utf-8") as f:
        json.dump(favs, f, ensure_ascii=False, indent=2)

# -------------------- Frontend Routes -------------------- #
@app.route("/")
def index():
    return redirect(url_for('register'))

@app.route("/register", methods=["GET", "POST"])
def register():
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    return render_template("login.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/categories")
def categories():
    return render_template("categories.html")

@app.route("/soil-scan")
def soil_scan():
    return render_template("soil_scan.html")

# -------------------- Soil Scan -------------------- #
@app.route('/manual_soil', methods=['POST'])
def manual_soil():
    soil_type = request.form['soil_type']
    crops = SOIL_CROPS.get(soil_type, ["No crop data available"])
    return render_template("soil_scan.html", result=soil_type, crops=crops, image_path=None)

@app.route('/predict_soil', methods=['POST'])
def predict_soil():
    file = request.files['file']
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    file.save(filepath)

    if soil_model:  # if a model is loaded
        img = Image.open(filepath).resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = soil_model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        soil_types = list(SOIL_CROPS.keys())
        soil_type = soil_types[predicted_class]
    else:  # fallback to random
        soil_type = random.choice(list(SOIL_CROPS.keys()))

    crops = SOIL_CROPS.get(soil_type, ["No crop data available"])
    return render_template("soil_scan.html", result=soil_type, crops=crops, image_path=filepath)

@app.route("/live_scan", methods=["POST"])
def live_scan():
    if not soil_model:
        return jsonify({"soil_type": "Model not found", "crops": []})

    data = request.get_json()
    img_data = data["image"]

    img_data = re.sub('^data:image/.+;base64,', '', img_data)
    img = Image.open(BytesIO(base64.b64decode(img_data))).resize((128, 128))

    live_filename = os.path.join(UPLOAD_DIR, "live_capture.jpg")
    img.save(live_filename)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = soil_model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    soil_types = list(SOIL_CROPS.keys())
    soil_type = soil_types[predicted_class]
    crops = SOIL_CROPS.get(soil_type, ["No crop data available"])

    return jsonify({
        "soil_type": soil_type,
        "crops": crops,
        "saved_path": live_filename
    })

# -------------------- Crop Pages -------------------- #
@app.route("/crop/<crop_id>")
def crop_page(crop_id):
    crops = load_crops()
    crop = next((c for c in crops if c["id"] == crop_id), None)
    if not crop:
        return "Crop not found", 404
    return render_template("crop.html", crop=crop)

# -------------------- API Routes -------------------- #
@app.route("/api/crops")
def api_crops():
    q = request.args.get("q", "").strip().lower()
    lang = request.args.get("lang", "en")
    crops = load_crops()

    if q:
        crops = [c for c in crops if q in c["name_en"].lower() 
                 or q in c["name_kn"].lower() 
                 or q in c.get("short_en", "").lower() 
                 or q in c.get("short_kn", "").lower() 
                 or q in c.get("desc_en", "").lower() 
                 or q in c.get("desc_kn", "").lower()]

    out = []
    for c in crops:
        out.append({
            "id": c["id"],
            "name": c["name_en"] if lang == "en" else c["name_kn"],
            "thumb": c.get("thumb", ""),
            "short": c.get("short_en", "") if lang == "en" else c.get("short_kn", "")
        })
    return jsonify(out)

@app.route("/api/crop/<crop_id>")
def api_crop(crop_id):
    crops = load_crops()
    crop = next((c for c in crops if c["id"] == crop_id), None)
    if not crop:
        return jsonify({"error": "not found"}), 404
    return jsonify(crop)

@app.route("/api/favorites", methods=["GET", "POST", "DELETE"])
def api_favorites():
    if request.method == "GET":
        return jsonify(load_favs())

    data = request.get_json() or {}
    favs = load_favs()

    if request.method == "POST":
        crop_id = data.get("id")
        if crop_id and crop_id not in favs:
            favs.append(crop_id)
            save_favs(favs)
        return jsonify(favs)

    if request.method == "DELETE":
        crop_id = data.get("id")
        if crop_id and crop_id in favs:
            favs.remove(crop_id)
            save_favs(favs)
        return jsonify(favs)

# -------------------- Run -------------------- #
if __name__ == "__main__":
    app.run(debug=True)
