from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import joblib
import base64
import re

app = Flask(__name__)
CORS(app)

# Load model and scaler
model = load_model("cnn_model/emotion_cnn_model.h5")
scaler = joblib.load("cnn_model/scaler.pkl")

mood_labels = ["happy", "sad", "angry"]
mood_to_song = {
    "happy": "https://www.youtube.com/embed/ZbZSe6N_BXs",
    "sad": "https://www.youtube.com/embed/hLQl3WQQoQ0",
    "angry": "https://www.youtube.com/embed/7wtfhZwyrcc"
}

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=True)

def preprocess_image(base64_img):
    img_data = re.sub('^data:image/.+;base64,', '', base64_img)
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def extract_features(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    features = []

    if results.face_landmarks:
        base_x = results.face_landmarks.landmark[1].x
        base_y = results.face_landmarks.landmark[1].y
        for lm in results.face_landmarks.landmark:
            features.append(lm.x - base_x)
            features.append(lm.y - base_y)
    else:
        features.extend([0.0] * (468 * 2))
    return np.array(features).reshape(1, -1)

@app.route('/detect_mood', methods=['POST'])
def detect_mood():
    data = request.get_json()
    img_data = data.get('image')

    if not img_data:
        return jsonify({"error": "No image data"}), 400

    try:
        img = preprocess_image(img_data)
        features = extract_features(img)
        features_scaled = scaler.transform(features).reshape(1, -1, 1)
        predictions = model.predict(features_scaled)[0]
        mood = mood_labels[np.argmax(predictions)]
        return jsonify({
            "mood": mood,
            "confidence": float(np.max(predictions)),
            "youtube_url": mood_to_song.get(mood, "")
        })
    except Exception as e:
        return jsonify({"error": "Failed", "details": str(e)}), 500

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)