import numpy as np
import os
import cv2
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Dummy dataset generator (replace with real data)
def generate_data():
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=True)
    X, y = [], []
    moods = ['happy', 'sad', 'angry']
    for mood_idx, mood in enumerate(moods):
        for i in range(10):  # 10 samples per mood
            img = np.zeros((480, 640, 3), dtype=np.uint8)  # dummy black image
            features = extract_features(img, holistic)
            X.append(features)
            y.append(mood_idx)
    return np.array(X), np.array(y), moods

# Feature extraction
def extract_features(image, holistic):
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
        features.extend([0.0] * (468 * 2))  # 468 points * 2 (x, y)
    return np.array(features)

# Prepare dataset
X, y, mood_labels = generate_data()

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.reshape(-1, X_scaled.shape[1], 1)  # For Conv1D

# One-hot encode labels
y_cat = to_categorical(y, num_classes=len(mood_labels))

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_cat, test_size=0.2, random_state=42)

# Define CNN model with more layers and regularization
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_scaled.shape[1], 1)),
    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(128, 3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(mood_labels), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=8,
    callbacks=[early_stop]
)

# Save model and scaler
os.makedirs("cnn_model", exist_ok=True)
model.save("cnn_model/emotion_cnn_model.h5")
joblib.dump(scaler, "cnn_model/scaler.pkl")
print("✅ Training complete. Model and scaler saved.")