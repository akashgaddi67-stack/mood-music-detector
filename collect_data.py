import cv2
import numpy as np
import mediapipe as mp

# Setup MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic()

# Start webcam
cap = cv2.VideoCapture(0)

label = input("Enter mood label (e.g., happy, sad, angry): ")
data = []
frame_count = 0

print("Collecting data... Press ESC to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror frame
    frame = cv2.flip(frame, 1)

    # Convert color to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    landmarks = []

    # Face landmarks (normalized relative to base point)
    if results.face_landmarks:
        base_x = results.face_landmarks.landmark[1].x
        base_y = results.face_landmarks.landmark[1].y
        for lm in results.face_landmarks.landmark:
            landmarks.extend([lm.x - base_x, lm.y - base_y])
    else:
        landmarks.extend([0.0] * 468 * 2)

    # Left hand landmarks
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y])
    else:
        landmarks.extend([0.0] * 21 * 2)

    # Right hand landmarks
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y])
    else:
        landmarks.extend([0.0] * 21 * 2)

    data.append(landmarks)
    frame_count += 1

    # Draw landmarks
    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    cv2.putText(frame, f"Frames collected: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Collecting Data", frame)

    if cv2.waitKey(1) == 27:  # ESC key to stop
        break

cap.release()
cv2.destroyAllWindows()

# Save collected data as .npy file named by label
np.save(f"{label}.npy", np.array(data))
print(f"Saved {frame_count} samples to '{label}.npy'")