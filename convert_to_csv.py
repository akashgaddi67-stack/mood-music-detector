import numpy as np
import pandas as pd

# Load all .npy files for moods you collected
moods = ['happy', 'sad', 'angry']  # Add or remove moods accordingly

all_data = []
all_labels = []

for mood in moods:
    data = np.load(f"{mood}.npy")
    all_data.append(data)
    all_labels.extend([mood] * len(data))

X = np.concatenate(all_data)
y = all_labels

df = pd.DataFrame(X)
df.insert(0, "label", y)
df.to_csv("emotion_data.csv", index=False)

print("Saved combined data to emotion_data.csv")