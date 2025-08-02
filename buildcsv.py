import os
import pandas as pd

# Path to your CatMeows audio folder
AUDIO_DIR = r"C:\Users\user\Desktop\myenv\myenv\pet\dataset meow"  # ⬅️ Replace this with your actual path

# Mapping from file prefix to emotion
emotion_map = {
    "B": "Calm",        # Brushing
    "I": "Anxious",     # Isolation
    "F": "Hungry"       # Waiting for food
}

# Build rows for the CSV
rows = []
for filename in os.listdir(AUDIO_DIR):
    if filename.endswith(".wav"):
        prefix = filename[0].upper()
        emotion = emotion_map.get(prefix, "Unknown")
        file_path = os.path.join(AUDIO_DIR, filename)
        rows.append({
            "id": len(rows) + 1,
            "pet_type": "Cat",
            "behavior": "Meow",
            "modality": "Audio",
            "file_path": file_path,
            "emotion": emotion
        })

# Create DataFrame and save as CSV
df = pd.DataFrame(rows)
df.to_csv("labels.csv", index=False)
print(f"✅ labels.csv generated with {len(df)} entries.")
