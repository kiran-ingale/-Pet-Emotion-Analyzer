import os
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# STEP 1: Load CSV
print("🔄 Loading dataset...")
df = pd.read_csv("labels.csv")

features = []
labels = []

# STEP 2: Extract MFCC + delta + delta2
print("🎧 Extracting MFCC, delta, and delta-delta features...")
for i, row in df.iterrows():
    try:
        y, sr = librosa.load(row["file_path"], sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        combined = np.vstack([mfcc, delta, delta2])
        mfcc_mean = np.mean(combined.T, axis=0)
        features.append(mfcc_mean)
        labels.append(row["emotion"])
    except Exception as e:
        print(f"❌ Error loading {row['file_path']}: {e}")

X = np.array(features)
y = np.array(labels)

print(f"✅ Extracted features from {len(X)} audio samples.")

# STEP 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 4: Optional SMOTE oversampling (if needed)
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

# STEP 5: Standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)
import joblib
joblib.dump(scaler, "scaler.pkl")


# STEP 6: Train classifier
print("🧠 Training RandomForestClassifier...")
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
clf.fit(X_train_scaled, y_train_bal)

# STEP 7: Predict and evaluate
print("📊 Classification Report:")
y_pred = clf.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# STEP 8: Visualize feature clusters with t-SNE
try:
    print("📈 Visualizing feature clusters with t-SNE...")
    X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=pd.factorize(y)[0], cmap="Set2")
    plt.title("MFCC Feature Clustering (t-SNE)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()
except Exception as e:
    print(f"⚠️ Visualization skipped: {e}")

# STEP 9: Save model
import joblib
joblib.dump(clf, "emotion_model.pkl")
joblib.dump(scaler, "scaler.pkl")
