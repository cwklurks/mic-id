import os, glob, json, numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
from features import load_mono, extract_features

DATA_DIR, MODEL_DIR, REPORT_DIR = "data", "models", "reports"
os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(REPORT_DIR, exist_ok=True)


def load_dataset():
    X, y = [], []
    for device in sorted(d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))):
        for wav in glob.glob(os.path.join(DATA_DIR, device, "*.wav")):
            x, sr = load_mono(wav); feats = extract_features(x, sr)
            X.append(feats); y.append(device)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = load_dataset()
    le = LabelEncoder(); y_enc = le.fit_transform(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y_enc, test_size=0.25, stratify=y_enc, random_state=42)

    clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr); yhat = clf.predict(Xte)

    report = classification_report(yte, yhat, target_names=le.classes_, output_dict=True)
    with open(os.path.join(REPORT_DIR, "metrics.json"), "w") as f: json.dump(report, f, indent=2)

    cm = confusion_matrix(yte, yhat, normalize="true")
    fig, ax = plt.subplots(figsize=(5,4)); im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(le.classes_))); ax.set_xticklabels(le.classes_, rotation=45, ha="right")
    ax.set_yticks(range(len(le.classes_))); ax.set_yticklabels(le.classes_)
    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Confusion (normalized)"); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=160)

    joblib.dump(clf, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(le,  os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print("Saved model + reports.")
