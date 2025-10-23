import numpy as np, librosa


def load_mono(path, sr=16000):
    path = str(path)
    x, sr = librosa.load(path, sr=sr, mono=True)
    x, _ = librosa.effects.trim(x, top_db=30)
    rms = np.sqrt(np.mean(x**2)) + 1e-8
    x = x * (0.05 / rms)  # simple RMS target ≈ loudness norm
    return x, sr


def extract_features(x, sr=16000, n_mels=64, n_mfcc=20):
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=n_mels)
    logmel = librosa.power_to_db(S + 1e-9)
    logmel_stats = np.hstack([logmel.mean(axis=1), logmel.std(axis=1)])

    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S + 1e-9), sr=sr, n_mfcc=n_mfcc)
    frames = mfcc.shape[1]
    width = min(9, frames if frames % 2 else frames - 1)
    if width < 3:
        d1 = np.zeros_like(mfcc)
        d2 = np.zeros_like(mfcc)
    else:
        d1 = librosa.feature.delta(mfcc, width=width)
        d2 = librosa.feature.delta(mfcc, width=width, order=2)
    mfcc_stats = np.hstack([mfcc.mean(axis=1), mfcc.std(axis=1),
                            d1.mean(axis=1), d1.std(axis=1),
                            d2.mean(axis=1), d2.std(axis=1)])

    zcr = librosa.feature.zero_crossing_rate(x).mean()
    centroid = librosa.feature.spectral_centroid(y=x, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr).mean()
    flatness = librosa.feature.spectral_flatness(y=x).mean()
    return np.hstack([logmel_stats, mfcc_stats, [zcr, centroid, rolloff, flatness]])
