import numpy as np
import pandas as pd

EPS = 1e-9
N_SAMPLES = 400


def cut_signal(signal, threshold=0.001, target_len=400):
    idx = np.argmax(np.abs(signal) > threshold)
    cut = signal[idx:]

    if len(cut) < target_len:
        cut = np.pad(cut, (0, target_len - len(cut)))

    return cut[:target_len]


def load_signal(path):
    df = pd.read_csv(path, compression="gzip", header=None, engine="python")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="any")
    return df.values.astype(np.float32)


# =========================
# Load data
# =========================
Xx = load_signal("DB_THREE_UPSCALE_x_SIGNALS.csv")
Xy = load_signal("DB_THREE_UPSCALE_y_SIGNALS.csv")
Xz = load_signal("DB_THREE_UPSCALE_z_SIGNALS.csv")

Y = pd.read_csv("DB_THREE_UPSCALE_LOCATONS.csv").values.astype(np.float32)

N = Y.shape[0]
Xx, Xy, Xz = Xx[:N], Xy[:N], Xz[:N]

# Stack → (N, L, 3)
X_raw = np.stack([Xx, Xy, Xz], axis=-1)

# =========================
# Cut-off (CRITICAL STEP)
# =========================
X_cut = np.zeros((N, N_SAMPLES, 3), dtype=np.float32)

for i in range(N):
    for ch in range(3):
        X_cut[i, :, ch] = cut_signal(X_raw[i, :, ch])

# =========================
# Normalize (USE X_cut)
# =========================
X_norm = np.zeros_like(X_cut)

for i in range(N):
    for ch in range(3):
        s = X_cut[i, :, ch]
        X_norm[i, :, ch] = s / (np.max(np.abs(s)) + EPS)

X_norm = np.nan_to_num(X_norm)

# =========================
# Concatenate 3 sensors (CRITICAL)
# =========================
X_concat = np.concatenate(
    [X_norm[:, :, 0], X_norm[:, :, 1], X_norm[:, :, 2]],
    axis=1
)

# Shape: (N, 1200, 1)
X_concat = X_concat[..., np.newaxis]

np.save("X_1200.npy", X_concat)


# =========================
# Label processing
# =========================
Y_shift = Y - Y.min(axis=0)
Y_scaled = Y_shift / Y_shift.max(axis=0)

assert not np.isnan(X_norm).any()
assert not np.isnan(Y_scaled).any()


np.save("Y_scaled.npy", Y_scaled)

print("✅ Preprocessing DONE:", X_norm.shape)
