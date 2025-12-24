import seaborn as sns
from scipy.stats import kruskal
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from scipy.stats import pearsonr, spearmanr

# ==================================
# 1. CONFIGURATION
# ==================================
MODEL_PATH = "cnn_3sensor_transfer_learning.keras"

# اگر مدل در حافظه است، False کن
LOAD_MODEL_FROM_DISK = True

# ----------------------------------
# نام سنسورها (برای نمایش)
# ----------------------------------
SENSOR_NAMES = ["Sensor-1", "Sensor-2", "Sensor-3"]

# ==================================
# 2. LOAD MODEL
# ==================================
if LOAD_MODEL_FROM_DISK:
    print("[INFO] Loading trained model...")
    model = load_model(
        MODEL_PATH,
        compile=False
        # safe_mode = False
    )


else:
    print("[INFO] Using existing model from memory")

# ==================================
# 3. BUILD ATTENTION EXTRACTION MODEL
# ==================================
print("[INFO] Building attention extractor...")
attention_layer = model.get_layer("attention_weights")


attention_model = Model(
    inputs=model.inputs,
    outputs=attention_layer.output
)


attention_model.summary()

# ==================================
# LOAD TEST DATA
# ==================================
X = np.load("X_400.npy")
y = np.load("Y_scaled.npy")


_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=11
)

X1_test = X_test[:, :, 0:1]
X2_test = X_test[:, :, 1:2]
X3_test = X_test[:, :, 2:3]

# ==================================
# 4. EXTRACT ATTENTION WEIGHTS
# ==================================

print("[INFO] Predicting attention weights...")
att_weights = attention_model.predict(
    [X1_test, X2_test, X3_test],
    batch_size=64,
    verbose=1
)
# ===== SANITY CHECK =====
print("[CHECK] Attention weights sanity check")
print("Shape:", att_weights.shape)
print("NaN count:", np.isnan(att_weights).sum())
print("Min:", att_weights.min(), "Max:", att_weights.max())
print("Sum per sample (mean):", att_weights.sum(axis=1).mean())

assert att_weights.shape[0] > 0, "Attention output is EMPTY"
assert not np.isnan(att_weights).any(), "Attention contains NaN"
assert np.all(att_weights >= 0), "Negative attention detected"
assert np.all(att_weights <= 1), "Attention > 1 detected"


# Shape check
assert att_weights.shape[1] == 3, "Attention output must have size 3"

print("[INFO] Attention weights extracted")
print("Shape:", att_weights.shape)

print("Feature fusion output shape:", att_weights.shape)


# ==================================
# 5. BASIC STATISTICAL ANALYSIS
# ==================================
mean_att = att_weights.mean(axis=0)
std_att = att_weights.std(axis=0)

print("\n=== ATTENTION STATISTICS ===")
for i, name in enumerate(SENSOR_NAMES):
    print(f"{name}: Mean = {mean_att[i]:.4f}, Std = {std_att[i]:.4f}")

# ==================================
# 6. ATTENTION DISTRIBUTION (KDE)
# ==================================


print("[INFO] Plotting attention KDE distributions...")

plt.figure(figsize=(10, 4))

for i in range(3):
    plt.subplot(1, 3, i + 1)

    sns.kdeplot(
        att_weights[:, i],
        fill=True,
        linewidth=2
    )

    plt.title(SENSOR_NAMES[i])
    plt.xlabel("Attention Weight")
    plt.ylabel("Density")

plt.tight_layout()
plt.savefig("fig_attention_kde.png", dpi=300)
plt.close()


# ==================================
# 7. CORRELATION WITH PD LOCATION (OPTIONAL)
# ==================================
print("[INFO] Plotting attention vs PD location...")

coords = ["x", "y", "z"]

for dim, coord in enumerate(["x", "y", "z"]):
    plt.figure(figsize=(7, 5))
    for s in range(3):
        plt.scatter(
            y_test[:, dim],
            att_weights[:, s],
            alpha=0.4,
            label=SENSOR_NAMES[s]
        )
    plt.xlabel(f"PD location ({coord})")
    plt.ylabel("Attention weight")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==================================
# 8. CORRELATION ANALYSIS (STATISTICAL)
# ==================================


coords = ["x", "y", "z"]

print("\n=== CORRELATION ANALYSIS ===")

results = []

for s_idx, sensor_name in enumerate(SENSOR_NAMES):
    for c_idx, coord in enumerate(coords):
        # Pearson correlation
        pearson_r, pearson_p = pearsonr(
            att_weights[:, s_idx],
            y_test[:, c_idx]
        )

        # Spearman correlation (rank-based, robust)
        spearman_r, spearman_p = spearmanr(
            att_weights[:, s_idx],
            y_test[:, c_idx]
        )

        results.append({
            "Sensor": sensor_name,
            "Coord": coord,
            "Pearson_r": pearson_r,
            "Pearson_p": pearson_p,
            "Spearman_r": spearman_r,
            "Spearman_p": spearman_p
        })

        print(
            f"{sensor_name} vs PD-{coord.upper()} | "
            f"Pearson r = {pearson_r:+.4f} (p={pearson_p:.3e}), "
            f"Spearman r = {spearman_r:+.4f} (p={spearman_p:.3e})"
        )

# ==================================
# 9. REGION-WISE BOXPLOT ANALYSIS
# ==================================

coords = ["x", "y", "z"]

for c_idx, coord in enumerate(coords):

    values = y_test[:, c_idx]

    # Define regions using quartiles
    q1, q3 = np.percentile(values, [25, 75])

    regions = np.zeros(len(values), dtype=int)
    regions[values <= q1] = 0        # Low
    regions[(values > q1) & (values <= q3)] = 1  # Mid
    regions[values > q3] = 2         # High

    plt.figure(figsize=(9, 5))

    box_data = []
    labels = []

    for s_idx, sensor_name in enumerate(SENSOR_NAMES):
        for r in range(3):
            box_data.append(att_weights[regions == r, s_idx])
            labels.append(f"{sensor_name}\n{['Low','Mid','High'][r]}")

    plt.boxplot(box_data, showfliers=False)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45)
    plt.ylabel("Attention Weight")
    plt.title(f"Region-wise Attention Distribution (PD {coord.upper()})")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    fname = f"fig_attention_boxplot_PD_{coord}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()

    print(f"[INFO] Saved {fname}")

# ==================================
# 10. KRUSKAL-WALLIS REGION TEST
# ==================================

print("\n=== KRUSKAL-WALLIS REGION TEST ===")

coords = ["x", "y", "z"]
regions_name = ["Low", "Mid", "High"]

for c_idx, coord in enumerate(coords):

    values = y_test[:, c_idx]
    q1, q3 = np.percentile(values, [25, 75])

    regions = np.zeros(len(values), dtype=int)
    regions[values <= q1] = 0
    regions[(values > q1) & (values <= q3)] = 1
    regions[values > q3] = 2

    print(f"\nPD-{coord.upper()}")

    for s_idx, sensor_name in enumerate(SENSOR_NAMES):

        group_data = [
            att_weights[regions == r, s_idx]
            for r in range(3)
        ]

        H, p = kruskal(*group_data)

        print(
            f"  {sensor_name}: "
            f"H = {H:.4f}, p = {p:.4e}"
        )


print("[DONE] Attention analysis completed successfully!")
