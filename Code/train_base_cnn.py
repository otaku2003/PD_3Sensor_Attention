import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Dense, Flatten, Input
)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# --------------------------------------------------
# Config
# --------------------------------------------------
SEED = 11
EPOCHS = 60
BATCH_SIZE = 32
LR = 1e-4

np.random.seed(SEED)
tf.random.set_seed(SEED)

# --------------------------------------------------
# Load data (Single Sensor Only)
# --------------------------------------------------
X = np.load("X_400.npy")
Y = np.load("Y_scaled.npy")

print("Loaded X shape:", X.shape)

# ------------------------------------
# Handle different X formats safely
# ------------------------------------
if X.ndim == 3:
    # (N, 400, 3)  ← multichannel
    SENSOR_ID = 0   # 0, 1, or 2
    X = X[:, :, SENSOR_ID]   # → (N, 400)
    X = X[..., np.newaxis]   # → (N, 400, 1)

elif X.ndim == 2:
    # (N, 400)
    X = X[..., np.newaxis]

else:
    raise ValueError(f"Unexpected X shape: {X.shape}")

print("Using X shape for training:", X.shape)

print("Y shape:", Y.shape)
print("X min / max:", X.min(), X.max())

# --------------------------------------------------
# Base CNN (Fig.4)
# --------------------------------------------------
# ------------------------------------
# Train / Test Split
# ------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=11,
    shuffle=True
)

model = Sequential([
    Input(shape=(400, 1)),

    Conv1D(64, kernel_size=10, activation="relu"),
    MaxPooling1D(pool_size=4),

    Conv1D(128, kernel_size=7, activation="relu"),
    MaxPooling1D(pool_size=4),

    Conv1D(256, kernel_size=5, activation="relu"),
    MaxPooling1D(pool_size=2),

    Flatten(),

    Dense(512, activation="relu"),
    Dense(256, activation="relu", name="embedding"),
    Dense(3)

])

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="mse"
)

model.summary()

# --------------------------------------------------
# Train
# --------------------------------------------------
history = model.fit(
    X_train, Y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2
)


# --------------------------------------------------
# Evaluate (scaled space)
# --------------------------------------------------
Y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
print(f"RMSE (scaled): {rmse:.4f}")

# --------------------------------------------------
# Save model
# --------------------------------------------------
os.makedirs("models", exist_ok=True)
model.save("base_cnn_single_sensor.h5")

print("Std of predictions:", np.std(Y_pred, axis=0))
model.save("base_cnn_sensor0.h5")
print("Base CNN saved successfully.")
