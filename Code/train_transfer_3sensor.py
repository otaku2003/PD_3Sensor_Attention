import numpy as np
import tensorflow as tf
from keras import ops
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Softmax, Multiply, Reshape
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Concatenate, Softmax

# ----------------------------
# Load data
# ----------------------------
X = np.load("X_400.npy")      # shape: (N, 400, 3)
y = np.load("Y_scaled.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=11
)

# Split sensors
X1 = X_train[:, :, 0:1]
X2 = X_train[:, :, 1:2]
X3 = X_train[:, :, 2:3]

X1_test = X_test[:, :, 0:1]
X2_test = X_test[:, :, 1:2]
X3_test = X_test[:, :, 2:3]


# ----------------------------
# Load base feature extractor
# ----------------------------
base_model = tf.keras.models.load_model(
    "base_cnn_single_sensor.h5",
    compile=False
)

# Extract feature layer (256-D)
base_feature_model = Model(
    inputs=base_model.inputs[0],
    outputs=base_model.get_layer("embedding").output
)


# ✅ PARTIAL UNFREEZE — allow attention interaction
for layer in base_feature_model.layers[:-20]:
    layer.trainable = False

for layer in base_feature_model.layers[-20:]:
    layer.trainable = True


# ----------------------------
# Inputs
# ----------------------------
inp1 = Input(shape=(400, 1), name="sensor_1")
inp2 = Input(shape=(400, 1), name="sensor_2")
inp3 = Input(shape=(400, 1), name="sensor_3")

# Extract features
f1 = base_feature_model(inp1)
f2 = base_feature_model(inp2)
f3 = base_feature_model(inp3)

# ----------------------------
# Attention mechanism (NO Lambda ✅)
# ----------------------------

# (batch, 256) → (batch, 1, 256)
f1_exp = Reshape((1, 256))(f1)
f2_exp = Reshape((1, 256))(f2)
f3_exp = Reshape((1, 256))(f3)

# (batch, 3, 256)
features = Concatenate(axis=1)([f1_exp, f2_exp, f3_exp])

# Attention scores
scores = Dense(64, activation="relu")(features)
scores = Dense(1)(scores)

scores = Reshape((3,))(scores)                   # (batch, 3)

temperature = 0.1   # ⬅ خیلی مهم
scores_scaled = scores / temperature

attention_weights = Softmax(
    axis=1,
    name="attention_weights"
)(scores_scaled)


# Apply attention
weights_exp = Reshape((3, 1))(attention_weights)
weighted_features = Multiply()([features, weights_exp])

# Feature fusion
fused = ops.sum(weighted_features, axis=1)

# ----------------------------
# Regression head
# ----------------------------
h = Dense(512, activation="relu")(fused)
h = Dense(256, activation="relu")(h)
out = Dense(3, activation="linear")(h)

model = Model(inputs=[inp1, inp2, inp3], outputs=out)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="mse"
)

model.summary()


# ----------------------------
# Train
# ----------------------------
history = model.fit(
    [X1, X2, X3], y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=32,
    verbose=2
)

# ----------------------------
# Evaluate
# ----------------------------
pred = model.predict([X1_test, X2_test, X3_test])
rmse = np.sqrt(np.mean((pred - y_test) ** 2, axis=0))

print("RMSE (scaled):", rmse)
print("Std of predictions:", np.std(pred, axis=0))
model.save("cnn_3sensor_transfer_learning.keras")
