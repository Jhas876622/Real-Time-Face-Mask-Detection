import tensorflow as tf
import numpy as np
import os

print("Step 1: Loading old model weights only...")
BASE = os.path.dirname(__file__)

# Load old model
old_model = tf.keras.models.load_model(
    os.path.join(BASE, "model", "mask_detector.h5"),
    compile=False
)

print("Step 2: Building fresh model with same architecture...")
# Rebuild same architecture fresh
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None
)

x = base_model.output
x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(2, activation="softmax")(x)

new_model = tf.keras.Model(inputs=base_model.input, outputs=x)

print("Step 3: Copying weights...")
new_model.set_weights(old_model.get_weights())

print("Step 4: Saving new model...")
new_model.save(os.path.join(BASE, "model", "mask_detector_v2.h5"))

print("Done! mask_detector_v2.h5 created")