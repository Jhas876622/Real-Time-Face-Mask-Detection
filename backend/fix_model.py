import tensorflow as tf
import numpy as np

print("Loading model...")
# Force legacy loader
model = tf.keras.models.load_model(
    "model/mask_detector.h5",
    compile=False
)

print("Building model...")
model.build((None, 224, 224, 3))

print("Re-saving model...")
# Save in SavedModel format (not h5)
model.save("model/mask_detector_new", save_format="tf")

print("Done! New model saved as mask_detector_new folder")