import tensorflow as tf
import os

BASE = os.path.dirname(__file__)

print("Loading old model...")
old_model = tf.keras.models.load_model(
    os.path.join(BASE, "model", "mask_detector.h5"),
    compile=False
)

print("Saving as .keras format...")
old_model.save(os.path.join(BASE, "model", "mask_detector_final.keras"))
print("Done!")