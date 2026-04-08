import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

# Load the old model
old_model_path = 'model/mask_detector_new.h5'
print(f"Loading model from {old_model_path}...")

model = tf.keras.models.load_model(old_model_path, compile=False)
print("Model loaded successfully!")

# Save in SavedModel format (more compatible)
new_model_path = 'model/mask_detector_saved'
model.save(new_model_path)
print(f"Model saved to {new_model_path}")

print("Done! Now commit and push.")