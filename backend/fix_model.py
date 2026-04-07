import tensorflow as tf

# Load old model
model = tf.keras.models.load_model("model/mask_detector.h5")

# Re-save in new format
model.save("model/mask_detector.h5", save_format="h5")

print("Model re-saved successfully!")