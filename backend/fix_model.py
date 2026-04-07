import tensorflow as tf

print("Loading model...")
model = tf.keras.models.load_model(
    "model/mask_detector.h5",
    compile=False
)

print("Re-saving...")
model.save("model/mask_detector_new.h5")

print("Done!")