import tensorflow as tf

model = tf.keras.models.load_model("model/mask_detector.h5")
model.save("model/mask_detector.h5")
print("Done!")