import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
 
dataset = "dataset"
 
data = []
labels = []
 
for category in ["with_mask", "without_mask"]:
    path = os.path.join(dataset, category)
 
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
 
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)  # Use MobileNetV2's own preprocessing
 
        data.append(image)
        labels.append(category)
 
data = np.array(data, dtype="float32")
labels = np.array(labels)
 
le = LabelEncoder()
labels = le.fit_transform(labels)
 
labels = to_categorical(labels, num_classes=2)
 
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)
 
baseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)
 
head = baseModel.output
head = AveragePooling2D(pool_size=(7, 7))(head)
head = Flatten()(head)
head = Dense(128, activation="relu")(head)
head = Dropout(0.5)(head)
# BUG FIX: Output layer was Dense(3) — must match num_classes=2
head = Dense(2, activation="softmax")(head)
 
model = Model(inputs=baseModel.input, outputs=head)
 
for layer in baseModel.layers:
    layer.trainable = False
 
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=1e-4),
    metrics=["accuracy"]
)
 
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)
 
print("[INFO] Training model...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=32),
    steps_per_epoch=len(trainX) // 32,
    validation_data=(testX, testY),
    validation_steps=len(testX) // 32,
    epochs=25
)
 
model.save("mask_detector.h5")
print("[INFO] Model saved to mask_detector.h5")
print("[INFO] Class order:", le.classes_)

