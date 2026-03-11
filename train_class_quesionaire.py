import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
import cv2

# =========================
# PATH
# =========================
IMG_PATH = "Questionair Images/Questionair Images"
CSV_PATH = "data_from_questionaire.csv"
IMG_SIZE = 128

# =========================
# CLASS MAP
# =========================
CLASS_INDICES = {
    "Burger":  0,
    "Dessert": 1,
    "Pizza":   2,
    "Ramen":   3,
    "Sushi":   4,
}
NUM_CLASSES = len(CLASS_INDICES)

# =========================
# LOAD CSV
# =========================
df = pd.read_csv(CSV_PATH)

# ดึงรูปทั้งหมดจาก Image 1 และ Image 2 พร้อม label จากคอลัมน์ Menu
img1_df = df[["Image 1", "Menu"]].rename(columns={"Image 1": "filename", "Menu": "food_type"})
img2_df = df[["Image 2", "Menu"]].rename(columns={"Image 2": "filename", "Menu": "food_type"})

all_images = pd.concat([img1_df, img2_df]).drop_duplicates(subset=["filename"]).reset_index(drop=True)

print(f"Total unique images: {len(all_images)}")
print(all_images["food_type"].value_counts())

# =========================
# LOAD IMAGE FUNCTION
# =========================
def load_image(img_name):
    img_path = os.path.join(IMG_PATH, img_name)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # normalize เหมือน inference
    return img

# =========================
# PREPARE DATA
# =========================
X, y = [], []

for _, row in all_images.iterrows():
    try:
        img = load_image(row["filename"])
        label = CLASS_INDICES[row["food_type"]]
        X.append(img)
        y.append(label)
    except Exception as e:
        print(f"Skip {row['filename']}: {e}")

X = np.array(X)
y = np.array(y)
y_onehot = tf.keras.utils.to_categorical(y, NUM_CLASSES)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Label distribution: {np.bincount(y)}")

# =========================
# MODEL
# =========================
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = True  # unfreeze ทั้งหมด เพื่อให้ overfit ง่าย

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=True)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dense(256, activation="relu")(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN (OVERFIT)
# =========================
history = model.fit(
    X, y_onehot,
    epochs=50,
    batch_size=16,
    shuffle=True,
)

# =========================
# REAL TRAIN ACC CHECK
# =========================
preds = model.predict(X)
pred_labels = np.argmax(preds, axis=1)

real_acc = np.mean(pred_labels == y)
print(f"\nREAL TRAIN ACC: {real_acc:.4f}")

# =========================
# SAVE
# =========================
model.save("food_class.keras")
print("Model saved to food_class.keras")
