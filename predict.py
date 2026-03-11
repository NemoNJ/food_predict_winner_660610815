# ==========================================================
# FOOD ATTRACTIVENESS RANKING TRAINING PIPELINE (FINAL)
# ==========================================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import Sequence
from PIL import Image

# ==========================================================
# CONFIG
# ==========================================================

import glob

DRIVE_ROOT = "/content/drive/MyDrive"
all_dirs = os.listdir(DRIVE_ROOT)

contest_folder = [d for d in all_dirs if "contest" in d.lower()][0]
BASE_PATH = os.path.join(DRIVE_ROOT, contest_folder)

print("BASE_PATH =", BASE_PATH)

CSV_INSTAGRAM = glob.glob(os.path.join(BASE_PATH, "*intragram*.csv"))[0]
CSV_QUESTIONNAIRE = glob.glob(os.path.join(BASE_PATH, "*question*.csv"))[0]

print("CSV_INSTAGRAM =", CSV_INSTAGRAM)
print("CSV_QUESTIONNAIRE =", CSV_QUESTIONNAIRE)

INSTAGRAM_FOLDER = glob.glob(os.path.join(BASE_PATH, "*Instagram*"))[0]
QUESTIONNAIRE_FOLDER = glob.glob(os.path.join(BASE_PATH, "*Question*"))[0]

print("INSTAGRAM_FOLDER =", INSTAGRAM_FOLDER)
print("QUESTIONNAIRE_FOLDER =", QUESTIONNAIRE_FOLDER)

TESTER_CSV = os.path.join(BASE_PATH, "tester.csv")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 17

FOOD_CLASSES = ["Burger", "Pizza", "Sushi", "Ramen", "Dessert"]

# ==========================================================
# STEP 0: SPLIT TESTER DATA (ก่อน merge)
# ==========================================================

def split_tester_data(csv_instagram, csv_questionnaire, tester_csv_path,
                      n_instagram=8, n_questionnaire=32, random_state=42):
    """
    แบ่งข้อมูลออกเป็น tester set ก่อน training
    - Instagram  : 8  แถว
    - Questionnaire : 32 แถว
    รวม tester.csv = 40 แถว
    คืนค่า df_instagram และ df_questionnaire ที่ตัด tester ออกแล้ว
    """

    df_ig  = pd.read_csv(csv_instagram)
    df_qs  = pd.read_csv(csv_questionnaire)

    # ── ตรวจว่ามีข้อมูลพอแบ่งไหม ──────────────────────────
    if len(df_ig) < n_instagram:
        raise ValueError(
            f"Instagram CSV มีแค่ {len(df_ig)} แถว "
            f"ไม่พอแบ่ง tester {n_instagram} แถว"
        )
    if len(df_qs) < n_questionnaire:
        raise ValueError(
            f"Questionnaire CSV มีแค่ {len(df_qs)} แถว "
            f"ไม่พอแบ่ง tester {n_questionnaire} แถว"
        )

    # ── สุ่มแบ่ง tester ────────────────────────────────────
    tester_ig,  train_ig  = train_test_split(
        df_ig,  test_size=(len(df_ig)  - n_instagram)  / len(df_ig),
        random_state=random_state
    )
    tester_qs,  train_qs  = train_test_split(
        df_qs,  test_size=(len(df_qs)  - n_questionnaire) / len(df_qs),
        random_state=random_state
    )

    # ── แท็กแหล่งที่มา (optional แต่มีประโยชน์) ────────────
    tester_ig["source"]  = "instagram"
    tester_qs["source"]  = "questionnaire"

    # ── รวมและบันทึก tester.csv ────────────────────────────
    tester_df = pd.concat([tester_ig, tester_qs], ignore_index=True)
    tester_df.to_csv(tester_csv_path, index=False)

    print(f"\n{'='*50}")
    print(f"TESTER SET SUMMARY")
    print(f"{'='*50}")
    print(f"  Instagram   : {len(tester_ig):>4} แถว")
    print(f"  Questionnaire: {len(tester_qs):>4} แถว")
    print(f"  รวม tester  : {len(tester_df):>4} แถว")
    print(f"  บันทึกที่   : {tester_csv_path}")
    print(f"{'='*50}\n")

    print(f"TRAINING SET (หลังตัด tester ออก)")
    print(f"  Instagram   : {len(train_ig):>4} แถว")
    print(f"  Questionnaire: {len(train_qs):>4} แถว")
    print(f"{'='*50}\n")

    return train_ig, train_qs

# ==========================================================
# STEP 1: MERGE DATASETS
# ==========================================================

def merge_datasets(df_instagram, df_questionnaire):
    """
    รับ DataFrame ที่ตัด tester ออกแล้วมา merge
    """
    df = pd.concat([df_instagram, df_questionnaire], ignore_index=True)
    df = df.drop_duplicates()
    print(df["Winner"].unique())
    print(f"Total merged pairs: {len(df)}")
    return df

# ==========================================================
# STEP 2: VALIDATE FOOD TYPE
# ==========================================================

def validate_food_types(df):

    df.columns = df.columns.str.strip()

    rename_map = {
        "Menu": "Food_Type",
        "Image 1": "Image1",
        "Image 2": "Image2"
    }
    df = df.rename(columns=rename_map)

    required_cols = ["Image1", "Image2", "Food_Type", "Winner"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    invalid = set(df["Food_Type"].unique()) - set(FOOD_CLASSES)
    if len(invalid) > 0:
        raise ValueError(f"Invalid classes found: {invalid}")

    print("Food types validated.")
    return df

# ==========================================================
# PATH RESOLVER
# ==========================================================

def resolve_image_path(row, column_name):

    filename = str(row[column_name]).strip()
    food_type = row["Food_Type"]

    if not filename.lower().endswith(".jpg"):
        filename = filename + ".jpg"

    insta_path = os.path.join(INSTAGRAM_FOLDER, food_type, filename)
    ques_path  = os.path.join(QUESTIONNAIRE_FOLDER, filename)

    if os.path.exists(insta_path):
        return insta_path
    elif os.path.exists(ques_path):
        return ques_path
    else:
        raise FileNotFoundError(f"Image not found: {filename}")

# ==========================================================
# IMAGE LOADER
# ==========================================================

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img = np.array(img)
    img = preprocess_input(img)
    return img

# ==========================================================
# DATA GENERATOR
# ==========================================================

class SiameseGenerator(Sequence):

    def __init__(self, df, batch_size=16, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.df         = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.indexes    = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return len(self.df) // self.batch_size

    def __getitem__(self, index):

        batch_idx = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_df  = self.df.iloc[batch_idx]

        imgs1, imgs2, labels = [], [], []

        for _, row in batch_df.iterrows():

            path1 = resolve_image_path(row, "Image1")
            path2 = resolve_image_path(row, "Image2")

            img1  = load_image(path1)
            img2  = load_image(path2)

            winner = int(row["Winner"])

            if winner == 1:
                label = 1.0
            elif winner == 2:
                label = 0.0
            else:
                raise ValueError(f"Invalid Winner value: {winner}")

            imgs1.append(img1)
            imgs2.append(img2)
            labels.append(label)

        return (
            np.array(imgs1, dtype=np.float32),
            np.array(imgs2, dtype=np.float32)
        ), np.array(labels, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# ==========================================================
# BUILD SIAMESE MODEL
# ==========================================================

def build_siamese_model():

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    input_a = layers.Input(shape=(224, 224, 3))
    input_b = layers.Input(shape=(224, 224, 3))

    feat_a  = base_model(input_a, training=False)
    feat_b  = base_model(input_b, training=False)

    feat_a  = layers.GlobalAveragePooling2D()(feat_a)
    feat_b  = layers.GlobalAveragePooling2D()(feat_b)

    diff    = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([feat_a, feat_b])

    x       = layers.Dense(128, activation="relu")(diff)
    x       = layers.Dropout(0.3)(x)
    output  = layers.Dense(1, activation="sigmoid")(x)

    model   = models.Model(inputs=[input_a, input_b], outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ==========================================================
# TRAIN PER CLASS
# ==========================================================

def train_class_model(food_class, df):

    print(f"\nTraining class: {food_class}")

    class_df = df[df["Food_Type"] == food_class]

    if len(class_df) < 10:
        print("Not enough data. Skipping.")
        return

    train_df, val_df = train_test_split(
        class_df,
        test_size=0.2,
        random_state=42
    )

    train_gen = SiameseGenerator(train_df, BATCH_SIZE)
    val_gen   = SiameseGenerator(val_df,   BATCH_SIZE)

    model = build_siamese_model()

    checkpoint = ModelCheckpoint(
        f"/content/drive/MyDrive/food_compare_{food_class}.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop]
    )

    print(f"Saved model: food_compare_{food_class}.keras")

# ==========================================================
# MAIN
# ==========================================================

def main():

    print("FOOD ATTRACTIVENESS TRAINING PIPELINE START")

    # ── STEP 0: แบ่ง tester ก่อนทุกอย่าง ──────────────────
    train_ig, train_qs = split_tester_data(
        csv_instagram     = CSV_INSTAGRAM,
        csv_questionnaire = CSV_QUESTIONNAIRE,
        tester_csv_path   = TESTER_CSV,
        n_instagram       = 8,
        n_questionnaire   = 32,
        random_state      = 42
    )

    # ── STEP 1: merge เฉพาะ training data ─────────────────
    df = merge_datasets(train_ig, train_qs)

    # ── STEP 2: validate ───────────────────────────────────
    df = validate_food_types(df)

    # ── STEP 3: train per class ────────────────────────────
    for food_class in FOOD_CLASSES:
        train_class_model(food_class, df)

    print("\nALL TRAINING COMPLETE")
    print(f"Tester set saved at: {TESTER_CSV}")

if __name__ == "__main__":
    main()
