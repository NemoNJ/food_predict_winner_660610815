import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Constants
BATCH_SIZE = 16
IMAGE_SIZE = (224, 224)
EPOCHS = 50
BASE_DIR = "/content/drive/MyDrive/contestdatset/Photos/Intragram Images [Original]"

# Define training order and config
FOOD_CLASSES = [
    {"name": "Burger",  "save_as": "food_class_Burger.keras"},
    {"name": "Ramen",   "save_as": "food_class_Ramen.keras"},
    {"name": "Pizza",   "save_as": "food_class_Pizza.keras"},
    {"name": "Sushi",   "save_as": "food_class_Sushi.keras"},
    {"name": "Dessert", "save_as": "food_class_Dessert.keras"},
]

def build_model():
    """Build and compile a fresh MobileNet-based binary classifier."""
    base_model = MobileNet(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    base_model.trainable = False  # Freeze layers

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def get_generators(food_name):
    """Create train and validation data generators for a given food class."""
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2
    )

    train_generator = datagen.flow_from_directory(
        f"{BASE_DIR}/{food_name}/Train",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training"
    )

    val_generator = datagen.flow_from_directory(
        f"{BASE_DIR}/{food_name}/Validation",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation"
    )

    return train_generator, val_generator

def plot_history(history, food_name):
    """Plot and save training/validation loss curve."""
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title(f"Loss Curve — {food_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"loss_{food_name}.png")
    plt.show()
    print(f"[{food_name}] Loss plot saved as loss_{food_name}.png")

# ── Main Training Loop ──────────────────────────────────────────────────────────

all_histories = {}

for food in FOOD_CLASSES:
    name     = food["name"]
    save_as  = food["save_as"]

    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}\n")

    # Prepare data
    train_gen, val_gen = get_generators(name)

    # Build a fresh model for each class
    model = build_model()

    # Callbacks
    callbacks = [
        ModelCheckpoint(save_as, save_best_only=True, monitor="val_loss", mode="min"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ]

    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    all_histories[name] = history
    plot_history(history, name)

    print(f"\n[{name}] Model saved → {save_as}")

print(f"\n{'='*60}")
print("  All 5 models trained successfully!")
print(f"{'='*60}\n")
