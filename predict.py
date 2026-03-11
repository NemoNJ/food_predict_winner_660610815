import keras
keras.config.enable_unsafe_deserialization()

from keras.models import load_model
import numpy as np
import pandas as pd
import cv2
from tabulate import tabulate
import os

FOLDER_PATH      = "Test Images"
CSV_OUTPUT_PATH  = "test.csv"

IMAGE_SIZE_CLASS = (128, 128)
IMAGE_SIZE_PAIR  = (224, 224)

CLASS_INDICES = {0: "Burger", 1: "Dessert", 2: "Pizza", 3: "Ramen", 4: "Sushi"}


QUALITY_THRESHOLD    = 0.2
QUALITY_SIGMOID_HIGH = 0.6
QUALITY_SIGMOID_LOW  = 0.4


print("Loading base models...")
classify_model = load_model("food_class.keras")
print("Base models loaded.\n")

quality_models = {}
pair_models    = {}


def get_model_input_count(model):
    inp = model.input
    if isinstance(inp, list):
        return len(inp)
    return 1


def load_image(image_path, size):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)


def step1_and_2_classify_food(img1_path, img2_path):
    img1 = load_image(img1_path, IMAGE_SIZE_CLASS)
    img2 = load_image(img2_path, IMAGE_SIZE_CLASS)

    pred1 = classify_model.predict(img1, verbose=0)[0]
    pred2 = classify_model.predict(img2, verbose=0)[0]

    idx1, conf1 = int(np.argmax(pred1)), float(np.max(pred1)) * 100
    idx2, conf2 = int(np.argmax(pred2)), float(np.max(pred2)) * 100

    food1 = CLASS_INDICES.get(idx1, "Unknown")
    food2 = CLASS_INDICES.get(idx2, "Unknown")
    food_type = food1 if conf1 >= conf2 else food2

    return food_type, food1, conf1, food2, conf2


def step3_quality(img1_path, img2_path, food_type):
    global quality_models
    if food_type not in quality_models:
        model_path = f"food_class_{food_type}.keras"
        print(f"  [Load] Quality model: {model_path}")
        quality_models[food_type] = load_model(model_path)

    model      = quality_models[food_type]
    num_inputs = get_model_input_count(model)
    print(f"  [Step3] {food_type} model → {num_inputs} input(s)")

    img1 = load_image(img1_path, IMAGE_SIZE_PAIR)
    img2 = load_image(img2_path, IMAGE_SIZE_PAIR)

    if num_inputs == 1:
        score1 = float(model.predict(img1, verbose=0)[0][0])
        score2 = float(model.predict(img2, verbose=0)[0][0])
        diff   = abs(score1 - score2)
        print(f"  [Step3] score1={score1:.4f}  score2={score2:.4f}  diff={diff:.4f}")
        if diff > QUALITY_THRESHOLD:
            winner = 1 if score1 > score2 else 2
            return winner, {"score1": score1, "score2": score2}, True
        else:
            return None, {"score1": score1, "score2": score2}, False

    elif num_inputs == 2:
        p_forward  = float(model.predict([img1, img2], verbose=0)[0][0])
        p_backward = float(model.predict([img2, img1], verbose=0)[0][0])
        p_avg = (p_forward + (1.0 - p_backward)) / 2.0

        print(f"  [Step3] forward={p_forward:.4f}  backward={p_backward:.4f}  avg={p_avg:.4f}")

        if p_avg > QUALITY_SIGMOID_HIGH:
            return 2, {"p": p_avg, "p_forward": p_forward, "p_backward": p_backward}, True
        elif p_avg < QUALITY_SIGMOID_LOW:
            return 1, {"p": p_avg, "p_forward": p_forward, "p_backward": p_backward}, True
        else:
            return None, {"p": p_avg, "step3_p_fallback": p_avg,
                          "p_forward": p_forward, "p_backward": p_backward}, False

    else:
        print(f"  [Step3] Unknown input count ({num_inputs}) → skip to Step 4")
        return None, {}, False

def step4_pair_model(img1_path, img2_path, food_type):
    global pair_models
    if food_type not in pair_models:
        candidate_paths = [
            f"food_compare_{food_type} (2).keras",
            f"food_compare_{food_type} (2).h5",
        ]
        loaded = False
        for model_path in candidate_paths:
            if not os.path.exists(model_path):
                continue
            try:
                print(f"  [Load] Pair model: {model_path}")
                pair_models[food_type] = load_model(model_path, compile=False, safe_mode=False)
                loaded = True
                break
            except Exception as e:
                print(f"  [Warn] Failed to load {model_path}: {e}")

        if not loaded:
            raise FileNotFoundError(
                f"Cannot load  pair model for {food_type} "
                f"(tried: {', '.join(candidate_paths)})"
            )

    model = pair_models[food_type]
    img1  = load_image(img1_path, IMAGE_SIZE_PAIR)
    img2  = load_image(img2_path, IMAGE_SIZE_PAIR)

    p_forward  = float(model.predict([img1, img2], verbose=0)[0][0])
    p_backward = float(model.predict([img2, img1], verbose=0)[0][0])
    p_avg  = (p_forward + (1.0 - p_backward)) / 2.0
    winner = 2 if p_avg > 0.5 else 1

    print(f"  [Step4] forward={p_forward:.4f}  backward={p_backward:.4f}  avg={p_avg:.4f} → Image {winner}")
    return winner, p_avg


def predict_winner(img1_path, img2_path):
    detail = {}

    food_type, food1, conf1, food2, conf2 = step1_and_2_classify_food(img1_path, img2_path)
    detail.update({
        "food_type": food_type,
        "food1": food1, "conf1": round(conf1, 2),
        "food2": food2, "conf2": round(conf2, 2),
    })
    print(f"  [Step1+2] {food1}({conf1:.1f}%) vs {food2}({conf2:.1f}%) → food_type={food_type}")

    winner, q_info, quality_decided = step3_quality(img1_path, img2_path, food_type)
    detail["quality_info"] = q_info

    if quality_decided:
        detail["decided_at"] = "Step3_Quality"
        return winner, detail

    try:
        winner, pair_p = step4_pair_model(img1_path, img2_path, food_type)
        detail["pair_p"]     = round(pair_p, 4)
        detail["decided_at"] = "Step4_Pair"
    except Exception as e:
        fallback_p = q_info.get("step3_p_fallback")
        if fallback_p is not None:
            winner = 2 if fallback_p > 0.5 else 1
            print(f"  [Fallback] Step4 failed ({e}) → ใช้ Step3 avg p={fallback_p:.4f} → Image {winner}")
            detail["decided_at"] = "Step3_Fallback(Step4_Failed)"
        else:
            print(f"  [Fallback] Step4 failed ({e}) → ไม่มี p fallback → default winner = 1")
            winner = 1
            detail["decided_at"] = "Default_Fallback(Step4_Failed)"

    return winner, detail


if __name__ == "__main__":

    df = pd.read_csv(CSV_OUTPUT_PATH, delimiter=",", header=0)

    predictions = []
    table_data  = []

    for i, row in df.iterrows():
        img1_name = str(row["Image 1"]).strip() if pd.notna(row["Image 1"]) else ""
        img2_name = str(row["Image 2"]).strip() if pd.notna(row["Image 2"]) else ""

        img1_path = os.path.join(FOLDER_PATH, img1_name) if img1_name else ""
        img2_path = os.path.join(FOLDER_PATH, img2_name) if img2_name else ""

        print(f"\n{'─'*60}")
        print(f"[{i+1}/{len(df)}] {img1_name}  vs  {img2_name}")

        if not img2_name or not os.path.exists(img2_path):
            print("  ⚠️  Image 2 missing → default winner = 1")
            winner = 1
            detail = {"decided_at": "Default(No_Img2)", "food_type": "-",
                      "food1": "-", "conf1": "-", "food2": "-", "conf2": "-"}
        elif not img1_name or not os.path.exists(img1_path):
            print("  ⚠️  Image 1 missing → default winner = 2")
            winner = 2
            detail = {"decided_at": "Default(No_Img1)", "food_type": "-",
                      "food1": "-", "conf1": "-", "food2": "-", "conf2": "-"}
        else:
            try:
                winner, detail = predict_winner(img1_path, img2_path)
            except Exception as e:
                print(f"  ❌ ERROR: {e}")
                winner = -1
                detail = {"decided_at": f"ERROR: {e}", "food_type": "-",
                          "food1": "-", "conf1": "-", "food2": "-", "conf2": "-"}

        predictions.append(winner)

        food1_str = (f"{detail.get('food1','-')} ({detail.get('conf1',0):.1f}%)"
                     if isinstance(detail.get("conf1"), float) else "-")
        food2_str = (f"{detail.get('food2','-')} ({detail.get('conf2',0):.1f}%)"
                     if isinstance(detail.get("conf2"), float) else "-")

        table_data.append([
            img1_name, food1_str,
            img2_name, food2_str,
            f"Image {winner}",
            detail.get("food_type", "-"),
            detail.get("decided_at", "-"),
        ])

    df["Winner"] = predictions
    try:
        df.to_csv(CSV_OUTPUT_PATH, index=False)
        print(f"\n✅ Saved predictions → {CSV_OUTPUT_PATH}")
    except PermissionError:
        fallback = CSV_OUTPUT_PATH.replace(".csv", "_result.csv")
        df.to_csv(fallback, index=False)
        print(f"\n✅ Permission denied → Saved to {fallback}")

    headers = ["Image 1", "Food1 (conf%)", "Image 2", "Food2 (conf%)",
               "Predicted", "Food Type", "Decided At"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
