import os
import shutil

BASE_DIR = "/content/drive/MyDrive/contestdatset/Photos/Intragram Images [Original]"
FOOD_CLASSES = ["Burger", "Ramen", "Pizza", "Sushi", "Dessert"]
SPLITS = ["Train", "Validation"]

def get_images(folder_path):
    """Return list of image file paths in a folder (non-recursive)."""
    exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    if not os.path.exists(folder_path):
        return []
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(exts)
    ]

for food in FOOD_CLASSES:
    non_label = f"Non_{food}"
    others = [c for c in FOOD_CLASSES if c != food]

    for split in SPLITS:
        # ── Paths ──────────────────────────────────────────────
        pos_src  = os.path.join(BASE_DIR, food, split)          # รูปของ class นั้น
        pos_dst  = os.path.join(BASE_DIR, food, split, food)    # subfolder class
        neg_dst  = os.path.join(BASE_DIR, food, split, non_label)  # subfolder non-class

        # ── สร้าง subfolder ────────────────────────────────────
        os.makedirs(pos_dst, exist_ok=True)
        os.makedirs(neg_dst, exist_ok=True)

        # ── ย้ายรูป positive (ที่อยู่ตรงๆ ใน split folder) ────
        images_in_src = get_images(pos_src)
        moved_pos = 0
        for img_path in images_in_src:
            fname = os.path.basename(img_path)
            dst_path = os.path.join(pos_dst, fname)
            if not os.path.exists(dst_path):
                shutil.move(img_path, dst_path)
                moved_pos += 1

        # ── copy รูป negative จาก class อื่น ──────────────────
        copied_neg = 0
        for other in others:
            other_split_path = os.path.join(BASE_DIR, other, split)

            # ดึงรูปจาก subfolder ของ other (ถ้ามี) หรือจาก root
            other_subfolders = [
                os.path.join(other_split_path, d)
                for d in os.listdir(other_split_path)
                if os.path.isdir(os.path.join(other_split_path, d))
            ] if os.path.exists(other_split_path) else []

            src_paths = []
            if other_subfolders:
                for sf in other_subfolders:
                    src_paths += get_images(sf)
            else:
                src_paths = get_images(other_split_path)

            for img_path in src_paths:
                fname = f"{other}_{os.path.basename(img_path)}"  # prefix ป้องกันชื่อซ้ำ
                dst_path = os.path.join(neg_dst, fname)
                if not os.path.exists(dst_path):
                    shutil.copy2(img_path, dst_path)
                    copied_neg += 1

        print(f"✅ {food}/{split} → {food}/: {moved_pos} imgs | {non_label}/: {copied_neg} imgs")

print("\n🎉 Dataset structure ready!")
print("โครงสร้างตัวอย่าง:")
print(f"  Burger/Train/Burger/       ← รูป burger สำหรับ train")
print(f"  Burger/Train/Non_Burger/   ← รูป ramen+pizza+sushi+dessert สำหรับ train")
print(f"  Burger/Validation/Burger/")
print(f"  Burger/Validation/Non_Burger/")
