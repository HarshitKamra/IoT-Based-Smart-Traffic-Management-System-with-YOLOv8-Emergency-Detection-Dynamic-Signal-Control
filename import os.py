import os
import random
import shutil

# ✅ Set your dataset base path
dataset_dir = r"C:\Users\HP\OneDrive\Desktop\ambulance_dataset_v2"

# 📁 Update paths based on your structure
train_img_dir = os.path.join(dataset_dir, "train/images")
train_label_dir = os.path.join(dataset_dir, "train/labels")
val_img_dir = os.path.join(dataset_dir, "valid/images")
val_label_dir = os.path.join(dataset_dir, "valid/labels")

# 🔢 Number of images to copy
num_to_copy = 1000  # You can change this number

# 🔥 Step 1: Clear the existing files in val folders
for folder in [val_img_dir, val_label_dir]:
    for file in os.listdir(folder):
        os.remove(os.path.join(folder, file))
print("Cleared existing validation images and labels.")

# 📋 Step 2: Get list of train images
image_files = [f for f in os.listdir(train_img_dir) if f.endswith((".jpg", ".png"))]
selected_files = random.sample(image_files, min(num_to_copy, len(image_files)))

# 📤 Step 3: Copy selected image-label pairs
copied_count = 0
for fname in selected_files:
    img_src = os.path.join(train_img_dir, fname)
    label_name = os.path.splitext(fname)[0] + ".txt"
    label_src = os.path.join(train_label_dir, label_name)

    img_dst = os.path.join(val_img_dir, fname)
    label_dst = os.path.join(val_label_dir, label_name)

    if os.path.exists(label_src):
        shutil.copy(img_src, img_dst)
        shutil.copy(label_src, label_dst)
        copied_count += 1
    else:
        print(f"Skipped: No label found for {fname}")

print(f"Copied {copied_count} image(s) with labels from TRAIN to VALID.")
