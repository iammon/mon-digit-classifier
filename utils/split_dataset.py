import os
import shutil
import random
from pathlib import Path

# === CONFIG ===
SOURCE_DIR = "mon_segmented_raw"
TRAIN_DIR = "mon_digits/train"
TEST_DIR = "mon_digits/test"
SPLIT_RATIO = 0.8  # 80% train, 20% test

# === Ensure destination folders exist ===
digits = ['၀', '၁', '၂', '၃', '၄', '၅', '၆', '၇', '၈', '၉']
for d in digits:
    os.makedirs(os.path.join(TRAIN_DIR, d), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, d), exist_ok=True)

# === Process each digit folder ===
for d in digits:
    src_folder = os.path.join(SOURCE_DIR, d)
    train_folder = os.path.join(TRAIN_DIR, d)
    test_folder = os.path.join(TEST_DIR, d)

    # Get all .png files and shuffle
    all_files = [f for f in os.listdir(src_folder) if f.endswith(".png")]
    random.shuffle(all_files)

    split_idx = int(len(all_files) * SPLIT_RATIO)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]

    # Copy files
    for f in train_files:
        shutil.copy(os.path.join(src_folder, f), os.path.join(train_folder, f))

    for f in test_files:
        shutil.copy(os.path.join(src_folder, f), os.path.join(test_folder, f))

    print(f"[✓] {d}: {len(train_files)} train | {len(test_files)} test")

print("\n🎉 All files split successfully!")
