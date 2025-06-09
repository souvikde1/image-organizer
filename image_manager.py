import os
import shutil
import cv2
import numpy as np
import time
import pickle
import random
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics.pairwise import cosine_similarity

# ——— Helpers ———

def get_encoder():
    return MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(128, 128, 3))

def preprocess_image(img_path, target_size=(128, 128)):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Cannot read {img_path}. Skipping...")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return (img / 127.5) - 1


def get_feature_vector(model, img_array):
    batch = np.expand_dims(img_array, axis=0)
    vec = model.predict(batch, verbose=0)
    return vec.flatten()

def get_unique_filename(folder, filename):
    base, ext = os.path.splitext(filename)
    new = filename
    cnt = 1
    while os.path.exists(os.path.join(folder, new)):
        new = f"{base}_{cnt}{ext}"
        cnt += 1
    return new

def build_feature_db(model, root_folder, max_per_folder=15, cache_path=None):
    db = {}
    for sub in os.listdir(root_folder):
        fp = os.path.join(root_folder, sub)
        if not os.path.isdir(fp):
            continue

        if os.path.basename(fp).lower() == 'unidentified_images':
            continue 

        # List all valid images
        images = [fn for fn in os.listdir(fp) if fn.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not images:
            db[fp] = []
            continue

        # Random shuffle
        random.shuffle(images)
        selected_images = images[:max_per_folder]

        # Batch processing
        img_arrays = []
        for fn in selected_images:
            img_path = os.path.join(fp, fn)
            img = preprocess_image(img_path)
            if img is None:
                continue  # Skip bad files

            img_arrays.append(img)

        img_arrays = np.array(img_arrays)

        # Predict features in batch
        preds = model.predict(img_arrays, verbose=0)
        vecs = [vec.flatten() for vec in preds]

        db[fp] = vecs

    if cache_path:
        with open(cache_path, 'wb') as f:
            pickle.dump(db, f)

    return db

# ——— Main ———

query_path      = r"D:\project random pic\pic"
storing_root    = r"D:\project random pic\managed_image"
unknown_folder  = os.path.join(storing_root, "unidentified_images")
threshold       = 0.25
cache_file      = os.path.join(storing_root, "feature_db.pkl")
max_per_folder  = 27
sleep_time      = 10  # seconds between checks
auto_update = True



encoder = get_encoder()
os.makedirs(unknown_folder, exist_ok=True)


while True:
    # Load or build feature database
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            feature_db = pickle.load(f)
    else:
        feature_db = build_feature_db(encoder, storing_root, max_per_folder, cache_path=cache_file)

    moved_any_image = False

    # Move all images from query_path
    for fn in os.listdir(query_path):
        if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        src = os.path.join(query_path, fn)
        img = preprocess_image(src)
        if img is None:
            continue
        qvec = get_feature_vector(encoder, img)

        # Find best match
        best_folder, best_sim = None, -1
        for folder, vecs in feature_db.items():
            if not vecs:
                continue
            sim = cosine_similarity([qvec], vecs)[0].mean()
            if sim > best_sim:
                best_sim = sim
                best_folder = folder

        # Decide destination
        dest = best_folder if best_sim >= threshold else unknown_folder

        new_name = get_unique_filename(dest, fn)
        shutil.move(src, os.path.join(dest, new_name))
        print(f"Moved {fn} → {os.path.basename(dest)}  (sim={best_sim:.3f})")
        moved_any_image = True

    # Rebuild feature database after moving all
    if moved_any_image and auto_update :
        print("\nUpdating feature database...")
        feature_db = build_feature_db(encoder, storing_root, max_per_folder, cache_path=cache_file)
        print("Feature database updated!\n")

    time.sleep(sleep_time)
