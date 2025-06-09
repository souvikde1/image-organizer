# 🧠 Intelligent Image Organizer

Automatically sorts and moves images into relevant folders based on visual similarity using deep learning (MobileNetV2). Unrecognized images are sent to a special folder for review.

---

## 📂 Folder Structure

```plaintext
project/
│
├── pic/                    # Incoming unorganized images (query_path)
├── managed_image/         # Destination root where images are sorted
│   ├── class1/            # Class folders (optional pre-created)
│   └── unidentified_images/ # Auto-created for unmatched images
```

🚀 How It Works
Loads all incoming images from query_path
Extracts features using pretrained MobileNetV2
Compares each image to folders in managed_image/ using cosine similarity
Moves it to the best match folder (if similarity ≥ 0.25), else to unidentified_images/
Updates feature database after each iteration if auto_update = True

⚙️ Configuration (in main.py)
| Parameter        | Description                              |
| ---------------- | ---------------------------------------- |
| `query_path`     | Folder containing unorganized images     |
| `storing_root`   | Root directory with class folders        |
| `threshold`      | Cosine similarity threshold for matching |
| `sleep_time`     | Seconds to wait before next scan         |
| `auto_update`    | Rebuild feature DB after each run        |
| `max_per_folder` | Max samples per folder for comparison    |

📦 Dependencies
Install the requirements:
```
pip install -r requirements.txt
```
▶️ Run
```
python main.py
```
Leave it running. It continuously scans and processes new images every sleep_time seconds.

📁 Notes
The feature database is cached in feature_db.pkl
Images that cannot be classified are moved to unidentified_images/
All processed images are renamed uniquely to avoid overwriting

🛠️ Tech Stack
TensorFlow / Keras
OpenCV
NumPy
Cosine Similarity (scikit-learn)
