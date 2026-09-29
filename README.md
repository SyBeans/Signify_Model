Signify — Training Environment Setup Guide (UPDATED)
📋 Overview

Signify requires 4 separate virtual environments because MediaPipe and TensorFlow conflict on protobuf versions:
Env	Purpose	Key Packages
kaggle_datasets	Download Kaggle datasets (FER2013 CSV)	kaggle, numpy, Pillow
venv_extract	Extract landmarks (both FSL hand + FER+ face)	MediaPipe 0.10.13, protobuf 4.25.9
venv_train	Train models	TensorFlow 2.16.2
venv_test	Test with webcam	Both mediapipe + tensorflow
🔧 Step 1: Create Virtual Environments
bash

cd ~/Signify/Signify_Model

# 1. Kaggle download environment
python3.12 -m venv kaggle_dl/kaggle_datasets

# 2. Extraction environment
python3.12 -m venv venv_extract

# 3. Training environment
python3.12 -m venv venv_train

# 4. Testing environment
python3.12 -m venv venv_test

⚠️ Use Python 3.12 — MediaPipe 0.10.13 doesn't work with Python 3.13.
📦 Step 2: Install Dependencies
requirements/requirements_kaggle.txt (minimal):
txt

kaggle
numpy==1.26.4
Pillow

requirements/requirements_extract.txt:
txt

numpy==1.26.4
pandas
opencv-python
mediapipe==0.10.13
protobuf==4.25.9
tqdm

requirements/requirements_train.txt:
txt

tensorflow==2.16.2
numpy==1.26.4
pandas
matplotlib
seaborn
scikit-learn
tqdm

requirements/requirements_test.txt:
txt

numpy==1.26.4
opencv-python
mediapipe==0.10.13
protobuf==4.25.9
tensorflow==2.16.2
pandas
scikit-learn

Install Commands (one env at a time):
bash

# Kaggle
source kaggle_dl/kaggle_datasets/bin/activate
pip install -r requirements/requirements_kaggle.txt
deactivate

# Extraction
source venv_extract/bin/activate
pip install -r requirements/requirements_extract.txt
deactivate

# Training
source venv_train/bin/activate
pip install -r requirements/requirements_train.txt
deactivate

# Testing
source venv_test/bin/activate
pip install -r requirements/requirements_test.txt
deactivate

🎬 Step 3: Activate/Deactivate Environments
Task	Linux/Mac	Windows
Activate	source <venv>/bin/activate	<venv>\Scripts\activate
Deactivate	deactivate	deactivate

Rule: Only ONE venv can be active at a time. Always deactivate before switching.
🔑 Step 4: Kaggle API Setup
4.1: Get API Token

    Go to: https://www.kaggle.com/settings

    Scroll to API section

    Click "Create New Token"

    Downloads kaggle.json

4.2: Move Token
bash

mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Verify
cat ~/.kaggle/kaggle.json

4.3: Verify
bash

source kaggle_dl/kaggle_datasets/bin/activate
kaggle datasets list -s fer2013

📥 Downloading Datasets
Dataset 1: FSL-105 (Hand Signs)
Option A: Manual (Recommended for team)

    Go to: https://data.mendeley.com/datasets/48y2y99mb9/2

    Click "Download All" → extract to datasets/FSL/

Option B: Hugging Face
python

from datasets import load_dataset
dset = load_dataset("SEACrowd/fsl_105", trust_remote_code=True)

Expected Structure:
text

datasets/FSL/
├── clips/         (2,130 .MOV files, 105 folders)
├── labels.csv
├── train.csv
└── test.csv

Dataset 2: FER+ (Facial Emotions)
Step 2.1: Download FER2013 CSV (from Kaggle)
bash

cd ~/Signify/Signify_Model/datasets
mkdir -p FERPLUS
cd FERPLUS

# Activate kaggle venv
source ~/Signify/Signify_Model/kaggle_dl/kaggle_datasets/bin/activate

# Download FER2013 (CSV version)
kaggle datasets download -d msambare/fer2013
unzip -o fer2013.zip

# If it gives PNG folders instead of CSV, use deadskull7 instead:
kaggle datasets download -d deadskull7/fer2013
unzip -o fer2013.zip -d csv_version
cp csv_version/fer2013.csv .

Should show: fer2013.csv (~288 MB)
Step 2.2: Download FER+ Labels
bash

cd ~/Signify/Signify_Model/datasets/FERPLUS
wget https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv

Should show: fer2013new.csv (~1.5 MB)
Step 2.3: Download Merge Script
bash

wget https://raw.githubusercontent.com/microsoft/FERPlus/master/src/generate_training_data.py

Step 2.4: Install Merge Dependencies
bash

# Still in kaggle_datasets venv (has numpy + Pillow)
pip install numpy Pillow

Step 2.5: Run Merge
bash

python generate_training_data.py -d FER2013Plus -fer fer2013.csv -ferplus fer2013new.csv

Output:
text

Start generating ferplus images.
...
FER2013Plus/
├── FER2013Train/
├── FER2013Valid/
└── FER2013Test/

Time: ~3-5 min for ~35,887 images
Step 2.6: Generate Label Files

The merge script doesn't include labels.csv, so we create them:
bash

python3 << 'EOF'
import csv

EMOTIONS = ['neutral', 'happiness', 'surprise', 'sadness', 
            'anger', 'disgust', 'fear', 'contempt']

with open("fer2013new.csv", "r") as f:
    reader = csv.DictReader(f)
    labels = list(reader)

train_labels, valid_labels, test_labels = [], [], []
skipped = 0

for row in labels:
    usage = row['Usage']
    image_name = row['Image name']
    votes = {e: int(row[e]) for e in EMOTIONS}
    
    max_emotion = max(votes, key=votes.get)
    max_count = votes[max_emotion]
    
    if max_count == 0:
        skipped += 1
        continue
    
    entry = (image_name, max_emotion)
    
    if usage == 'Training':
        train_labels.append(entry)
    elif usage == 'PublicTest':
        valid_labels.append(entry)
    elif usage == 'PrivateTest':
        test_labels.append(entry)

print(f"Train: {len(train_labels)}")
print(f"Valid: {len(valid_labels)}")
print(f"Test:  {len(test_labels)}")
print(f"Skipped: {skipped}")

def save(lbls, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'emotion'])
        w.writerows(lbls)

save(train_labels, "FER2013Plus/FER2013Train/labels.csv")
save(valid_labels, "FER2013Plus/FER2013Valid/labels.csv")
save(test_labels, "FER2013Plus/FER2013Test/labels.csv")
print("✅ Labels saved")
EOF

Step 2.7: Verify FER+
bash

cd ~/Signify/Signify_Model/datasets/FERPLUS/FER2013Plus

echo "Train: $(find FER2013Train -name '*.png' | wc -l)"
echo "Valid: $(find FER2013Valid -name '*.png' | wc -l)"
echo "Test:  $(find FER2013Test -name '*.png' | wc -l)"

ls FER2013Train/labels.csv
head -3 FER2013Train/labels.csv

Expected:
text

Train: ~28,558
Valid: ~3,579
Test:  ~3,573
labels.csv
filename,emotion
fer0000000.png,neutral
...

Step 2.8: Deactivate Kaggle Venv
bash

deactivate

🚀 Extraction & Training Workflow
Step 5: Extract Hand Landmarks (FSL-105)
bash

cd ~/Signify/Signify_Model
source venv_extract/bin/activate

python extract/extract_landmarks_twohands.py

# Output: landmarks/hand/
#   X_train.npy (1703, 30, 252)
#   y_train.npy (1703,)
#   X_test.npy  (426, 30, 252)
#   y_test.npy  (426,)

deactivate

Time: ~30-60 min
Step 6: Extract Face Landmarks (FER+)
bash

cd ~/Signify/Signify_Model
source venv_extract/bin/activate

mkdir -p landmarks/face_ferplus
python extract/extract_facial_features_ferplus.py

# Output: landmarks/face_ferplus/
#   X_face_train.npy (~25,000, 1404)
#   y_face_train.npy (~25,000,)
#   X_face_val.npy   (~3,100, 1404)
#   y_face_val.npy   (~3,100,)
#   X_face_test.npy  (~3,100, 1404)
#   y_face_test.npy  (~3,100,)

deactivate

Time: ~10-15 min
Expected valid: ~85-90% (skips no-face + contempt)
Step 7: Train Sign Model
bash

cd ~/Signify/Signify_Model
source venv_train/bin/activate

python train/train_model_v3_motion.py

# Output: models/hand/sign_model_motion.h5
# Expected: 81-91% accuracy

deactivate

Time: ~1-2 hours (with augmentation)
Step 8: Train Emotion Model (FER+)
bash

cd ~/Signify/Signify_Model
source venv_train/bin/activate

python train/train_emotion_model_ferplus.py

# Output: models/face_ferplus/emotion_model.h5
# Expected: 65-70% accuracy

deactivate

Time: ~30-60 min
Step 9: Test with Webcam
bash

cd ~/Signify/Signify_Model
source venv_test/bin/activate

# Test sign recognition
python test/test_webcam.py

# Test emotion recognition
python test/test_webcam_face.py

deactivate

📊 Complete Summary Table
Task	Venv	Script	Input	Output	Time
Download FSL-105	any	manual	Mendeley URL	datasets/FSL/	~10 min
Download FER+	kaggle_datasets	kaggle + merge	Kaggle/GitHub	datasets/FERPLUS/FER2013Plus/	~15 min
Extract hand	venv_extract	extract_landmarks_twohands.py	FSL clips	landmarks/hand/	~30-60 min
Extract face	venv_extract	extract_facial_features_ferplus.py	FER+ PNGs	landmarks/face_ferplus/	~10-15 min
Train sign	venv_train	train_model_v3_motion.py	landmarks/hand/	models/hand/	~1-2 hr
Train emotion	venv_train	train_emotion_model_ferplus.py	landmarks/face_ferplus/	models/face_ferplus/	~30-60 min
Test webcam	venv_test	test_webcam.py	webcam	live predictions	—
⚠️ Key Rules

    Use Python 3.12 — NOT 3.13

    One venv at a time — deactivate before switching

    Kaggle venv — for downloading datasets

    Extract with venv_extract — has MediaPipe

    Train with venv_train — has TensorFlow

    Test with venv_test — has both

    Contempt is skipped — 7 emotions only (matches Signify docs)

    FER2013 CSV ≠ FER2013 folders — CSV is needed for FER+ merge

🎯 Directory Structure (Final)
text

Signify_Model/
├── datasets/
│   ├── FSL/                    # Hand signs
│   │   ├── clips/
│   │   ├── labels.csv
│   │   ├── train.csv
│   │   └── test.csv
│   └── FERPLUS/
│       └── FER2013Plus/
│           ├── FER2013Train/
│           │   ├── fer*.png
│           │   └── labels.csv
│           ├── FER2013Valid/
│           └── FER2013Test/
│
├── landmarks/
│   ├── hand/                   # FSL hand landmarks
│   │   ├── X_train.npy
│   │   ├── y_train.npy
│   │   ├── X_test.npy
│   │   └── y_test.npy
│   └── face_ferplus/           # FER+ face landmarks
│       ├── X_face_train.npy
│       ├── y_face_train.npy
│       ├── X_face_val.npy
│       ├── y_face_val.npy
│       ├── X_face_test.npy
│       └── y_face_test.npy
│
├── models/
│   ├── hand/
│   │   └── sign_model_motion.h5
│   └── face_ferplus/
│       └── emotion_model.h5
│
├── extract/
│   ├── extract_landmarks_twohands.py
│   └── extract_facial_features_ferplus.py
│
├── train/
│   ├── train_model_v3_motion.py
│   └── train_emotion_model_ferplus.py
│
├── test/
│   ├── test_webcam.py
│   └── test_webcam_face.py
│
├── requirements/
│   ├── requirements_kaggle.txt
│   ├── requirements_extract.txt
│   ├── requirements_train.txt
│   └── requirements_test.txt
│
├── kaggle_dl/
│   └── kaggle_datasets/
├── venv_extract/
├── venv_train/
└── venv_test/

Save as docs/SETUP.md 📄

Extraction is still running — let it finish! 🎯🚀
