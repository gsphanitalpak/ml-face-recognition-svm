# Face Recognition Model using InsightFace and SVM

Welcome!  
This repository contains a machine learning project for **face recognition** using **InsightFace** for feature extraction and **Support Vector Machine (SVM)** for classification.

---

## 📖 Table of Contents
- [About the Project](#about-the-project)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Training the Model](#training-the-model)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

---

## About the Project

- **What it does**:  
  This project builds a robust face recognition model. It extracts features from images and uses an SVM to classify faces.

- **Why it's useful**:  
  Efficient for security systems, personal identification, and real-world face recognition applications.

---

## Requirements

- **Python 3.x**
- Required libraries:
  - `opencv-python` (cv2)
  - `numpy`
  - `pandas`
  - `joblib`
  - `tqdm`
  - `scikit-learn`
  - `insightface`
  - `seaborn`
  - `matplotlib`

You can install dependencies by running:

[![Terminal Command](https://img.shields.io/badge/Install%20Dependencies-terminal-blue)]()
```terminal
pip install -r requirements.txt
```

---

## Project Structure

```
project-directory/
│
├── archives/
│   ├── Dataset.csv                               # CSV file with image names and corresponding labels
│   └── Original Images/                          # Directory containing images
│         └── Person_001/                         
│               └── person_001_image_001          
│               └── person_001_image_002        
│               ...
├── models/
│   └── svm_insightface.pkl                       # Saved trained SVM model
│
├── train.py                                      # Script to train the model
├── requirements.txt                              # Dependency list
└── README.md                                     # Project information
```

---

## Getting Started

To clone and set up the project:

[![Terminal Command](https://img.shields.io/badge/Clone%20Repo-terminal-blue)]()
```terminal
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/gsphanitalpak/ml-face-recognition-svm)
cd ml-face-recognition-svm
pip install -r requirements.txt
```

---
## Dataset
Before starting with the model development you need the data that can be downloaded from this google drive link
After downloading, place the extracted folders inside the archives/ directory or simple change the folder name to archives and place it in your project folder following the directory structure.
```Google Drive link
[https://drive.google.com/your-dataset-link](https://drive.google.com/drive/folders/1CPI4Z2qlnvSAJfuTHRTNvamMGlvlTVSC?usp=drive_link)
```
---

## Training the Model

To train the face recognition model, run:

[![Terminal Command](https://img.shields.io/badge/Train%20Model-terminal-blue)]()
```terminal
python train.py
```

### Script Workflow
1. **Load Dataset** from `archives/Dataset.csv`
2. **Feature Extraction** using **InsightFace** to get embeddings
3. **SVM Training** on extracted embeddings
4. **Model Evaluation** with a classification report and confusion matrix
5. **Model Saving** into the `models/` directory

---

## Model Evaluation

Evaluation Metrics:
- **Classification Report**: Precision, Recall, F1-score
- **Confusion Matrix**: Visualized using `seaborn`

---

## Usage

### Saving the Model

[![Python Code](https://img.shields.io/badge/Save%20Model-python-green)]()
```python
import os
import joblib

os.makedirs("models", exist_ok=True)
joblib.dump(svm_model, MODEL_SAVE_PATH)
print(f"Model saved at {MODEL_SAVE_PATH} ✅")
```

---

### Loading the Trained Model

[![Python Code](https://img.shields.io/badge/Loading%20Saved%20Model-python-green)]()
```python
import joblib

svm_model = joblib.load('models/svm_insightface.pkl')
```

---

### Predicting and Visualizing

Make sure to first preprocess the image and extract its embedding using **InsightFace**.
- To use the model for prediction:
- `embedding` is the feature vector extracted from a face image

[![Python Code](https://img.shields.io/badge/Model%20Extraction-python-green)]()
```python
prediction = svm_model.predict([embedding])
print(prediction)
```

---

## License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for more details.

---

## Contact

Maintained by **Santhosh Phanitalpak Gandhala**.  
For support, open an [issue](../../issues) or submit a [pull request](../../pulls).

---
