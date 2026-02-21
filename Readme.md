# 🐟 Multiclass Fish Image Classification (Deep Learning)

## 📌 Project Overview
This project classifies fish images into 11 categories using Deep Learning.
Two approaches were implemented:
- CNN from scratch
- Transfer Learning using MobileNetV2 (best performing model)

A Streamlit web app is deployed for real-time image classification.

---

## 🚀 Tech Stack
- Python
- TensorFlow / Keras
- Streamlit
- Scikit-learn
- Matplotlib, Seaborn

---

## 📂 Project Structure

Fish_classification_project/
│
├── data/ # (ignored in GitHub, download separately)
│ ├── train/
│ ├── val/
│ └── test/
│
├── models/
│ ├── cnn_from_scratch.h5
│ └── mobilenet_best.h5
│
├── train.py # CNN training
├── train_mobilenet.py # Transfer Learning training
├── evaluate.py # Metrics + Confusion Matrix
├── app.py # Streamlit App
├── requirements.txt
└── README.md

## 🧠 Workflow

1. Dataset loading using ImageDataGenerator  
2. Data preprocessing & augmentation  
3. CNN model from scratch (baseline)  
4. Transfer Learning with MobileNetV2  
5. Model evaluation (accuracy, precision, recall, F1-score, confusion matrix)  
6. Best model saved and used in Streamlit app  
7. Streamlit web app for real-time predictions  

---

## 📊 Results
| Model              | Validation Accuracy |
|--------------------|---------------------|
| CNN from Scratch   | ~59%                |
| MobileNetV2 (TL)   | ~93%                |

---

## 🖥️ How to Run

### 1️⃣ Setup

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

2️⃣ Train Models

python train.py
python train_mobilenet.py

3️⃣ Evaluate
python evaluate.py

4️⃣ Run Web App
streamlit run app.py

🎥 Demo Video

👉 (Add your LinkedIn video link here)

## ⚠️ Challenges Faced

- Handling large model files (>100MB) restricted by GitHub  
- Overfitting in CNN from scratch  
- Environment setup issues (TensorFlow + venv on Windows)  
- Improving model performance with transfer learning  

**Solution:**  
Used transfer learning, data augmentation, early stopping, and external storage for model files.

## 🚀 Future Improvements

- Fine-tune upper layers of MobileNetV2  
- Try EfficientNetB0 for higher accuracy  
- Add Grad-CAM for explainability  
- Deploy the app using Streamlit Cloud  


📌 Conclusion

Transfer learning using MobileNetV2 significantly outperformed the CNN built from scratch.
The deployed Streamlit application allows real-time fish species prediction.


---  
