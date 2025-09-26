# 🖼️ AI Image Detection

This portfolio project is an **AI Image Detector** that classifies whether an image is created by a **human** or **AI-generated**.  
The model was trained using a **Custom CNN**, with additional image quality analysis (blur, noise, resolution) and interpretability powered by **Grad-CAM**.

---

## 🚀 Key Features

- 🔍 **Detection of AI vs Human Images**
- 📊 **Prediction visualization** with confidence score
- ⚡ **Deployment via Docker & Hugging Face Spaces**
- 🎨 **UI/UX built with TailwindCSS** (landing page & result screen)

---

## 🛠️ Tech Stack

- **Python** (TensorFlow/Keras, OpenCV, NumPy, Pandas)
- **Flask API** for backend
- **Docker** for containerization
- **TailwindCSS** for frontend
- **Hugging Face Spaces** for deployment → [AI-Image-Detection](https://huggingface.co/spaces/alfando/AI-Image-Detection)

---

## 🔧 Techniques Applied

- **Data Preparation**

  - Image dataset + CSV annotations loaded from Google Drive
  - Split into 80% training and 20% validation

- **Data Augmentation**

  - Pixel normalization (`rescale=1./255`)
  - Augmentations: rotation, shift, shear, zoom, horizontal flip

- **Modeling (CNN)**

  - 3 layers of **Conv2D + MaxPooling** (32 → 64 → 128 filters)
  - **Flatten → Dense(512, ReLU)**
  - **Output Dense(2, Softmax)** → Human vs AI classification

- **Training**

  - Optimizer: **Adam**
  - Loss: **Categorical Crossentropy**
  - Epochs: 40
  - Batch size: 32

- **Evaluation**
  - Accuracy & loss curves
  - **Classification report** (precision, recall, f1-score)
  - **Confusion matrix** per class (Human vs AI)

---

## 📊 Final Results

- **Training accuracy:** ~98%
- **Validation accuracy:** ~91%
- **Per-class accuracy:**
  - Human → 0.90
  - AI → 0.92

---
