# 🖼️ AI Image Detection

Proyek ini adalah **AI Image Detector** untuk mengklasifikasikan gambar apakah dibuat oleh **manusia** atau **AI-generated**.  
Model dilatih menggunakan **Custom CNN** dengan tambahan analisis kualitas gambar (blur, noise, resolusi) serta interpretabilitas menggunakan **Grad-CAM**.

---

## 🚀 Fitur Utama

- 🔍 **Deteksi AI vs Human Image**
- 📊 **Visualisasi hasil** dengan confidence score
- ⚡ **Deploy via Docker & Hugging Face Spaces**
- 🎨 **UI/UX berbasis TailwindCSS** (landing page & result screen)

---

## 🛠️ Tech Stack

- **Python** (TensorFlow/Keras, OpenCV, NumPy, Pandas)
- **Flask API** untuk backend
- **Docker** untuk containerization
- **TailwindCSS** untuk frontend
- **Hugging Face Spaces** untuk deployment → [AI-Image-Detection](https://huggingface.co/spaces/alfando/AI-Image-Detection)

---

## 🔧 Teknik yang Digunakan

- **Data Preparation**

  - Dataset gambar + anotasi CSV dari Google Drive
  - Split data: 80% training, 20% validation

- **Data Augmentation**

  - Normalisasi piksel (`rescale=1./255`)
  - Augmentasi: rotasi, shift, shear, zoom, horizontal flip

- **Modeling (CNN)**

  - 3 lapisan **Conv2D + MaxPooling** (32 → 64 → 128 filter)
  - **Flatten → Dense(512, ReLU)**
  - **Output Dense(2, Softmax)** → klasifikasi Human vs AI

- **Training**

  - Optimizer: **Adam**
  - Loss: **Categorical Crossentropy**
  - Epoch: 40
  - Batch size: 32

- **Evaluasi**
  - Plot kurva **akurasi & loss**
  - **Classification report** (precision, recall, f1-score)
  - **Confusion matrix** per kelas (Human vs AI)

---

## 📊 Hasil Akhir

- **Akurasi training:** ~98%
- **Akurasi validasi:** ~91%
- **Per-class accuracy:**
  - Human → 0.90
  - AI → 0.92

---
