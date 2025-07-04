# Sketch-recognition-System
This is a project that recognize the sketch drawn through fingers.
# ✍️ QuickDraw Gesture-Based Sketch Recognition

This project is a real-time sketch recognition system that lets users draw in the air using hand gestures. Using a trained Convolutional Neural Network (CNN) on the QuickDraw dataset, it predicts hand-drawn objects from webcam input. The system leverages OpenCV and MediaPipe for gesture tracking and TensorFlow/Keras for model prediction.

---

## 🚀 Features

- ✋ Hand gesture tracking using MediaPipe
- 🎨 Draw sketches in the air using your finger
- 🧠 Trained CNN model using Google’s QuickDraw dataset (10 classes)
- 📊 Live predictions with class probabilities and accuracy metrics
- 📷 Real-time webcam integration with visual feedback

---

## 🧠 Classes Trained On

- cat  
- diamond  
- eye  
- ladder  
- moon  
- necklace  
- snowflake  
- sword  
- tornado  
- watermelon

---

## 🗃️ Dataset

Google’s **QuickDraw Dataset** was used. The `.npy` files were automatically downloaded using a Python script. Each class included 20,000 training samples.

- Dataset link: [https://quickdraw.withgoogle.com/data](https://quickdraw.withgoogle.com/data)

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy, Matplotlib, Seaborn
- Google QuickDraw Dataset

---

## 📁 Project Structure
│
├── modelTraining.py # Train the CNN model
├── predict.py / main.py # Gesture-based real-time prediction
├── loadData.py # Load and preprocess .npy sketch data
├── keys.py # List of selected classes
├── QuickDrawCNN.keras # Saved model (recommended format)
├── dataset/ # (Auto-downloaded QuickDraw .npy files)
├── myenv
└── README.md
## 🧪 How to Run

### 📦 1. Create a Virtual Environment

```bash
python -m venv quickdraw_env
source quickdraw_env/bin/activate

 ### Requirements (requirements.txt)
tensorflow>=2.10
opencv-python
mediapipe
matplotlib
numpy
seaborn
Demo:
![Screenshot (600)](https://github.com/user-attachments/assets/cb704806-f9cc-46bc-8346-8baac2825f9a)
![Screenshot (601)](https://github.com/user-attachments/assets/12c69f00-05fe-4f99-bba2-1dfb87249fdb)


