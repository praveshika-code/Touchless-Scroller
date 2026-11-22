# Touchless Scroller 🖐️

### Smart Hands-Free Scrolling System

Touchless Scroller is a Python-based application that allows users to scroll their computer screen without touching a mouse or keyboard.

It uses computer vision and MediaPipe to detect hand gestures and eye blinks through a webcam.

## ✨ Features

- **Hand Gesture Mode:** Control scrolling using pinch gestures.
- **Eye Blink Mode:** Scroll using eye blink detection.
- **Sensitivity Control:** Adjust scrolling speed from 1 to 10.
- **Pause & Resume:** Temporarily pause and restart scrolling.
- **Start & Stop:** Control the application easily.
- **Integrated Dashboard:** Simple graphical interface using Tkinter.

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe
- PyAutoGUI
- Tkinter
- Pillow (PIL)

## 📂 Project Structure

```text
Touchless-Scroller/
│
├── models/
│   ├── hand_landmarker.task
│   └── face_landmarker.task
│
├── integrated_dashboard.py
├── hand_tracking.py
├── eye_tracking.py
└── README.md
```

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/praveshika-code/Touchless-Scroller.git
```

2. Open the project folder:

```bash
cd Touchless-Scroller
```

3. Install the required libraries:

```bash
pip install opencv-python mediapipe pyautogui pillow
```

## ▶️ How to Run

Run the integrated dashboard:

```bash
python integrated_dashboard.py
```

1. Select Hand Gesture Mode or Eye Blink Mode.
2. Click START.
3. Adjust scrolling sensitivity.
4. Use PAUSE, RESUME, and STOP as needed.

## 🎯 Project Objective

The objective of this project is to develop a touch-free computer interaction system that makes screen navigation easier through hand gestures and eye blink detection.

## 👩‍💻 Author

**Praveshika Jain**

GitHub: [praveshika-code](https://github.com/praveshika-code)
