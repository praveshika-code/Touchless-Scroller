import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import pyautogui
import os
import time
from PIL import Image, ImageTk


# ---------------- SETTINGS ----------------

pyautogui.PAUSE = 0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
FACE_MODEL = os.path.join(BASE_DIR, "models", "face_landmarker.task")

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

SCROLL_THRESHOLD = 0.015
BLINK_COOLDOWN = 1.0


# ---------------- APP STATE ----------------

cap = None
landmarker = None
running = False
paused = False
start_time = 0

previous_y = None
last_blink_time = 0
last_blink_detected = False


# ---------------- MEDIAPIPE ----------------

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandOptions = mp.tasks.vision.HandLandmarkerOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceOptions = mp.tasks.vision.FaceLandmarkerOptions


# ---------------- START MODE ----------------

def start_mode():
    global cap, landmarker, running, paused
    global start_time, previous_y
    global last_blink_time, last_blink_detected

    if running:
        return

    mode = selected_mode.get()

    try:
        if mode == "Hand Gesture":
            if not os.path.exists(HAND_MODEL):
                messagebox.showerror(
                    "Error",
                    f"Hand model not found:\n{HAND_MODEL}"
                )
                return

            options = HandOptions(
                base_options=BaseOptions(model_asset_path=HAND_MODEL),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=1
            )
            landmarker = HandLandmarker.create_from_options(options)

        else:
            if not os.path.exists(FACE_MODEL):
                messagebox.showerror(
                    "Error",
                    f"Face model not found:\n{FACE_MODEL}"
                )
                return

            options = FaceOptions(
                base_options=BaseOptions(model_asset_path=FACE_MODEL),
                running_mode=VisionRunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=True
            )
            landmarker = FaceLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            cap.release()
            cap = None

            if landmarker is not None:
                landmarker.close()
                landmarker = None

            messagebox.showerror(
                "Camera Error",
                "Webcam not detected. Check that your camera is connected "
                "and not being used by another application."
            )
            return

        running = True
        paused = False
        start_time = time.monotonic()
        previous_y = None
        last_blink_time = 0
        last_blink_detected = False

        status_label.config(
            text=f"Status: {mode} Mode Running",
            fg="#22c55e"
        )

        update_camera()

    except Exception as error:
        messagebox.showerror("Error", str(error))
        stop_mode()


# ---------------- CAMERA UPDATE ----------------

def update_camera():
    global previous_y, last_blink_time, last_blink_detected

    if not running or cap is None or landmarker is None:
        return

    success, frame = cap.read()

    if not success:
        stop_mode()
        messagebox.showerror("Camera Error", "Could not read from webcam.")
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = int((time.monotonic() - start_time) * 1000)
    results = landmarker.detect_for_video(mp_image, timestamp)

    mode = selected_mode.get()
    display_status = "Searching..."

    # -------- HAND MODE --------

    if mode == "Hand Gesture":
        if results.hand_landmarks:
            hand = results.hand_landmarks[0]
            h, w, _ = frame.shape
            points = []

            for landmark in hand:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            for a, b in CONNECTIONS:
                cv2.line(
                    frame,
                    points[a],
                    points[b],
                    (255, 0, 0),
                    2
                )

            thumb = hand[4]
            index = hand[8]

            distance = (
                (thumb.x - index.x) ** 2 +
                (thumb.y - index.y) ** 2
            ) ** 0.5

            if distance < 0.05:
                display_status = "PINCH DETECTED"

                current_y = (thumb.y + index.y) / 2

                # While paused, keep updating the reference position,
                # but do not send any scroll command.
                if previous_y is not None and not paused:
                    movement = previous_y - current_y
                    amount = scroll_speed.get()

                    if movement > SCROLL_THRESHOLD:
                        pyautogui.scroll(amount)
                    elif movement < -SCROLL_THRESHOLD:
                        pyautogui.scroll(-amount)

                previous_y = current_y

            else:
                display_status = "HAND DETECTED"
                previous_y = None

        else:
            display_status = "NO HAND DETECTED"
            previous_y = None

    # -------- EYE MODE --------

    else:
        if results.face_landmarks:
            display_status = "FACE DETECTED"

            if results.face_blendshapes:
                categories = results.face_blendshapes[0]
                scores = {
                    item.category_name: item.score
                    for item in categories
                }

                left = scores.get("eyeBlinkLeft", 0)
                right = scores.get("eyeBlinkRight", 0)
                blink = left > 0.5 and right > 0.5

                if blink:
                    display_status = "BLINK DETECTED"
                    current_time = time.monotonic()

                    if (
                        not paused
                        and not last_blink_detected
                        and current_time - last_blink_time > BLINK_COOLDOWN
                    ):
                        pyautogui.scroll(-scroll_speed.get())
                        last_blink_time = current_time

                else:
                        display_status = "EYES OPEN"

                    # Update this even while paused. This prevents a held blink
                    # from triggering a scroll immediately after Resume.
                        last_blink_detected = blink

        else:
            display_status = "NO FACE DETECTED"
            last_blink_detected = False

    # -------- DISPLAY STATUS --------

    if paused:
        status_label.config(
            text=f"PAUSED | {display_status}",
            fg="#facc15"
        )
    else:
        status_label.config(
            text=display_status,
            fg="#22c55e"
        )

    # Convert camera frame for Tkinter
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    image = image.resize((600, 400))

    photo = ImageTk.PhotoImage(image=image)
    camera_label.config(image=photo)
    camera_label.image = photo

    if running:
        root.after(20, update_camera)


# ---------------- PAUSE / RESUME / STOP ----------------

def pause_mode():
    global paused

    if running:
        paused = True
        status_label.config(
            text="Status: PAUSED",
            fg="#facc15"
        )


def resume_mode():
    global paused

    if running:
        paused = False
        status_label.config(
            text="Status: RESUMED",
            fg="#22c55e"
        )


def stop_mode():
    global cap, landmarker, running, paused, previous_y
    global last_blink_detected

    running = False
    paused = False
    previous_y = None
    last_blink_detected = False

    if cap is not None:
        cap.release()
        cap = None

    if landmarker is not None:
        landmarker.close()
        landmarker = None

    camera_label.config(image="")
    camera_label.image = None

    status_label.config(
        text="Status: Stopped",
        fg="#f87171"
    )


def close_app():
    stop_mode()
    root.destroy()


# ---------------- GUI DESIGN ----------------

root = tk.Tk()
root.title("Touchless Scroller")
root.geometry("850x780")
root.configure(bg="#111827")
root.resizable(False, False)

title = tk.Label(
    root,
    text="TOUCHLESS SCROLLER",
    font=("Arial", 25, "bold"),
    fg="#60a5fa",
    bg="#111827"
)
title.pack(pady=(15, 5))

subtitle = tk.Label(
    root,
    text="Smart Hands-Free Control System",
    font=("Arial", 12),
    fg="#d1d5db",
    bg="#111827"
)
subtitle.pack(pady=5)

selected_mode = tk.StringVar(value="Hand Gesture")

mode_frame = tk.Frame(root, bg="#111827")
mode_frame.pack(pady=12)

tk.Radiobutton(
    mode_frame,
    text="Hand Gesture Mode",
    variable=selected_mode,
    value="Hand Gesture",
    font=("Arial", 12),
    fg="white",
    bg="#111827",
    selectcolor="#2563eb"
).pack(side="left", padx=15)

tk.Radiobutton(
    mode_frame,
    text="Eye Blink Mode",
    variable=selected_mode,
    value="Eye Blink",
    font=("Arial", 12),
    fg="white",
    bg="#111827",
    selectcolor="#2563eb"
).pack(side="left", padx=15)

scroll_speed = tk.IntVar(value=5)

tk.Scale(
    root,
    from_=1,
    to=10,
    orient=tk.HORIZONTAL,
    variable=scroll_speed,
    label="Scrolling Sensitivity (1 = Slow, 10 = Fast)",
    length=350,
    bg="#111827",
    fg="white",
    highlightthickness=0,
    troughcolor="#2563eb"
).pack(pady=8)

camera_label = tk.Label(root, bg="#273449")
camera_label.pack(pady=10)

status_label = tk.Label(
    root,
    text="Status: Not Running",
    font=("Arial", 13, "bold"),
    fg="#f87171",
    bg="#111827"
)
status_label.pack(pady=8)

button_frame = tk.Frame(root, bg="#111827")
button_frame.pack(pady=10)

tk.Button(
    button_frame,
    text="START",
    command=start_mode,
    font=("Arial", 13, "bold"),
    bg="#16a34a",
    fg="white",
    width=12,
    pady=8
).pack(side="left", padx=8)

tk.Button(
    button_frame,
    text="PAUSE",
    command=pause_mode,
    font=("Arial", 13, "bold"),
    bg="#d97706",
    fg="white",
    width=12,
    pady=8
).pack(side="left", padx=8)

tk.Button(
    button_frame,
    text="RESUME",
    command=resume_mode,
    font=("Arial", 13, "bold"),
    bg="#2563eb",
    fg="white",
    width=12,
    pady=8
).pack(side="left", padx=8)

tk.Button(
    button_frame,
    text="STOP",
    command=stop_mode,
    font=("Arial", 13, "bold"),
    bg="#dc2626",
    fg="white",
    width=12,
    pady=8
).pack(side="left", padx=8)

root.protocol("WM_DELETE_WINDOW", close_app)
root.mainloop()
