import cv2
import tkinter as tk
from tkinter import Label, Button, Frame, messagebox
from PIL import Image, ImageTk
import mediapipe as mp
import os

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection App")
        self.root.geometry("900x700")

        self.video_frame = Frame(self.root, width=640, height=480)
        self.video_frame.pack(pady=20)

        self.video_label = Label(self.video_frame)
        self.video_label.pack()

        self.face_check_btn = Button(root, text="Check for Face", command=self.check_for_face)
        self.face_check_btn.pack(pady=10)

        self.cap = cv2.VideoCapture(0)

        self.current_frame = None

        self.update_video_feed()

    def update_video_feed(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_video_feed)

    def check_for_face(self):
        if self.current_frame is not None:
            with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                image_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image_rgb)

                if results.detections:
                    for detection in results.detections:
                        mp_drawing.draw_detection(self.current_frame, detection)

                    self.save_image_with_landmarks(self.current_frame)
                else:
                    messagebox.showinfo("Face Detection", "Face not detected")

    def save_image_with_landmarks(self, frame):
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        save_path = "face_with_landmarks.jpg"
        cv2.imwrite(save_path, bgr_frame)

        messagebox.showinfo("Face Detection", f"Image saved successfully at {os.path.abspath(save_path)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
