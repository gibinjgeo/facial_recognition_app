import cv2
import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk
import mediapipe as mp

# Initialize MediaPipe face detection and drawing utilities
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection App")
        self.root.geometry("900x700")

        # Frame to hold the video feed
        self.video_frame = Frame(self.root, width=640, height=480)
        self.video_frame.pack(pady=20)

        # Label to display the video feed
        self.video_label = Label(self.video_frame)
        self.video_label.pack()

        # Button to check for face detection
        self.face_check_btn = Button(root, text="Check for Face", command=self.check_for_face)
        self.face_check_btn.pack(pady=10)

        # Capture video from the webcam
        self.cap = cv2.VideoCapture(0)

        # Placeholder for the current frame
        self.current_frame = None

        # Start updating the video feed
        self.update_video_feed()

    def update_video_feed(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            # Convert the frame to RGB for displaying in Tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the Tkinter label with the new frame
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # Call this function again after 10 milliseconds
        self.root.after(10, self.update_video_feed)

    def check_for_face(self):
        if self.current_frame is not None:
            # Use MediaPipe to detect faces
            with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                image_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image_rgb)

                if results.detections:
                    # Draw face detection landmarks on the frame
                    for detection in results.detections:
                        mp_drawing.draw_detection(self.current_frame, detection)

                # Update the Tkinter video label with the face-detected frame
                rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

# Initialize the Tkinter window and start the application
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
