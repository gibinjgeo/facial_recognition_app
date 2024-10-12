import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import threading

# Class for managing the GUI and Facial Recognition
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Recognition App")
        self.root.geometry("800x600")

        # Create a label to hold the video feed
        self.video_label = Label(root)
        self.video_label.pack()

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Start video capture
        self.cap = cv2.VideoCapture(0)

        # Create a thread to continuously update the video feed
        self.thread = threading.Thread(target=self.update_video_feed)
        self.thread.daemon = True
        self.thread.start()

    def update_video_feed(self):
        # Continuously capture frames from the webcam
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convert the frame to RGB (as MediaPipe expects RGB input)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame to find facial landmarks
            results = self.face_mesh.process(rgb_frame)

            # Draw the facial landmarks on the frame
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, face_landmarks, self.mp_face_mesh.FACE_CONNECTIONS)

            # Convert the frame back to BGR to display using OpenCV
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the Tkinter label with the new frame
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.cap.release()

# Initialize the Tkinter window and start the application
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
