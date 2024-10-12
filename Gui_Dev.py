import cv2
import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk

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
        # This function is called when the button is pressed
        if self.current_frame is not None:
            # Pass the current frame (self.current_frame) to your custom function
            # You can now process the frame here, for example, using face detection
            gray_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            # Custom processing (e.g., face detection can be added here)
            # Example: Just display the frame in a new window (you can replace this)
            cv2.imshow("Processed Frame", gray_frame)

# Initialize the Tkinter window and start the application
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
