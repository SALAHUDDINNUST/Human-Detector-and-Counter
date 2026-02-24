import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time

class HumanCounterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Detector and Counter")
        self.root.geometry("900x700")
        
        # Variables
        self.running = False
        self.camera = None
        self.detector = None
        self.current_count = 0
        self.max_count = 0
        
        # Create GUI
        self.create_widgets()
        
        # Initialize detector
        self.init_detector()
    
    def create_widgets(self):
        """Create GUI elements"""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Buttons
        ttk.Button(control_frame, text="Start Camera", 
                  command=self.start_camera).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Stop Camera", 
                  command=self.stop_camera).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Select Video File", 
                  command=self.select_video).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Reset Stats", 
                  command=self.reset_stats).grid(row=0, column=3, padx=5)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding="10")
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Statistics labels
        self.stats_current = ttk.Label(stats_frame, text="Current Count: 0", font=('Arial', 12))
        self.stats_current.grid(row=0, column=0, padx=20, pady=5)
        
        self.stats_max = ttk.Label(stats_frame, text="Max Count: 0", font=('Arial', 12))
        self.stats_max.grid(row=0, column=1, padx=20, pady=5)
        
        self.stats_fps = ttk.Label(stats_frame, text="FPS: 0", font=('Arial', 12))
        self.stats_fps.grid(row=0, column=2, padx=20, pady=5)
        
        # Video display
        video_frame = ttk.LabelFrame(main_frame, text="Video Feed", padding="10")
        video_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.grid(row=0, column=0)
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
    
    def init_detector(self):
        """Initialize HOG detector"""
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())
        self.update_status("Detector initialized")
    
    def start_camera(self):
        """Start webcam"""
        if not self.running:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                self.running = True
                self.update_status("Camera started")
                self.process_video()
            else:
                messagebox.showerror("Error", "Could not open camera")
    
    def stop_camera(self):
        """Stop video processing"""
        self.running = False
        if self.camera:
            self.camera.release()
        self.update_status("Stopped")
    
    def select_video(self):
        """Select video file"""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        
        if filename:
            self.stop_camera()
            self.camera = cv2.VideoCapture(filename)
            if self.camera.isOpened():
                self.running = True
                self.update_status(f"Playing: {filename}")
                self.process_video()
    
    def reset_stats(self):
        """Reset statistics"""
        self.current_count = 0
        self.max_count = 0
        self.update_stats()
        self.update_status("Statistics reset")
    
    def process_video(self):
        """Main video processing loop"""
        if self.running:
            ret, frame = self.camera.read()
            
            if ret:
                # Process frame
                processed_frame, count = self.detect_people(frame)
                
                # Update statistics
                self.current_count = count
                self.max_count = max(self.max_count, count)
                self.update_stats()
                
                # Convert for display
                self.display_frame(processed_frame)
            
            # Schedule next frame
            self.root.after(30, self.process_video)
    
    def detect_people(self, frame):
        """Detect people in frame"""
        # Resize for faster processing
        frame = cv2.resize(frame, (640, 480))
        
        # Detect
        (rects, _) = self.hog.detectMultiScale(
            frame, winStride=(4, 4), padding=(8, 8), scale=1.05
        )
        
        # Draw rectangles
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return frame, len(rects)
    
    def display_frame(self, frame):
        """Convert and display frame in GUI"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(frame_rgb)
        image = ImageTk.PhotoImage(image)
        
        # Update label
        self.video_label.configure(image=image)
        self.video_label.image = image
    
    def update_stats(self):
        """Update statistics display"""
        self.stats_current.config(text=f"Current Count: {self.current_count}")
        self.stats_max.config(text=f"Max Count: {self.max_count}")
    
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=f"Status: {message}")

def main():
    """Main function"""
    root = tk.Tk()
    app = HumanCounterGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()