import cv2
import numpy as np
from datetime import datetime
import time

class HumanDetector:
    def __init__(self):
        """Initialize the detector"""
        print("Initializing Human Detector...")
        
        # Load the pre-trained model
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())
        
        # Statistics variables
        self.total_detections = 0
        self.max_people = 0
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
        print("Detector initialized successfully!")
    
    def process_frame(self, frame):
        """Process a single frame"""
        
        # Update frame count
        self.frame_count += 1
        
        # Calculate FPS
        current_time = time.time()
        self.fps = 1 / (current_time - self.last_time)
        self.last_time = current_time
        
        # Resize for faster processing
        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (640, 480))
        
        # Detect people
        (rects, weights) = self.hog.detectMultiScale(
            frame, 
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05
        )
        
        # Get current count
        current_count = len(rects)
        
        # Update statistics
        self.total_detections += current_count
        self.max_people = max(self.max_people, current_count)
        
        # Draw detections
        for (x, y, w, h) in rects:
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw confidence if available
            cv2.putText(frame, f'Person', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display statistics
        self.display_stats(frame, current_count)
        
        return frame, current_count
    
    def display_stats(self, frame, current_count):
        """Display all statistics on frame"""
        
        # Current count (large text)
        cv2.putText(frame, f'Current: {current_count}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Maximum people
        cv2.putText(frame, f'Max: {self.max_people}', 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Average (if we have frames)
        if self.frame_count > 0:
            avg = self.total_detections / self.frame_count
            cv2.putText(frame, f'Avg: {avg:.2f}', 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # FPS
        cv2.putText(frame, f'FPS: {self.fps:.1f}', 
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, 'Press: q=quit, r=reset, s=screenshot', 
                   (frame.shape[1] - 300, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    """Main function"""
    
    # Step 1: Create detector
    detector = HumanDetector()
    
    # Step 2: Open webcam
    print("\nOpening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        return
    
    print("Webcam opened successfully!")
    print("\nControls:")
    print("- Press 'q' to quit")
    print("- Press 'r' to reset statistics")
    print("- Press 's' to save screenshot")
    print("- Press 'h' for help")
    print("\nDetecting people...")
    
    screenshot_count = 0
    
    # Step 3: Main loop
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to read from webcam")
            break
        
        # Process frame
        processed_frame, count = detector.process_frame(frame)
        
        # Show frame
        cv2.imshow('Human Detector - Enhanced Version', processed_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('r'):
            # Reset statistics
            detector.total_detections = 0
            detector.max_people = 0
            detector.frame_count = 0
            print("Statistics reset!")
        elif key == ord('s'):
            # Save screenshot
            screenshot_count += 1
            filename = f"screenshot_{screenshot_count}.jpg"
            cv2.imwrite(filename, processed_frame)
            print(f"Screenshot saved as {filename}")
        elif key == ord('h'):
            # Show help
            print("\n--- HELP ---")
            print("q: Quit program")
            print("r: Reset statistics")
            print("s: Save screenshot")
            print("h: Show this help")
            print("------------\n")
    
    # Step 4: Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Step 5: Show final report
    print("\n" + "="*40)
    print("FINAL REPORT")
    print("="*40)
    print(f"Total frames processed: {detector.frame_count}")
    print(f"Maximum people detected: {detector.max_people}")
    print(f"Average people per frame: {detector.total_detections/detector.frame_count:.2f}")
    print(f"Final FPS: {detector.fps:.1f}")
    print("="*40)

if __name__ == "__main__":
    main()