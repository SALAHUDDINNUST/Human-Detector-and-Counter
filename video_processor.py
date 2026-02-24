import cv2
import numpy as np
from datetime import datetime
import os

class VideoHumanCounter:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())
        
        # Statistics
        self.total_people = 0
        self.max_in_frame = 0
        self.frame_count = 0
        
    def process_video(self, video_path, output_path=None):
        """
        Process a video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
        """
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"ERROR: Video file '{video_path}' not found!")
            return
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video Info:")
        print(f"- File: {video_path}")
        print(f"- Resolution: {width}x{height}")
        print(f"- FPS: {fps}")
        print(f"- Total frames: {total_frames}")
        print(f"- Duration: {total_frames/fps:.2f} seconds")
        
        # Setup video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("\nProcessing video...")
        print("Press 'q' to stop early")
        
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_number += 1
            
            # Process frame
            processed_frame, count = self.process_frame(frame)
            
            # Update statistics
            self.frame_count += 1
            self.total_people += count
            self.max_in_frame = max(self.max_in_frame, count)
            
            # Add frame number
            cv2.putText(processed_frame, f'Frame: {frame_number}/{total_frames}', 
                       (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show progress
            progress = (frame_number / total_frames) * 100
            cv2.putText(processed_frame, f'Progress: {progress:.1f}%', 
                       (10, height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display
            cv2.imshow('Video Processing', processed_frame)
            
            # Write output if requested
            if out:
                out.write(processed_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nProcessing stopped by user")
                break
            
            # Print progress every 100 frames
            if frame_number % 100 == 0:
                print(f"Processed {frame_number}/{total_frames} frames...")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        self.print_report(video_path)
    
    def process_frame(self, frame):
        """Process a single frame"""
        
        # Detect people
        (rects, weights) = self.hog.detectMultiScale(
            frame, 
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05
        )
        
        # Draw detections
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add count
        count = len(rects)
        cv2.putText(frame, f'Count: {count}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, count
    
    def print_report(self, video_path):
        """Print final report"""
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"Video: {video_path}")
        print(f"Frames processed: {self.frame_count}")
        print(f"Total people detected: {self.total_people}")
        print(f"Average per frame: {self.total_people/self.frame_count:.2f}")
        print(f"Maximum in single frame: {self.max_in_frame}")
        print("="*50)

def main():
    """Main function for video processing"""
    
    print("Video Human Counter")
    print("-" * 30)
    
    # Get video file path from user
    video_path = input("Enter video file path (or drag and drop): ").strip().strip('"')
    
    # Remove quotes if present
    video_path = video_path.strip("'").strip('"')
    
    # Ask for output
    save_output = input("Save output video? (y/n): ").lower() == 'y'
    
    output_path = None
    if save_output:
        output_name = input("Enter output filename (default: output.mp4): ").strip()
        if not output_name:
            output_name = "output.mp4"
        output_path = output_name
    
    # Create counter and process
    counter = VideoHumanCounter()
    counter.process_video(video_path, output_path)

if __name__ == "__main__":
    main()