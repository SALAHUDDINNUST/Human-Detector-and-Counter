"""
STEP 1: Import required libraries
"""
import cv2
import numpy as np

"""
STEP 2: Initialize the detector
"""
print("Starting Human Detector...")
print("Press 'q' to quit")

# Initialize HOG descriptor (pre-trained for people detection)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())

"""
STEP 3: Open webcam
"""
# 0 = default webcam, you can also use video file path
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("ERROR: Could not open webcam")
    print("Troubleshooting:")
    print("1. Make sure webcam is connected")
    print("2. Close other apps using webcam")
    print("3. Try changing 0 to 1 in VideoCapture(1)")
    exit()

"""
STEP 4: Main loop for processing video frames
"""
while True:
    # Read frame from webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))
    
    """
    STEP 5: Detect humans in the frame
    """
    # Detect people
    (rects, weights) = hog.detectMultiScale(
        frame, 
        winStride=(4, 4),      # Window stride
        padding=(8, 8),         # Padding
        scale=1.05              # Scale factor
    )
    
    # Get count of people detected
    num_people = len(rects)
    
    """
    STEP 6: Draw rectangles around detected people
    """
    for (x, y, w, h) in rects:
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Add label
        cv2.putText(frame, 'Person', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    """
    STEP 7: Display the count on screen
    """
    # Show current count
    cv2.putText(frame, f'People Count: {num_people}', 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show instructions
    cv2.putText(frame, 'Press q to quit', (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    """
    STEP 8: Display the frame
    """
    cv2.imshow('Human Detector - Basic Version', frame)
    
    """
    STEP 9: Check for quit command
    """
    # Wait for 1ms and check if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

"""
STEP 10: Clean up
"""
cap.release()
cv2.destroyAllWindows()
print("Program ended. Final count:", num_people)