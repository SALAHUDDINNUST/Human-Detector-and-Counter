import cv2
import sys

def test_installation():
    """Test if everything is installed correctly"""
    
    print("="*50)
    print("Testing Human Detector Installation")
    print("="*50)
    
    # Test 1: Check Python version
    print(f"\n✓ Python version: {sys.version}")
    
    # Test 2: Check OpenCV
    try:
        print(f"✓ OpenCV version: {cv2.__version__}")
    except:
        print("✗ OpenCV not installed correctly")
        return False
    
    # Test 3: Check camera
    print("\nTesting camera...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✓ Camera detected")
        ret, frame = cap.read()
        if ret:
            print("✓ Camera working")
            print(f"  Frame size: {frame.shape}")
        else:
            print("✗ Camera not capturing frames")
        cap.release()
    else:
        print("✗ No camera found")
    
    # Test 4: Check HOG detector
    print("\nTesting HOG detector...")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())
    print("✓ HOG detector loaded")
    
    # Test 5: Quick detection test
    print("\nRunning quick detection test...")
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    (rects, _) = hog.detectMultiScale(test_image)
    print(f"✓ Detection function works (found {len(rects)} people in empty image)")
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50)
    
    return True

if __name__ == "__main__":
    test_installation()