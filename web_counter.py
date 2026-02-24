from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import threading
import time

app = Flask(__name__)

class WebHumanCounter:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())
        self.current_count = 0
        self.max_count = 0
        self.running = True
        self.lock = threading.Lock()
        
        # Start processing thread
        self.thread = threading.Thread(target=self.process_frames)
        self.thread.daemon = True
        self.thread.start()
    
    def process_frames(self):
        """Process frames in background"""
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                # Detect people
                (rects, _) = self.hog.detectMultiScale(
                    frame, winStride=(4, 4), padding=(8, 8), scale=1.05
                )
                
                with self.lock:
                    self.current_count = len(rects)
                    self.max_count = max(self.max_count, self.current_count)
                
                # Draw rectangles
                for (x, y, w, h) in rects:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add text
                cv2.putText(frame, f'Count: {self.current_count}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Encode frame
                ret, jpeg = cv2.imencode('.jpg', frame)
                self.frame = jpeg.tobytes()
    
    def get_frame(self):
        """Get current frame"""
        return getattr(self, 'frame', b'')
    
    def get_stats(self):
        """Get current statistics"""
        with self.lock:
            return {
                'current': self.current_count,
                'max': self.max_count
            }

# Initialize counter
counter = WebHumanCounter()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            frame = counter.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    """Statistics API"""
    return jsonify(counter.get_stats())

@app.route('/reset')
def reset():
    """Reset statistics"""
    with counter.lock:
        counter.max_count = 0
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)