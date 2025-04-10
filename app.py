from flask import Flask, render_template, request
import cv2
import os
import time
import base64
import numpy as np
import face_recognition
import gc  # Garbage collection

app = Flask(__name__)

# Global variable to store the last time the camera was used
last_camera_use = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    global last_camera_use
    
    if 'first_face_file' not in request.files:
        return "No file part"
    
    file = request.files['first_face_file']
    
    if file.filename == '':
        return "No selected file"

    if file:
        temp_path = "temp_upload.jpg"
        
        try:
            # Ensure any previous temp file is removed
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            file.save(temp_path)

            # Load and encode uploaded image
            uploaded_img = face_recognition.load_image_file(temp_path)
            uploaded_encodings = face_recognition.face_encodings(uploaded_img)

            if len(uploaded_encodings) == 0:
                return "No face detected in the uploaded image. Please try again."

            uploaded_encoding = uploaded_encodings[0]

            # Wait at least 3 seconds between camera sessions
            current_time = time.time()
            if current_time - last_camera_use < 3:
                time.sleep(3 - (current_time - last_camera_use))
            
            # Initialize camera with explicit resolution
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return "Failed to open webcam. Please check your camera connection."
            
            # Set camera properties for better quality
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Allow camera to warm up and adjust exposure
            time.sleep(2)
            
            # Discard first few frames to let camera adjust
            for _ in range(5):
                ret, _ = cap.read()
                if not ret:
                    break
            
            match_found = False
            start_time = time.time()
            timeout = 30
            img_str = None
            
            # Use a counter to reduce processing load
            frame_count = 0

            while time.time() - start_time < timeout:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only process every 3rd frame to reduce CPU load
                frame_count += 1
                if frame_count % 3 != 0:
                    continue
                
                # Create a copy to avoid modifying the original
                display_frame = frame.copy()
                
                # Reduce size for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(frame_rgb)
                
                if face_locations:
                    encodings = face_recognition.face_encodings(frame_rgb, face_locations)
                    
                    for i, encoding in enumerate(encodings):
                        # Adjust tolerance for better matching
                        match = face_recognition.compare_faces([uploaded_encoding], encoding, tolerance=0.6)
                        
                        # Get face distance for confidence level
                        face_distance = face_recognition.face_distance([uploaded_encoding], encoding)[0]
                        
                        if match[0]:
                            # Scale back the face location coordinates
                            top, right, bottom, left = face_locations[i]
                            top *= 2
                            right *= 2
                            bottom *= 2
                            left *= 2
                            
                            # Calculate confidence percentage
                            confidence = (1 - face_distance) * 100
                            
                            # Draw rectangle and confidence
                            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                            cv2.putText(display_frame, f"Matched: {confidence:.1f}%", (left, top-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(display_frame, current_time, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            _, buffer = cv2.imencode('.jpg', display_frame)
                            img_str = base64.b64encode(buffer).decode('utf-8')
                            match_found = True
                            break
                    
                    if match_found:
                        break
                
                cv2.putText(display_frame, "Scanning... Please wait", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Properly release resources
            last_camera_use = time.time()
            cap.release()
            cv2.destroyAllWindows()
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Force garbage collection
            gc.collect()

            if match_found:
                return render_template('matched_image.html', matched_image=img_str)
            else:
                return render_template('no_match.html')

        except Exception as e:
            # Ensure cleanup even on error
            if 'cap' in locals() and cap is not None:
                cap.release()
                
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            cv2.destroyAllWindows()
            print(f"Error: {str(e)}")
            return f"An error occurred: {str(e)}"

    return "Error processing the file"

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
