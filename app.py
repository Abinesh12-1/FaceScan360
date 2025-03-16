from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
import time
import base64
import numpy as np

app = Flask(__name__)

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'first_face_file' not in request.files:
        return "No file part"
    
    file = request.files['first_face_file']

    if file.filename == '':
        return "No selected file"

    if file:
        try:
            # Save the uploaded file temporarily
            temp_path = "temp_upload.jpg"
            file.save(temp_path)
            
            # Load the uploaded image and detect faces
            uploaded_img = cv2.imread(temp_path)
            uploaded_gray = cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the uploaded image
            uploaded_faces = face_cascade.detectMultiScale(uploaded_gray, 1.1, 4)
            
            # Check if any face was detected in the uploaded image
            if len(uploaded_faces) == 0:
                os.remove(temp_path)
                return "No face detected in the uploaded image. Please try again."
            
            # Take the first detected face
            (x, y, w, h) = uploaded_faces[0]
            
            # Draw rectangle around the face in the uploaded image
            cv2.rectangle(uploaded_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract face region of interest
            uploaded_face_roi = uploaded_gray[y:y+h, x:x+w]
            
            # Resize to normalize
            uploaded_face_roi = cv2.resize(uploaded_face_roi, (100, 100))
            
            # Open webcam
            cap = cv2.VideoCapture(0)
            
            # Check if webcam opened successfully
            if not cap.isOpened():
                os.remove(temp_path)
                return "Failed to open webcam. Please check your camera connection."

            match_found = False
            start_time = time.time()
            timeout = 30  # 30 seconds timeout
            
            while time.time() - start_time < timeout:
                ret, frame = cap.read()

                if not ret:
                    break

                # Convert the image to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the frame
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) == 0:
                    continue

                for (x, y, w, h) in faces:
                    # Draw rectangle around the face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Extract face region of interest
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Resize to same dimensions for comparison
                    face_roi = cv2.resize(face_roi, (100, 100))
                    
                    # Simple face "matching" based on template matching
                    # This is a simplified approach and not as accurate as face_recognition
                    result = cv2.matchTemplate(face_roi, uploaded_face_roi, cv2.TM_CCOEFF_NORMED)
                    similarity = np.max(result)
                    
                    if similarity > 0.5:  # Threshold can be adjusted
                        print(f"Face matched with similarity: {similarity}")
                        
                        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                        
                        cv2.putText(frame, "Matched at: " + current_time, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, f"Similarity: {similarity:.2f}", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        # Convert image to base64 for displaying in HTML
                        _, buffer = cv2.imencode('.jpg', frame)
                        img_str = base64.b64encode(buffer).decode('utf-8')
                        
                        match_found = True
                        break  
                
                if match_found:
                    break
                    
                # Display "Scanning..." message on frame
                cv2.putText(frame, "Scanning... Please wait", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
            # # Clean up
            # cap.release()
            # cv2.destroyAllWindows()
            # os.remove(temp_path)
            
            if match_found:
                return render_template('matched_image.html', matched_image=img_str)
            else:
                return render_template('no_match.html')
                
        except Exception as e:
            print(f"Error: {str(e)}")
            return f"An error occurred: {str(e)}"

    return "Error processing the file"

if __name__ == '__main__':
    app.run(debug=True)