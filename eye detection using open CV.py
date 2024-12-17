import cv2 

# Start video capture from the default camera
cap = cv2.VideoCapture(0)

# Load Haar cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

while cap.isOpened():
    # Read frames from the camera
    ret, img = cap.read()
    if not ret:  # Exit if there's an issue with the camera
        break
    
    # Convert the frame to grayscale (required for Haar cascades)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Draw rectangles and label detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv2.putText(img, "Face", (x, y-4), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        
        # Detect eyes within the face region
        eyes = eyes_cascade.detectMultiScale(gray, 2.3, 4)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 5)
            cv2.putText(img, "Eye", (ex, ey-3), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

    # Display the processed frame with detections
    cv2.imshow('img', img)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Properly release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()