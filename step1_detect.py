import cv2

# 1. Load the pre-trained face 'map' provided by OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Access your laptop's camera (0 is usually the default internal webcam)
video_capture = cv2.VideoCapture(0)

print("System Active. Press 'q' to exit the camera feed.")

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    # Convert to grayscale (Face detection works faster in black and white)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Search for faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw a blue rectangle around every face found
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display the resulting video feed
    cv2.imshow('Doorbell Preview', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
video_capture.release()
cv2.destroyAllWindows()
