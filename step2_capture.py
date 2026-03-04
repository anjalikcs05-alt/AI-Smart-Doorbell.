import cv2
import os

# 1. Use the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

# 2. SET NAME TO KARAN
name = "karan" 
path = f'dataset/{name}'
if not os.path.exists(path):
    os.makedirs(path)

count = 0
print("Starting. Look at the camera. Press 'q' to stop early.")

while count < 50: 
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        # Crop only the face part
        face_img = gray[y:y+h, x:x+w]
        # Save the image into your Karan folder
        cv2.imwrite(f"{path}/{count}.jpg", face_img)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Saving Photo: {count}/50", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Capturing Your Face', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Done! 50 images saved in {path}")
video_capture.release()
cv2.destroyAllWindows()
