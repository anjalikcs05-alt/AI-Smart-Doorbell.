import cv2
import winsound

# Load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Final Name Mapping: ID 1 = Anjali, ID 2 = Karan
names = ['None', 'Karan', 'Anjali']  

cam = cv2.VideoCapture(0)
print("\n [INFO] Smart Doorbell Active. Press 'q' to exit.")

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    for(x,y,w,h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # We keep the logic internal but don't print it to the terminal
        if (confidence < 115):
            name = names[id]
            color = (0, 255, 0) # Green for authorized
            winsound.Beep(1000, 200)
        else:
            name = "Unknown"
            color = (0, 0, 255) # Red for unauthorized
            winsound.Beep(400, 500)

        # Draw the box and the name on the camera screen
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        cv2.putText(img, name, (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('AI Smart Doorbell', img) 
    if cv2.waitKey(10) & 0xFF == ord('q'): 
        break

cam.release()
cv2.destroyAllWindows()