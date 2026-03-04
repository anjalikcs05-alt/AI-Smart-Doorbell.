import cv2
import numpy as np
import os
from PIL import Image

path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    faceSamples = []
    ids = []
    # Make sure folder names are exactly "Karan" and "Anjali" (Case Sensitive)
    name_to_id = {"Karan": 1, "Anjali": 2}

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("jpg"):
                imagePath = os.path.join(root, file)
                folder_name = os.path.basename(os.path.dirname(imagePath))
                user_id = name_to_id.get(folder_name)
                
                if user_id:
                    PIL_img = Image.open(imagePath).convert('L')
                    img_numpy = np.array(PIL_img,'uint8')
                    faces = detector.detectMultiScale(img_numpy)
                    for (x,y,w,h) in faces:
                        faceSamples.append(img_numpy[y:y+h,x:x+w])
                        ids.append(user_id)
    return faceSamples, ids

print("\n [INFO] Training system for Karan and Anjali. Please wait...")
faces, ids = getImagesAndLabels(path)

if len(ids) > 0:
    recognizer.train(faces, np.array(ids))
    if not os.path.exists('trainer'):
        os.makedirs('trainer')
    recognizer.write('trainer/trainer.yml') 
    print(f"\n [INFO] {len(np.unique(ids))} faces trained successfully!")
else:
    print("\n [ERROR] No images found. Check folder names and paths!")