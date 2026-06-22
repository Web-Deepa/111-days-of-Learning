#day65-Face Recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

#1. Download a sample face image
url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
urllib.request.urlretrieve(url, "face_test.jpg")

#2. Load Haar Cascade 
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(cascade_path)

# 3.Load image and convert to grayscale 
img  = cv2.imread("face_test.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 4.Detect faces
faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,     # reduce size
    minNeighbors=5,      
    minSize=(30,30)      
)
print(f"Faces detected: {len(faces)}")

#5. Draw boxes around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    print(f"  Face at x={x}, y={y}, width={w}, height={h}")

# 6.Show result
plt.figure(figsize=(6,6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title(f"Detected {len(faces)} face(s)")
plt.savefig("face_detection.png")
plt.show()

#7. Eye detection inside detected face 
eye_path     = cv2.data.haarcascades + 'haarcascade_eye.xml'
eye_detector = cv2.CascadeClassifier(eye_path)

for (x, y, w, h) in faces:
    face_roi = gray[y:y+h, x:x+w]               # crop to face region only
    eyes     = eye_detector.detectMultiScale(face_roi)
    print(f"  Eyes found in this face: {len(eyes)}")