#Day63-YOLO(You Only Look Once)
import urllib.request
import cv2
from ultralytics import YOLO

#1.Load pretrained YOLOv8 model
model=YOLO('yolov8n.pt')
print(f"Model can detect {len(model.names)} classes")

#2. Download a sample test image
url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"
urllib.request.urlretrieve(url, "test_image.jpg")

#3.Run detection
res=model('test_image.jpg')

#5.print all detections found
print(f"Detected {len(res[0].boxes)}objects:")
for box in res[0].boxes:
    cls_id=int(box.cls[0]) #class index
    conf=float(box.conf[0]) #confidence score
    name=model.names[cls_id] #class name
    print(f"name:{name:<12} confidence={conf:.2f}")

#6.Filter
result_filtered=model('test_image.jpg',conf=0.5)
print(f"High confidence only: {len(result_filtered[0].boxes)} objects")

#7.Detect only one specific class
p_id=[k for k,v in model.names.items() if v=='person'][0]
res_person=model('test_image.jpg',classes=[p_id])
print(f"People detected: {len(res_person[0].boxes)}")

