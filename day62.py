#day62-Object Detection Basics
import numpy as np
import matplotlib.patches as patches

#1.calculate IOU(Intersection Over Union) between two boxes -0=no overlap,1=perfect match
def iou(boxA,boxB):
    xA,yA=max(boxA[0],boxB[0]),max(boxA[1],boxB[1])
    xB,yB=min(boxA[2],boxB[2]),min(boxA[3],boxB[3])
    inter=max(0,xB-xA) *max(0,yB-yA)
    areaA=(boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB=(boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    union=areaA+areaB-inter
    return inter/union if union>0 else 0

def nms(dets,thres=0.5):
    dets=sorted(dets,key=lambda d:d['score'],reverse=True)
    keep=[]
    while dets:
        best=dets.pop(0)
        keep.append(best)
        dets=[d for d in dets if iou(best['box'],d['box'])<thres]
    return keep

#2.Sample data
img=np.random.rand(224,224,3)*0.3+0.4     #fake gray to draw
gt_box=(58,40,150,140) #ground truith box
pred_box=(60,50,160,145) #prediction box

#3.Detections
detections=[
    {'box':(50,40,150,140),'score':0.95}, #detction -high confidence
    {'box':(55,45,155,145),'score':0.88},#overlap detection
    {'box':(180,160,220,200),'score':0.90},#different objects
]
#4,Check predictions match
print(f"IoU(gt,pred)={iou(gt_box,pred_box):.3f}")

#5.Clean duplicate detction using NMS
kept=nms(detections)
print(f"Before NMS:{len(detections)} boxes") #3 boxes ,2 duplicates
print(f"After NMS:{len(kept)} boxes") #should be two boxes (1 merged +1 seperate)
