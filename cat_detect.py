import torch
import cv2
import numpy as np
import time

checked = 0

def confirm_pepsi_spotted(parameter):
    global checked

    if parameter == 'pepsi':
        if checked <= 100:
            checked +=1
            return(checked)
        else:
            checked = 0
            return("___________________________________________________________________________-oh my gawd")


# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

model.conf = 5.0

lst = []

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame)

    cv2.imshow('YOLOv5', np.squeeze(result.render()))

    df = result.pandas().xyxy[0]
    for i in df['name']: # name->labels
        lst.append(i)
        print(confirm_pepsi_spotted(i))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()