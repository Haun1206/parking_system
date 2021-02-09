import cv2 as cv
import numpy as np
import json as js
import time

cap = cv.VideoCapture('./parking5.mp4')
rects = np.load('./parkinglots.npy').tolist()

# YOLO 가중치 파일과 CFG 파일 로드
YOLO_net = cv.dnn.readNet("yolov3.weights","yolov3.cfg")

# YOLO NETWORK 재구성
classes = []
class_ids = []
confidences = []
boxes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = YOLO_net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in YOLO_net.getUnconnectedOutLayers()]


def car_detection(cam):
    h, w, c = cam.shape
    blob = cv.dnn.blobFromImage(cam, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    YOLO_net.setInput(blob)
    outs = YOLO_net.forward(output_layers)

    for out in outs:
        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                dw = int(detection[2] * w)
                dh = int(detection[3] * h)
                # Rectangle coordinate
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes.append([x, y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

def draw_box(cam):
    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)
    result = [0]*len(rects)
    is_find = 0
    for index, r in enumerate(rects):
        x_center = r[0] + r[2]/2
        y_center = r[1] + r[3]/2
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                score = confidences[i]
                if(x+w/2 < x_center + r[2]/2 and x+w/2 > x_center - r[2]/2 and y+h/2 < y_center + r[3]/2 and y+h/2 > y_center - r[2]/2):
                    cv.rectangle(cam, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (0, 0, 255))
                    result[index] = 1
                    is_find = 1
                    break
        if(not is_find):
            cv.rectangle(cam, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (0, 255, 0))
            result[index] = 0
        is_find = 0
    ##save result
    M = dict(zip(range(1, len(result) + 1), result))
    with open("./result.json", "w") as json_file:
        js.dump(M, json_file)

while(True):
    ret, cam = cap.read()

    if(ret) :
        car_detection(cam)
        draw_box(cam)
        cv.imshow('camera', cam)
        #time.sleep(10)
        #print("hello")
        if cv.waitKey(1) & 0xFF == ord('q'): # q 키를 누르면 닫음
            break
    else:
        break
                     
cap.release()
cv.destroyAllWindows()