import cv2 as cv
import numpy as np

cap = cv.VideoCapture('./parking5.mp4')
is_roi = 1


def draw_box(cam, rects):
    for r in rects:
        cv.rectangle(cam, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (0, 0, 255))

def save_parkinglot(rects):
    r = np.array(rects)
    np.save('./parkinglots', r)


while(True):
    ret, cam = cap.read()

    if(ret) :
        if(is_roi == 1):
            rects = cv.selectROIs('camera', cam, False, True)
            print('rects: ', rects)
            print(len(rects))
            save_parkinglot(rects)
            is_roi = 0
        draw_box(cam, rects)
        cv.imshow('camera', cam)
        if cv.waitKey(1) & 0xFF == ord('q'): # q 키를 누르면 닫음
            break
    else:
        break
                     
cap.release()
cv.destroyAllWindows()