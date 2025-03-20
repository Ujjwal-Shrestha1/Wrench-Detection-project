
# Use this code to get the pixel of known length
# then create a ratio variable which equal to length_in_cm/length_in_pixel
import cv2 as cv
import math
path = 'C:/Users/shres/OneDrive/Desktop/Robotics module/project/Yolov8/data/images/val/pic15_Color.png'
img = cv.imread(path)

points = []
def draw_circle(event,x,y, flags , params):
    global points
    if event == cv.EVENT_LBUTTONDOWN:
        if len(points) == 2:
            points = []
        points.append((x,y))

cv.namedWindow('Frame')
cv.setMouseCallback('Frame',draw_circle)

while True:
    frame = cv.imread(path) 

    for pt in points:
        cv.circle(frame,pt,5,(25,15,255,),-1)

    # measure distance 
    if len(points)== 2:
        pt1 = points[0]
        pt2 = points[1]

        distance_px = math.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1])

        
        cv.putText(frame,fr'{int(distance_px)}',(pt1[0], pt1[1]-10),cv.FONT_HERSHEY_PLAIN,2.5, (120,0,200),2)
    
    cv.imshow('Frame',frame)
    key = cv.waitKey(1)

    if key == 27:
        break

cap.release()
cv.destroyAllWindows()