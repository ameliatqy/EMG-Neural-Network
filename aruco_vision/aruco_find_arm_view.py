import cv2
import cv2.aruco as aruco
import yaml
import numpy as np
import pygame, sys
from pygame.locals import *

cap = cv2.VideoCapture(0)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000) #6x6 bit aruco marker with 1000 ids
parameters =  aruco.DetectorParameters_create() #detection method

axis = np.float32([[-20,-20,0], [-20,20,0], [20,20,0], [20,-20,0],
                   [-20,-20,20],[-20,20,20],[20,20,20],[20,-20,20] ])

pygame.init()
pygame.display.set_mode((100,100))

with open('calibration.yaml') as f:
  loadeddict = yaml.load(f)
camera_matrix = np.mat(loadeddict.get('camera_matrix'))
dist_coeffs = np.mat(loadeddict.get('dist_coeff'))

def draw(img, corners, imgpts):
  imgpts = np.int32(imgpts).reshape(-1,2)

  # draw ground floor in green
  img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

  # draw pillars in blue color
  for i,j in zip(range(4),range(4,8)):
      img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

  # draw top layer in red color
  img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
  return img

def calculate_center(corners):
	avg_center = (((corners[0]+corners[2])/2) + ((corners[1]+corners[3])/2))/2
	return (avg_center[0], avg_center[1])
	



while(True):

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    gray = aruco.drawDetectedMarkers(frame, corners)

    centers = np.zeros((3,2))
    
    
    if corners:
    	for i in range(0, len(corners)):
        	center = calculate_center(corners[i][0])
        	cv2.circle(frame, center, 20, (0,0,255), -1)
        	centers[ids[i]] = center

    #print ids, centers
    cv2.line(frame, (int(centers[0][0]), int(centers[0][1])), (int(centers[1][0]), int(centers[1][1])), (0, 255, 0), thickness=3, lineType=8)
    cv2.line(frame, (int(centers[1][0]), int(centers[1][1])), (int(centers[2][0]), int(centers[2][1])), (0, 255, 0), thickness=3, lineType=8)

    v1 = centers[1]-centers[0]
    v2 = centers[2]-centers[1]
    v1 = v1/np.linalg.norm(v1)
    v2 = v2/np.linalg.norm(v2)
    #print np.arccos(-np.dot(v1,v2))


    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

 
cap.release()
cv2.destroyAllWindows()
