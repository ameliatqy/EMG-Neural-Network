import cv2
import cv2.aruco as aruco
import yaml
import numpy as np
 
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000) #6x6 bit aruco marker with 1000 ids
img = aruco.drawMarker(aruco_dict, 2, 200)
cv2.imwrite('img.png', img)
