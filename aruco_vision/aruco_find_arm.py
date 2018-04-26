import cv2
import cv2.aruco as aruco
import yaml
import numpy as np
import csv
import os, os.path

cap = cv2.VideoCapture(0)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000) #6x6 bit aruco marker with 1000 ids
parameters =  aruco.DetectorParameters_create() #detection method

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

DIR = '/Users/ameliatee/Documents/Classes/ME-6104Project/markers/data'

fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
fileid = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
videofilename_raw = 'video_raw/video_raw%i.mp4' %(fileid)
videofilename = 'video/video%i.mp4' %(fileid)
out_raw = cv2.VideoWriter(videofilename_raw, fourcc, 20.0, (width, height))
out = cv2.VideoWriter(videofilename, fourcc, 20.0, (width, height))

axis = np.float32([[-20,-20,0], [-20,20,0], [20,20,0], [20,-20,0],
                   [-20,-20,20],[-20,20,20],[20,20,20],[20,-20,20] ])

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
	

database = []

while(True):

    

    ret, frame = cap.read()
    out_raw.write(frame)
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

    print ids
    if ids is not None and len(ids) == 3:
      v1 = centers[1]-centers[0]
      v2 = centers[2]-centers[1]
      v1 = v1/np.linalg.norm(v1)
      v2 = v2/np.linalg.norm(v2)
      angle = np.arccos(-np.dot(v1,v2))
    else:
      angle = 'nan'

    print centers
    data_entry = [angle, centers[0][0], centers[0][1], centers[1][0], centers[1][1], centers[2][0], centers[2][1]]
    database.append(data_entry)
    out.write(frame)
    print data_entry


    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



filename = 'data/data%i.csv' %(fileid)
file = open(filename, 'wb')
with file:
    writer = csv.writer(file)
    writer.writerows(database)


 
cap.release()
out.release()
out_raw.release()
cv2.destroyAllWindows()
