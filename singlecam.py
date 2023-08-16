import cv2
import numpy as np
import argparse
import time
#import matplotlib.pyplot as plt
import time
import os
#from pydub import AudioSegment
#from pydub.playback import play
import datetime
import glob



net = cv2.dnn.readNet('D:/ME/Fyp_work/screw_best.weights', 'D:/ME/Fyp_work/screw.cfg')

classes = []
with open("D:/ME/Fyp_work/classes2.txt", "r") as f:
     classes = f.read().splitlines()
     
 
y = datetime.datetime.now()
y  = y.strftime("%Y-%m-%d %H:%M")
filename =  'Realtime_videos/'+str(y)+'.avi'
res = '1080p'

# Set resolution for the video capture
def change_res(capture, width, height):
    capture.set(3, width)
    capture.set(4, height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


# grab resolution dimensions and set video capture to it.
def get_dims(capture, res='720p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
    	width,height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(capture, width, height)
    return width, height

# Video Encoding, might require additional installs
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi'] 
    
#sound = AudioSegment.from_wav('/home/syed/Desktop/Single-Multiple-Custom-Object-Detection-master/1. Project - Custom Object Detection/zero.wav')     
path = 'Realtime_Results/'     
cap = cv2.VideoCapture(0)
#output1 = cv2.VideoWriter(filename, get_video_type(filename), 20.0, get_dims(cap, res))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output1 = cv2.VideoWriter(filename,fourcc, 20.0, (640,480))
#output1 = cv2.VideoWriter(filename, get_video_type(filename), 30, get_dims(cap, res))
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
  _, img = cap.read()
  height, width, _ = img.shape                  
  blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
  net.setInput(blob)
  output_layers_names = net.getUnconnectedOutLayersNames()
  layerOutputs = net.forward(output_layers_names)
  boxes = []
  confidences = []
  class_ids = []

  for output in layerOutputs:
      for detection in output:
          scores = detection[5:]
          class_id = np.argmax(scores)
          confidence = scores[class_id]
          if confidence > 0.2:
             center_x = int(detection[0]*width)
             center_y = int(detection[1]*height)
             w = int(detection[2]*width)
             h = int(detection[3]*height)
             x = int(center_x - w/2)
             y = int(center_y - h/2)

             boxes.append([x, y, w, h])
             confidences.append((float(confidence)))
             class_ids.append(class_id)
             

  indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.5)

  if len(indexes)>0:
     for i in indexes.flatten():
         x, y, w, h = boxes[i]
         label = str(classes[class_ids[i]])
         confidence = str(round(confidences[i],2))
         color = colors[i]
         cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
         cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
         x = datetime.datetime.now()
         x = x.strftime("%Y-%m-%d %H:%M:%S")
         #if confidence > str(0.5) and label == "gun"  :
           #val =  'KHI-MADRAS CHOWK-A406-Gun-Detected'+ str(x)+'.png'
           #v2.imwrite(os.path.join(path , val),img)
           #play(sound)
         
         
  output1.write(img)
  cv2.imshow('Camera 1', img)
  key = cv2.waitKey(1)
  if key ==27:
     #break
     cap.release()
     output1.release()
     cv2.destroyAllWindows()
      
            
