#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:10:30 2020

@author: garima
"""

import cv2
import mtcnn
import glob
print(mtcnn.__version__)



from matplotlib import pyplot
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
#model = MTCNN(weights_file= 'faceDetector.npy')

#load the images

faces=glob.glob('/Users/garima/GAN/2019-computefest/Wednesday/auto_encoder/celeba/img_align_celeba/*.jpg')
#load image from file
filename = faces[0]


pixels=plt.imread(filename)

#create detector using default weights
detector = MTCNN()

# detect faces in the image
face= detector.detect_faces(pixels)

for f in face:
    print(f)

#draw rectange on the image by plotting the image
x,y, width, height= f['box'] 
rect = Rectangle((x,y), width, height, fill=False, color='red')
print(rect)

#dot around eyes, ears and mouth 
for key, value in f['keypoints'].items():
    dot=Circle(value, radius=2, color='red')
    ax.add_patch(dot)

print(dot)



def draw_image_with_boxes(filename, result_list):
    #load image
    pixels=plt.imread(filename)
    #plot the image
    plt.imshow(pixels)
    # get the context for drawing boxes
    ax= plt.gca()
    
    #plot each box
    for result in result_list:
      #get coordinates
      x,y, width, height= result['box']  
      # create the rectangle around the face
      rect=Rectangle((x,y), width, height, fill=False, color='red')
      
      #draw the box
      ax.add_patch(rect)
      
      #draw the dots
      for key, value in result['keypoints'].items():
          
          dot= Circle(value, radius=2, color='red')
          ax.add_patch(dot)
      # show the plot
    plt.show()
        

draw_image_with_boxes(filename,face)


filename1=faces[1]
pixels=plt.imread(filename1)
face=detector.detect_faces(pixels)

draw_image_with_boxes(filename1,face)

swimteamFile='/Users/garima/GAN/swimteam_test.jpg'
p = plt.imread(swimteamFile)

plt.imshow(p)

face=detector.detect_faces(p)

draw_image_with_boxes(swimteamFile, face)

#extract and print each detected face

#draw each face seperate;t

def draw_faces(filename, result_list):
    #load the image
    data= plt.imread(filename)
    
    #plot each face as subplot
    for i in range(len(result_list)):
        
        #get coordinates
        x1,y1, width,height=result_list[i]['box']
        x2,y2=x1+width,y1+height
        
        #define subplot
        plt.subplot(1,len(result_list), i+1)
        plt.axis('off')
        
        #plot face
        plt.imshow(data[y1:y2, x1:x2])
        
    #show the plot
    plt.show()

draw_faces(swimteamFile, face)


