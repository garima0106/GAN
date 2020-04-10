from __future__ import print_function
from SSH.test import detect
import cv2
from argparse import ArgumentParser
import os
from utils.get_config import cfg_from_file, cfg, cfg_print
import caffe
import glob
from PIL import Image
import numpy as np
from os.path import isfile, join

def convert_file(net,filename,output,cfg):
 
  assert os.path.isfile(filename),'Please provide a path to an existing image!'
  pyramid = True if len(cfg.TEST.SCALES)>1 else False
  cls_dets,_ = detect(net,filename,visualization_folder=output,visualize=True,pyramid=pyramid)

def extract_frame(video):

  vidcap = cv2.VideoCapture(video)
  success,image = vidcap.read()
  count = 0

  while success:
    cv2.imwrite("videos/frame%d.jpg" % count, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1

def vid_out():

  image_folder = '/home/ubuntu/SSH/out_vid/'
  video_name = 'portrait_out.avi'
  os.chdir(image_folder)

  images = [img for img in os.listdir(image_folder) 
              if img.endswith(".jpg") or
                 img.endswith(".jpeg") or
                 img.endswith("png")] 
  
  frame = cv2.imread(os.path.join(image_folder, images[0]))
  height, width, layers = frame.shape
  video = cv2.VideoWriter(video_name ,0, 10, (width, height))

  for image in images:  
      video.write(cv2.imread(os.path.join(image_folder, image)))
 
  cv2.destroyAllWindows()  
  video.release()
def bounding_box():

  prototxt = 'SSH/models/test_ssh.prototxt'
  model = 'data/SSH_models/SSH.caffemodel'  
  output = '/home/ubuntu/SSH/videos/'
  #cfg = 'SSH/configs/wider_pyramid.yml'

  #Load the network 
  cfg.GPU_ID = 0
  caffe.set_mode_gpu()
  caffe.set_device(0)
  assert os.path.isfile(prototxt),'Please provide a valid path for the prototxt!'
  assert os.path.isfile(model),'Please provide a valid path for the caffemodel!'
  
  
  print('Loading the network...', end="")  
  net = caffe.Net(prototxt, model, caffe.TEST)
  net.name = 'SSH'
  print('Done!')
  
  for filename in glob.glob('videos/*.jpg'):
     convert_file(net,filename,output,cfg) 
  
video = "portrait.mp4"
#extract_frame(video)    
#bounding_box()
vid_out()
