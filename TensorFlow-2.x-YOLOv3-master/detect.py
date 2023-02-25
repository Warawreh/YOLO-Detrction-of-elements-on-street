#================================================================
#
#   File name   : detect_mnist.py
#   Author      : PyLessons
#   Created date: 2020-08-12
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : mnist object detection example
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import random
import time
import tensorflow as tf
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import detect_image
from yolov3.configs import *
import pandas as pd

yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}") # use keras weights

data = pd.read_csv('test.csv') 
shift = 0
data = data[shift:]
batch = 2500
print("Start")
for i in range(len(data)):
    image_path = '/content/TensorFlow-2.x-YOLOv3-master/mnist/mnist_train/' + data.iloc[i]['image_path']
    
    print(shift + i + 1 , '/' , 2092)
    # if((i + 1)% 100 == 0):
    #   print(i + 1 , '/' , len(data))
    out = "result" + str((i + shift) // batch * batch) + ".txt"
    boxes = detect_image(yolo, image_path, out, input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
