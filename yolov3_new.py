# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np

image=cv2.imread('PennPed00096.png')

import os 
from pandas import DataFrame

file=open('PennPed00096.txt')


def get_boxes(lines):
  
  " input : file.txt labelled images, output: boxes list [xmin,ymin,xmax,ymax]"
  lines=lines[8:]
  list=[]
  list2=[]
  boxes=[]

  for i in range(len(lines)): 
    line=lines[i].split(':')
    if i%5==2:
      list.append(line[1].rstrip('\n'))
  

  for j in range (len(list)):
    list2.append(list[j].split('-')) 
    

    xmin,ymin=list2[j][0][2:-2].split(',')#comme c'est une chaîne de caractère, on n'en prend qu'un certain nombre
    xmax,ymax=list2[j][1][2:-1].split(',')
    boxes.append([int(xmin),int(ymin),int(xmax),int(ymax)])
    
  return boxes


#boxes=get_boxes(file)
#print(boxes)

def convert_boxes(boxes):
  "This function aims to convert boxes format to compare them with the prediction to train the network"
  convert_boxes=[]
  for i in range (len(boxes)):
    xmin,ymin,xmax,ymax=boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
    bx,by=xmin,ymin
    bw,bh=xmax-xmin,ymax-ymin
    convert_boxes.append([bx,by,bw,bh])
  return convert_boxes

#convert_boxes=convert_boxes(boxes)

def get_class(lines):
  lines=lines[8:]
  classes=[]
  for i in range(len(lines)):
    if i%5==0:
      classes.append(lines[i].split('"')[1])
  return classes
  
#print(get_class(file))

def index_center(boxes):
  "input : boxes [xmin,ymin,xmax,ymax], return the boxes center and its placement in the grid for the three scales"
  center=[]
  for i in range(len(boxes)):
    xmin,ymin,xmax,ymax=boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
    xcenter,ycenter = (xmax-xmin)/2,(ymax-ymin)/2
    xratio, yratio= xcenter/416, ycenter/416
    center_lg=[int(xratio*13),int(yratio*13)]
    center_md=[int(xratio*26),int(yratio*26)]
    center_sm=[int(xratio*52),int(yratio*52)]
    center.append([center_sm,center_md,center_lg])
  
  return center
#print(index_center(boxes))

def config(lines):
  boxes=get_boxes(lines)
  classes=get_class(lines)
  return boxes,classes

#print(config(lines))


"This script writes csv files with file path, boxes, classes"

path = os.getcwd()+'/PennFudanPed/Annotation/'
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
line={'file_path':[],'boxes':[],'classes':[]}

for file in files:
  file_path=path+file
  file = open(file_path, 'r')
  lines=file.readlines()
  boxes,classes=config(lines)
  line['file_path'].append(file_path.replace('txt','png').replace('Annotation','PNGImages'))
  line['boxes'].append(boxes)
  line['classes'].append(classes)
  
#print(line['boxes'][0])
boxes=line['boxes'][0]
  
df=DataFrame(line,columns=['file_path','boxes','classes'])
#df.to_csv(r'dataset_PennFudanPed.csv')
#print(df)

def reshape(image, input_size):
  "The function aim is to reshape images to 416x416 size"
  image=cv2.imread(image)
  h = image.shape[0]
  w = image.shape[1]

  ratio=max(h/input_size,w/input_size)
  
  image=cv2.resize(image,(int(h/ratio),int(w/ratio)))
  h = image.shape[0]
  w = image.shape[1]
  
  format=np.zeros((input_size,input_size,3))
  
  h0 = (input_size-h)//2
  h1=round((input_size-h)/2)
  w0 = (input_size-w)//2
  
  format[h0:(input_size-h1),w0:(input_size-w0),:]=image[:,:,:]
  
  return format

reshaped_image=reshape('PennPed00096.png',416)
#print(reshaped_image.shape)

#INPUT
import tensorflow as tf 

BBOXES_PER_CELL = 3
ANCHORS_SMALL = tf.constant([[10,13],[16,30],[33,23]], dtype=tf.float32) #we should do k-means clustering on our dataset
ANCHORS_MEDIUM = tf.constant([[30,61],[62,45],[59,119]], dtype=tf.float32)
ANCHORS_LARGE = tf.constant([[116,90],[156,198],[373,326]], dtype=tf.float32)
INPUT_SIZE = 416
CLASSES_LIST =['Distracted Person', 'Person']
DISPLAY_CLASSES_LIST = ['Distracted Person', 'Person']
COLORS = [(0,255,255),(0,255,0)]

import tensorflow as tf

"""
Clean and simple Keras implementation of network architectures described in:
    - (ResNet-50) [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf).
    - (ResNeXt-50 32x4d) [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf).
    
Python 3.
"""

from keras import layers
from keras import models


#
# image dimensions
#

img_height = 224
img_width = 224
img_channels = 3

#
# network params
#

cardinality = 3


def residual_network(x):
    """
    Input : image (416x416), output : tensor with bounding boxes (grid), 1 objectness_score, 3 classes 
    
    """
    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        return y

    def grouped_convolution(y, nb_channels, _strides):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y

    def residual_block(y, filtre_num1, filtre_num2, kernel_size1, kernel_size2):
        
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv2D(filtre_num1, kernel_size1, padding='same')(y)
        y = add_common_layers(y)
        
        y = layers.Conv2D(filtre_num2, kernel_size2, padding='same')(y)
        y = add_common_layers(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        #y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        #y = add_common_layers(y)

        #y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        #if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            #shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            #shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = layers.LeakyReLU()(y)

        return y

    # conv1
    x = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = add_common_layers(x)
    
    x = layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = add_common_layers(x)

    # conv2
    for i in range(1):
        x = residual_block(x, 32, 64, (1, 1), (3, 3))

    x = layers.Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = add_common_layers(x)

    # conv3
    for i in range(2):
        x = residual_block(x, 64, 128, (1, 1), (3, 3))

    x = layers.Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = add_common_layers(x)
    
    #conv4
    for i in range(8):
        x = residual_block(x, 128, 256, (1, 1), (3, 3))
        
    scale11=x
    
    x = layers.Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = add_common_layers(x)
    

    #conv5
    for i in range(8):
        x = residual_block(x, 256, 512, (1, 1), (3, 3))
        
    scale21 = x
    
    x = layers.Conv2D(1024, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = add_common_layers(x)
    
    #conv6
    for i in range(4):
        x = residual_block(x, 512, 1024, (1, 1), (3, 3))
        
    #conv7 
    for i in range(3):
      x=layers.Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
      x = add_common_layers(x)
      
      x=layers.Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
      x = add_common_layers(x)
    
    #scale3
    
    scale3=layers.Conv2D(255, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    scale3 = add_common_layers(scale3)
    
    #scale2
    
    scale22= layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    scale22= add_common_layers(scale22)
    
    scale22 = layers.UpSampling2D((2,2), interpolation='nearest')(scale22)
    
    scale2 = layers.Concatenate(axis=-1)([scale21, scale22])
    
    for i in range(3):
      scale2=layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same')(x) #kernel=1 nothing is changing
      scale2= add_common_layers(scale2)
      
      scale2=layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
      scale2= add_common_layers(scale2)
      
    scale12 = scale2
    
    scale2 = layers.Conv2D(255, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    scale2= add_common_layers(scale2)
    
    #scale1 
    
    scale12 = layers.Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    scale12= add_common_layers(scale12)
    
    """
    upsample : multiply image_size by 2
    """
    scale12 = layers.UpSampling2D((2,2), interpolation='nearest')(scale22)
    
    scale1 = layers.Concatenate(axis=-1)([scale11, scale12])
    
    for i in range(3):
      scale1=layers.Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
      scale1= add_common_layers(scale1)
      
      scale1=layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
      scale1= add_common_layers(scale1)
    
    scale1=layers.Conv2D(255, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    scale1= add_common_layers(scale1)

    return scale1,scale2,scale3


image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
network_output = residual_network(image_tensor)
  
model = models.Model(inputs=[image_tensor], outputs=[network_output[2]])
print(model.summary())

def IOU(b1, b2):
  """
  compare two side-to-side cells to define the large bounding box
  input : bounding boxes, objectness_score, classes, 
  """
  print(b1,b2)
  xmin1,ymin1,xmax1,ymax1=b1[0],b1[1],b1[2],b1[3]
  xmin2,ymin2,xmax2,ymax2=b2[0],b2[1],b2[2],b2[3]
  
  #intersection
  xminI,xmaxI=max(xmin1,xmin2),min(xmax1,xmax2)
  yminI,ymaxI=max(ymin1,ymin2),min(ymax1,ymax2)
  
  areaI=(ymaxI-yminI)*(xmaxI-xminI)
  
  #union
  area1=(ymax1-ymin1)*(xmax1-xmin1)
  area2=(ymax2-ymin2)*(xmax2-xmin2)
  areaU=area1+area2-areaI
  
  
  return areaI/areaU

print(IOU(boxes[0],boxes[0]))