import pathlib
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import Sequential, Model, save_model, load_model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Add, MaxPool2D
from keras import optimizers
import os
import shutil
from tensorflow.keras.applications.inception_v3 import InceptionV3
import random


import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:  
  try:
    tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=8192)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)




#read labels
import csv
import numpy as np
# list to store all labels
labels = []

with open('dataset.csv', newline='') as file:
  reader = csv.reader(file, delimiter = ',')
  for row in reader:
    labels.append(row[1])
  labels = np.array(labels)
print(labels[0])

path = "/mnt/sdc1/2022_va_gr15/dataset_v3_augmentation/training_caip_contest/" 

for i in range(82):
  #if i != 0:
  if not os.path.exists(path+str(i)):
    os.mkdir(path+str(i))
  else:
    shutil.rmtree(path+str(i))
    os.mkdir(path+str(i))
 
for index in range(575073):
  shutil.move(path+str(index)+".jpg", path+str(labels[index])+"/")
'''
for i in range (82):
  path_remove=path+str(i)
  files = os.listdir(path_remove)
  image_number = len(files)
  print(image_number)
  if (image_number > 8000):
    cont = image_number -8000
    for k in range(cont):
      image = files[k]
      os.remove(path_remove+'/'+image)
'''

#data augmentation on the classes containing few samples
from PIL import Image
from PIL import ImageOps
import random

def trasformation(img, num_trasformation, i, img_name):
  
  if num_trasformation == 3:
    flipped_img = ImageOps.mirror(img)
    flipped_img.save(path+str(i)+"/flipped_"+str(img_name))
    plus_rotated_img = img.rotate(10)
    plus_rotated_img.save(path+str(i)+"/plus_rotated_"+str(img_name))
    minus_rotated_img = img.rotate(-10)
    minus_rotated_img.save(path+str(i)+"/minusus_rotated_"+str(img_name))
 
  if num_trasformation == 2:
    y = random.randint(1,3)
    if y == 1:
      flipped_img = ImageOps.mirror(img)
      flipped_img.save(path+str(i)+"/flipped_"+str(img_name))
      plus_rotated_img = img.rotate(10)
      plus_rotated_img.save(path+str(i)+"/plus_rotated_"+str(img_name))
    if y == 2:
      plus_rotated_img = img.rotate(10)
      plus_rotated_img.save(path+str(i)+"/plus_rotated_"+str(img_name))
      minus_rotated_img = img.rotate(-10)
      minus_rotated_img.save(path+str(i)+"/minusus_rotated_"+str(img_name))
    if y == 3:
      flipped_img = ImageOps.mirror(img)
      flipped_img.save(path+str(i)+"/flipped_"+str(img_name))
      minus_rotated_img = img.rotate(-10)
      minus_rotated_img.save(path+str(i)+"/minusus_rotated_"+str(img_name))
  
  if num_trasformation == 1:
    y = random.randint(1,3)
    if y == 1:
      flipped_img = ImageOps.mirror(img)
      flipped_img.save(path+str(i)+"/flipped_"+str(img_name))
    if y == 2:
      plus_rotated_img = img.rotate(10)
      plus_rotated_img.save(path+str(i)+"/plus_rotated_"+str(img_name))
    if y == 3:
      minus_rotated_img = img.rotate(-10)
      minus_rotated_img.save(path+str(i)+"/minusus_rotated_"+str(img_name))
  
  if num_trasformation == 7:
    flipped_img = ImageOps.mirror(img)
    flipped_img.save(path+str(i)+"/flipped_"+str(img_name))
    plus_rotated_img = img.rotate(10)
    plus_rotated_img.save(path+str(i)+"/plus_rotated_"+str(img_name))
    minus_rotated_img = img.rotate(-10)
    minus_rotated_img.save(path +str(i)+"/minusus_rotated_"+str(img_name))
    plus_rotated_img = img.rotate(15)
    plus_rotated_img.save(path+str(i)+"/plus_rotated_15"+str(img_name))
    minus_rotated_img = img.rotate(-15)
    minus_rotated_img.save(path +str(i)+"/minusus_rotated_15"+str(img_name))
    plus_rotated_img = img.rotate(20)
    plus_rotated_img.save(path+str(i)+"/plus_rotated_20"+str(img_name))
    minus_rotated_img = img.rotate(-20)
    minus_rotated_img.save(path +str(i)+"/minusus_rotated_20"+str(img_name))
    #cropping
    width, height = img.size
    zoomed_img = img.crop((width/6,height/6,5*width/6,5*height/6))
    zoomed_img.save(path +str(i)+"/zoomed_1"+str(img_name))
    zoomed_img = img.crop((width/4,height/4,3*height/4,3*height/4))
    zoomed_img.save(path +str(i)+"/zoomed_2"+str(img_name))


#data augmentation on the classes containing few samples

for i in range(13):
  for img_name in os.listdir(path+str(i)):
    img = Image.open(path+str(i)+"/"+str(img_name))
    trasformation(img,7,i,img_name)

for i in range(13,16): #quadruplicate the images from 0 to 15 years
  for img_name in os.listdir(path+str(i)):
    img = Image.open(path+str(i)+"/"+str(img_name))
    trasformation(img, 3, i, img_name)

for i in range(16,18): #triplicate the images from 16 to 17 years
  for img_name in os.listdir(path+str(i)):
    img = Image.open(path+str(i)+"/"+str(img_name))
    trasformation(img, 1, i, img_name)





for img_name in os.listdir(path+str(18)): #double the images in the class 18
  img = Image.open(path+str(18)+"/"+str(img_name))
  trasformation(img, 1, 18, img_name)

cnt = 0
for img_name in os.listdir(path+str(19)): #take to 9000 the number of samples from the class 19
  if cnt < 9000:
    img = Image.open(path+str(19)+"/"+str(img_name))
    trasformation(img, 1, 19, img_name)
    cnt +=1

for i in range(58,63): #take to 8000 the classes from 58 to 62 years 
  cnt = 0
  for img_name in os.listdir(path+str(i)):
    if cnt < 8000:
      img = Image.open(path+str(i)+"/"+str(img_name))
      trasformation(img, 1, i, img_name)
      cnt += 1

for i in range(63,65): #double the images from 63 to 64 years
  for img_name in os.listdir(path+str(i)):
    img = Image.open(path+str(i)+"/"+str(img_name))
    trasformation(img,1,i,img_name)

for i in range(65,67): #triplicate the images from 65 to 66 years
  for img_name in os.listdir(path+str(i)):
    img = Image.open(path+str(i)+"/"+str(img_name))
    trasformation(img,2,i,img_name)

for i in range(67,82): #quadruplicate the images from 67 years on
  for img_name in os.listdir(path+str(i)):
    img = Image.open(path+str(i)+"/"+str(img_name))
    trasformation(img,3,i,img_name)


