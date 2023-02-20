
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:  
  try:
    tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=8192)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

import pathlib
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import scipy.io
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import tensorflow as tf

import keras
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Activation
from keras.layers import Conv2D, AveragePooling2D
from keras.models import Model, Sequential, save_model, load_model

from keras import metrics

from keras import backend as K

#from keras.models import model_from_json
#import matplotlib.pyplot as plt

class_weights={}
samples = [0, 170, 370, 550, 1160, 4010, 4310, 3450, 2590, 1840, 1630, 750, 3460, 2800, 3920, 5292, 5865, 9174, 9164, 12820, 9088, 12434, 15001, 15001, 15001, 15001, 15001, 15001, 15001, 15001, 15001, 15001, 15001, 15001, 15001, 15001, 15001, 15001, 15001, 15001, 15001, 15001, 15001, 13998, 13452, 12976, 12363, 11921, 11361, 11179, 11001, 10752, 10563, 10456, 10080, 9371, 8689, 8332, 14690, 13630, 12172, 10638, 9060, 8044, 6622, 8145, 6861, 7480, 6464, 4872, 3932, 2816, 2148, 1584, 1196, 772, 564, 396, 224, 144, 76, 8]

max_sample=np.max(samples)


for i in range (len(samples)):
    if i == 0:
        class_weights[0]=0.0
    elif i > 57 and i < 70:
        class_weights[i]=5.0
    elif i > 69:
        class_weights[i]=20.0
    elif i<58:
        class_weights[i]=1.0
    
for key, value in class_weights.items():
    print ( key, ' : ', value)    




BATCH_SIZE = 128
img_size = (218,178)
data_dir = "/mnt/sdc1/2022_va_gr15/dataset_for_inceptionv3_undersampling/training_caip_contest/"
data_dir = pathlib.Path(data_dir)

gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
train_generator = gen.flow_from_directory(data_dir, target_size=img_size, subset='training',batch_size=BATCH_SIZE, class_mode='categorical',shuffle=True,seed=999)
val_generator = gen.flow_from_directory(data_dir, target_size=img_size, subset='validation',batch_size=BATCH_SIZE, class_mode='categorical',shuffle=True,seed=999)

base_model = load_model("CelebAModel/weights.best.inc.male.hdf5")
x = base_model.get_layer(index=-2).output
predictions = Dense(82, activation='softmax',name='dense_output')(x)

# this is the model we will train
age_model = Model(inputs=base_model.input, outputs=predictions)

sgd = keras.optimizers.SGD(learning_rate=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

#-----CUSTOM LOSS FUNCTION-----

# Categorical Cross-Entropy plus Age Accuracy and Regularity index (from Guess The Age Contest)

import keras.backend as K
import tensorflow as tf
import math

# calculates the MAE
def age_mae(y_true, y_pred):
  #compute the ages
  true_age = K.sum(y_true * K.arange(0,82, dtype="float32"), axis=-1) 
  pred_age = K.sum(y_pred * K.arange(0,82, dtype="float32"), axis=-1)
  MAE = K.mean(K.abs(true_age - pred_age))
  return MAE

# returns the list of indexes of y_true in which a label in the range [start, end] is stored
def get_array_index_in_range(y_true,start,end):
  indexes=[]
  i = 0
  true_labels = K.argmax(y_true,axis=1) # gets the array of the ages from y_true
  #true_labels = tf.reshape(true_labels,[-1]) # flattening
  for label in true_labels:
    i += 1
    if label <= end and label >= start: # check if the label is in the range
      indexes.append(i)
  return indexes

# returns the sub-parts, of the arrays y_true and y_pred, belonging to the age group i
def get_arrays_by_group(y_true,y_pred,indexes,i):
  list_of_true_group_i = []
  list_of_pred_group_i = []
  if indexes[i] is not None: # check if there are elements belonging to the age-group i
    for index in indexes[i]:
      list_of_true_group_i.append(y_true[i]) # creates the list of the true labels belonging to the age-group i
      list_of_pred_group_i.append(y_pred[i]) # creates the list of the predicted labels belonging to the age-group i
  return list_of_true_group_i, list_of_pred_group_i

#compute the AAR Loss
def AAR_Loss(y_true,y_pred):
  # compute the total MAE
  MAE_tot = age_mae(y_true,y_pred)
  # the following dictionary has the age-group as key and as values the indexes of y_true where an age belonging to that age-group is contained
  indexes = {1:get_array_index_in_range(y_true,1,10),2:get_array_index_in_range(y_true,11,20),3:get_array_index_in_range(y_true,21,30),4:get_array_index_in_range(y_true,31,40),5:get_array_index_in_range(y_true,41,50),6:get_array_index_in_range(y_true,51,60),7:get_array_index_in_range(y_true,61,70),8:get_array_index_in_range(y_true,71,100)}
  # get the sub-parts of y_true and y_pred for each age-group
  true_labels_group_1, pred_labels_group_1 = get_arrays_by_group(y_true,y_pred,indexes,1)
  true_labels_group_2, pred_labels_group_2 = get_arrays_by_group(y_true,y_pred,indexes,2)
  true_labels_group_3, pred_labels_group_3 = get_arrays_by_group(y_true,y_pred,indexes,3)
  true_labels_group_4, pred_labels_group_4 = get_arrays_by_group(y_true,y_pred,indexes,4)
  true_labels_group_5, pred_labels_group_5 = get_arrays_by_group(y_true,y_pred,indexes,5)
  true_labels_group_6, pred_labels_group_6 = get_arrays_by_group(y_true,y_pred,indexes,6)
  true_labels_group_7, pred_labels_group_7 = get_arrays_by_group(y_true,y_pred,indexes,7)
  true_labels_group_8, pred_labels_group_8 = get_arrays_by_group(y_true,y_pred,indexes,8)
  # computes the sum of the squares of the differences between the MAE on a single age group and the total MAE
  sigma = K.square(age_mae(true_labels_group_1,pred_labels_group_1)-MAE_tot) + K.square(age_mae(true_labels_group_2,pred_labels_group_2)-MAE_tot) + K.square(age_mae(true_labels_group_3,pred_labels_group_3)-MAE_tot) + K.square(age_mae(true_labels_group_4,pred_labels_group_4)-MAE_tot) + K.square(age_mae(true_labels_group_5,pred_labels_group_5)-MAE_tot) + K.square(age_mae(true_labels_group_6,pred_labels_group_6)-MAE_tot) + K.square(age_mae(true_labels_group_7,pred_labels_group_7)-MAE_tot) + K.square(age_mae(true_labels_group_8,pred_labels_group_8)-MAE_tot)
  # ends the computation of sigma 
  sigma = K.sqrt(1/8*sigma)
  # aggregates the two contributes
  L_aar = tf.cast(MAE_tot + sigma,tf.float32)
  return L_aar

# computes the AAR_Loss as described above
def Categorical_Crossentropy_plus_New_AAR_Loss(y_true, y_pred):
  # compute the first contribute
  cce = tf.keras.losses.CategoricalCrossentropy()
  #compute the AAR Loss
  L_aar = AAR_Loss(y_true,y_pred)
  return cce(y_true,y_pred) + L_aar
  
# computes the AAR_Loss WITHOUT cce
def New_AAR_Loss(y_true, y_pred):
  # compute the first contribute
  cce = tf.keras.losses.CategoricalCrossentropy()
  #compute the AAR Loss
  L_aar = AAR_Loss(y_true,y_pred)
  return L_aar

#only variance
def variance(y_true,y_pred):
  # compute the total MAE
  MAE_tot = age_mae(y_true,y_pred)
  # the following dictionary has the age-group as key and as values the indexes of y_true where an age belonging to that age-group is contained
  indexes = {1:get_array_index_in_range(y_true,1,10),2:get_array_index_in_range(y_true,11,20),3:get_array_index_in_range(y_true,21,30),4:get_array_index_in_range(y_true,31,40),5:get_array_index_in_range(y_true,41,50),6:get_array_index_in_range(y_true,51,60),7:get_array_index_in_range(y_true,61,70),8:get_array_index_in_range(y_true,71,100)}
  # get the sub-parts of y_true and y_pred for each age-group
  true_labels_group_1, pred_labels_group_1 = get_arrays_by_group(y_true,y_pred,indexes,1)
  true_labels_group_2, pred_labels_group_2 = get_arrays_by_group(y_true,y_pred,indexes,2)
  true_labels_group_3, pred_labels_group_3 = get_arrays_by_group(y_true,y_pred,indexes,3)
  true_labels_group_4, pred_labels_group_4 = get_arrays_by_group(y_true,y_pred,indexes,4)
  true_labels_group_5, pred_labels_group_5 = get_arrays_by_group(y_true,y_pred,indexes,5)
  true_labels_group_6, pred_labels_group_6 = get_arrays_by_group(y_true,y_pred,indexes,6)
  true_labels_group_7, pred_labels_group_7 = get_arrays_by_group(y_true,y_pred,indexes,7)
  true_labels_group_8, pred_labels_group_8 = get_arrays_by_group(y_true,y_pred,indexes,8)
  # computes the sum of the squares of the differences between the MAE on a single age group and the total MAE
  sigma = K.square(age_mae(true_labels_group_1,pred_labels_group_1)-MAE_tot) + K.square(age_mae(true_labels_group_2,pred_labels_group_2)-MAE_tot) + K.square(age_mae(true_labels_group_3,pred_labels_group_3)-MAE_tot) + K.square(age_mae(true_labels_group_4,pred_labels_group_4)-MAE_tot) + K.square(age_mae(true_labels_group_5,pred_labels_group_5)-MAE_tot) + K.square(age_mae(true_labels_group_6,pred_labels_group_6)-MAE_tot) + K.square(age_mae(true_labels_group_7,pred_labels_group_7)-MAE_tot) + K.square(age_mae(true_labels_group_8,pred_labels_group_8)-MAE_tot)
  # ends the computation of sigma 
  sigma = K.sqrt(1/8*sigma)
  return tf.cast(sigma, tf.float32)

def AAR_metric(y_true, y_pred):
  # compute the total MAE
  MAE_tot = age_mae(y_true,y_pred)
    # the following dictionary has the age-group as key and as values the indexes of y_true where an age belonging to that age-group is contained
  indexes = {1:get_array_index_in_range(y_true,1,10),2:get_array_index_in_range(y_true,11,20),3:get_array_index_in_range(y_true,21,30),4:get_array_index_in_range(y_true,31,40),5:get_array_index_in_range(y_true,41,50),6:get_array_index_in_range(y_true,51,60),7:get_array_index_in_range(y_true,61,70),8:get_array_index_in_range(y_true,71,100)}
  # get the sub-parts of y_true and y_pred for each age-group
  true_labels_group_1, pred_labels_group_1 = get_arrays_by_group(y_true,y_pred,indexes,1)
  true_labels_group_2, pred_labels_group_2 = get_arrays_by_group(y_true,y_pred,indexes,2)
  true_labels_group_3, pred_labels_group_3 = get_arrays_by_group(y_true,y_pred,indexes,3)
  true_labels_group_4, pred_labels_group_4 = get_arrays_by_group(y_true,y_pred,indexes,4)
  true_labels_group_5, pred_labels_group_5 = get_arrays_by_group(y_true,y_pred,indexes,5)
  true_labels_group_6, pred_labels_group_6 = get_arrays_by_group(y_true,y_pred,indexes,6)
  true_labels_group_7, pred_labels_group_7 = get_arrays_by_group(y_true,y_pred,indexes,7)
  true_labels_group_8, pred_labels_group_8 = get_arrays_by_group(y_true,y_pred,indexes,8)
  # computes the sum of the squares of the differences between the MAE on a single age group and the total MAE
  sigma = K.square(age_mae(true_labels_group_1,pred_labels_group_1)-MAE_tot) + K.square(age_mae(true_labels_group_2,pred_labels_group_2)-MAE_tot) + K.square(age_mae(true_labels_group_3,pred_labels_group_3)-MAE_tot) + K.square(age_mae(true_labels_group_4,pred_labels_group_4)-MAE_tot) + K.square(age_mae(true_labels_group_5,pred_labels_group_5)-MAE_tot) + K.square(age_mae(true_labels_group_6,pred_labels_group_6)-MAE_tot) + K.square(age_mae(true_labels_group_7,pred_labels_group_7)-MAE_tot) + K.square(age_mae(true_labels_group_8,pred_labels_group_8)-MAE_tot)
  # ends the computation of sigma 
  sigma = K.sqrt(1/8*sigma)
  sigma_fin = tf.cast(sigma, tf.float32)
  
  MAE_metric=0.0
  sigma_metric=0.0
  
  if (5-MAE_tot > 0):
    MAE_metric=5-MAE_tot
  else:
    MAE_metric=0.0
    
  if (5-sigma_fin > 0):
    sigma_metric=5-sigma_fin
  else:
    sigma_metric=0.0
      
  return MAE_metric+sigma_metric

#----END OF CUSTOM LOSS FUNCTION----

age_model.compile(loss=Categorical_Crossentropy_plus_New_AAR_Loss
                  , optimizer=keras.optimizers.Adam()
                  #, optimizer = sgd
                  , metrics=['accuracy', age_mae, variance, AAR_metric]
                 )

from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=1,
    restore_best_weights=True
)

checkpointer = ModelCheckpoint(
    filepath='classification_age_model_InceptionV3_final_version_undersampling.h5'
    , monitor = "val_loss"
    , verbose=1
    , save_best_only=True
    , mode = 'auto'
)

history = age_model.fit(train_generator,epochs= 40,validation_data = val_generator,class_weight=class_weights,callbacks=[checkpointer,early_stop], steps_per_epoch= train_generator.samples // BATCH_SIZE,validation_steps=val_generator.samples // BATCH_SIZE )
age_model.save_weights('classification_age_model_InceptionV3_final_version_weights_undersampling.h5')
from keras.models import save_model
save_model(age_model,'./classification_age_model_InceptionV3_final_version_folder_undersampling')

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig('InceptionV3_Accuracy_v2.png')
plt.clf()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig('InceptionV3_Loss_v2.png')
plt.clf()

plt.plot(history.history['age_mae'])
plt.plot(history.history['val_age_mae'])
plt.title('model age_mae')
plt.ylabel('age_mae')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig('InceptionV3_Age_Mae_v2.png')
plt.clf()

plt.plot(history.history['variance'])
plt.plot(history.history['val_variance'])
plt.title('model variance')
plt.ylabel('variance')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig('InceptionV3_Variance_v2.png')
plt.clf()


plt.plot(history.history['AAR_metric'])
plt.plot(history.history['val_AAR_metric'])
plt.title('model AAR_metric')
plt.ylabel('AAR_metric')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig('InceptionV3_AAR_metric_v2.png')
plt.clf()


np.savez(('history_InceptionV3_final_undersampling.npz'), history=history.history)

