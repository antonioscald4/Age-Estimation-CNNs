import os, sys, cv2, csv, argparse, random
import numpy as np
from keras.models import Sequential, Model, save_model, load_model
import numpy as np
import keras.utils as image
from PIL import Image 

model1 = load_model("classification_age_model_InceptionV3_final_version_undersampling.h5", compile = False)
model2 = load_model("classification_age_model_vgg_final_test.h5", compile = False)
model3 = load_model("classification_age_model_vgg_final_test_folder_undersampling.h5", compile = False)
output_indexes = np.array([i for i in range(0, 82)])



def loadImage_Model1(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)    
    dim = (178,218)
    test_img =  cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    test_img = np.asarray(test_img)
    test_img = np.expand_dims(test_img, axis = 0)
    test_img = test_img / 255
    return test_img  
    
    
def loadImage_Model2(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)    
    dim = (224,224)
    test_img =  cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    test_img = np.asarray(test_img)
    test_img = np.expand_dims(test_img, axis = 0)
    test_img = test_img/ 255
    return test_img
    
def loadImage_Model3(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)    
    dim = (224,224)
    test_img =  cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    test_img = np.asarray(test_img)
    test_img = np.expand_dims(test_img, axis = 0)
    test_img = test_img/ 255
    return test_img
   
def soft_voting(img):
    
    percent_list1=[]
    percent_list2=[]
    percent_list3=[]
    predicts1 = (model1.predict(loadImage_Model1(img)))
    predicts2 = (model2.predict(loadImage_Model2(img)))
    predicts3 = (model3.predict(loadImage_Model3(img)))
    predicts1=predicts1.reshape(-1)
    predicts2=predicts2.reshape(-1)
    predicts3=predicts3.reshape(-1)
    for k in range(len(predicts1)):
      percent_list1.append("{0:.2%}".format(predicts1[k]))
      percent_list2.append("{0:.2%}".format(predicts2[k]))    
      percent_list3.append("{0:.2%}".format(predicts3[k]))    
    list_float1 = []
    list_float2 = []
    list_float3 = []

    for k in range(len(percent_list1)):
      var1 =str(percent_list1[k])
      var1 = var1[:-1]
      list_float1.append(float(var1))
      var2 =str(percent_list2[k])
      var2 = var2[:-1]
      list_float2.append(float(var2))
      var3 =str(percent_list3[k])
      var3 = var3[:-1]
      list_float3.append(float(var3))
    list_result = []
    for k in range(len(list_float1)):
      list_result.append(  (list_float1[k] + list_float2[k] + list_float3[k])/3)
    max_value = 0.0
    index = 0
    for k in range(len(list_result)):
      if(list_result[k] > max_value):
        max_value= list_result[k]
        index=k
    return index
  

    
def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--data", type=str, default='foo_test.csv', help="Dataset labels")
    parser.add_argument("--images", type=str, default='foo_test/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results.csv', help="CSV file of the results")
    args = parser.parse_args()
    return args

args = init_parameter()

# Reading CSV test file
with open(args.data, mode='r') as csv_file:
    gt = csv.reader(csv_file, delimiter=',')
    gt_num = 0
    gt_dict = {}
    for row in gt:
        gt_dict.update({row[0]: int(round(float(row[1])))})
        gt_num += 1
print(gt_num)

# Opening CSV results file
with open(args.results, 'w', newline='') as res_file:
    writer = csv.writer(res_file)
    # Processing all the images
    for image in gt_dict.keys():
        img = cv2.imread(args.images+image)
        if img.size == 0:
            print("Error")
        # Here you should add your code for applying your DCNN
        age  = soft_voting(img)
        # Writing a row in the CSV file
        writer.writerow([image, age])