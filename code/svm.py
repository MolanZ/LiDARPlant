import pandas as pd
from osgeo import gdal
from PIL import Image
import matplotlib.image as mpimg
from osgeo import ogr
import matplotlib.pyplot as plt
import IPython.display as display
import numpy as np
import pathlib
import os
import random
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import joblib

def get_allfile(path):  # 获取所有文件
    all_file = []
    for f in os.listdir(path):  #listdir返回文件中所有目录
        f_name = os.path.join(path, f)
        all_file.append(f_name)
    return all_file

def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
            if len(os.listdir(path)) == 1:
                list_name.append(file_path)
    return list_name


#read the GeoTIFF file
def read_img(dataset_path):
    dataset = gdal.Open(dataset_path)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)
    del dataset
    temp = np.where(np.isnan(im_data),0,im_data)
    temp = np.transpose(temp,(1,2,0))
    final = np.resize(temp,(100,100,9))
    final = final.flatten()
    return final

def load_data(txt, level, my_dict):
    path=txt
    imgs = []
    target = []
    for line in path:
        words = line.split('/')
        if len(words)< 6 or '.tif' not in line:
            continue
        label = int(my_dict[words[level+1]])
        image = read_img(line)
        imgs.append(image)
        target.append(label)
    flat_data=np.array(imgs)
    target=np.array(target)
    df=pd.DataFrame(flat_data)
    df['Target']=target
    return df

def train(level=1):
    file_list = listdir('data/single_tree',[])
    family = []
    genus = []
    spices = []
    for f in file_list:
        a = f.split('/')
        if len(a) != 6:
            continue
        family.append(a[2])
        genus.append(a[3])
        spices.append(a[4])
    value = []
    temp = []
    x = list(set(family))
    for i in range(len(x)):
        temp.append([x[i],str(i)])
    family_dict = dict(temp)
    f_class = len(x)
    value = []
    temp = []
    x = list(set(genus))
    for i in range(len(x)):
        temp.append([x[i],str(i)])
    genus_dict = dict(temp)
    g_class = len(x)
    value = []
    temp = []
    x = list(set(spices))
    for i in range(len(x)):
        temp.append([x[i],str(i)])
    spices_dict = dict(temp)
    s_class = len(x)

    if level == 1:
        df = load_data(txt=file_list, level=level,my_dict = family_dict)
    elif level == 2:
        df = load_data(txt=file_list, level=level,my_dict = genus_dict)
    else:
        df = load_data(txt=file_list, level=level,my_dict = spices_dict)
        df
    x=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    file_list
    param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
    svc=svm.SVC(probability=True)
    net=GridSearchCV(svc,param_grid)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,train_size=0.80, random_state=77,stratify=y)
    print('Splitted Successfully')
    net.fit(x_train,y_train)
    print('The Model is trained well with the given images')
    y_pred=net.predict(x_test)
    probability=net.predict_proba(x_test)
    print("The predicted Data is :")
    print(y_pred)
    print("The actual data is:")
    print(np.array(y_test))
    print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
    filename = 'svm_'+str(level)+'.sav'
    joblib.dump(net, filename)

train(level=1)
