# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 10:42:35 2018

@author: HP
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import argparse
import sys,os
import shutil
import xlrd
import xlutils
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import csv
from test1 import *
from pre_processing1 import *



path = './recorder'

#图片命名格式为： 'label1_train1',...,'labeln_trainn',...
#Label统一用数字表示（01-25）（一共25类）
#图片全部在文件夹train2里（已经分好类了），没有分好类的在train_img里


#path='D:/ZJU/train_img'
path2 = './train2'

def count(path):
    ls = os.listdir(path)
    count = 0
    for i in ls:
        if os.path.isfile(os.path.join(path,i)):
            count += 1
    return count



def get_file_names(path):
    name_list=[]
    dirs = os.listdir(path)
    for dir in dirs:
        #print(dir)
        name_list.append(dir)
    return name_list
#print(get_file_names('D:/ZJU/test_img'))




def train():
    os.system("python retrain.py")

#输出识别结果，需要事先把图片的名字存在一个csv文件中（此处为test.csv）

def infer():
    src=os.path.join('.','test_img')
    dest=os.path.join('.','test_img2')
    labels='./output_labels.txt'
    graph='./output_graph.pb'
    input_layer='DecodeJpeg/contents:0'
    output_layer='final_result:0'
    num_top_predictions=1
    labels = load_labels(labels)
    load_graph(graph)
    run_graph(src,dest,labels,input_layer,output_layer,num_top_predictions)
    
    
    
#要变成数据集的图片全部在文件夹'new_img_train'里面，我们要把所有图片存到‘collected_dataset’这个文件夹里面
name_list1=get_file_names(path) #包含所有图片名字的List


#标签和命名序号对应关系
dict1={'beans':'01','cake':'02','candy':'03','cereal':'04','chips':'05','chocolate':'06','coffee':'07','corn':'08','fish':'09','flour':'10','honey':'11','jam':'12','juice':'13','milk':'14','nuts':'15','oil':'16','pasta':'17','rice':'18','soda':'19','spices':'20','sugar':'21','tea':'22','tomatosauce':'23','vinegar':'24','water':'25'}

#所有图片文件命名格式为： e.g:01_1.png,10_2.png
#collect函数会把一个文件夹(recorder)里的所有图片先经过一系列处理后，将他们制作成一个新的数据集
def collect():
    with open('label_for_new_img.csv', 'w') as csvfile:
        fieldnames = ['image_id', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(count(path)):
            a=name_list1[i][:2]
            for j in dict1:
                if dict1[j]==a: 
                    writer.writerow({'image_id':name_list1[i][:-4] , 'label':j})
    shutil.move('./label_for_new_img.csv','./collected_dataset')
                    
#给定新图片,在recorder中，将其保存至reframed_dataset中                  
def reframe():
    a=get_file_names('./recorder')
    for i in a:
        shutil.copy('./recorder/'+i,'./reframed_dataset/train_img')
    
    with open('label_for_new_img2.csv','w') as csvfile:
        fieldnames = ['image_id', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(count(path)):
            a=name_list1[i][:2]
            for j in dict1:
                if dict1[j]==a: 
                    writer.writerow({'image_id':name_list1[i][:-4] , 'label':j})
    reader=csv.DictReader(open('./label_for_new_img2.csv'))
    header=reader.fieldnames   
    with open('./reframed_dataset/train.csv','a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writerows(reader)
def retrain(str):
    if str=="collect":
        os.system("python3 retrain2.py")
    elif str=="reframe":
        os.system("python3 retrain3.py")
    else:
        print("Wrong parameter")
    

#collect()
#reframe()  
        
        
    
    

        

    
    
    
    
    
    
        
    

    
    
    
    
    
