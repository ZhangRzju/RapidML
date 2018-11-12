# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:21:56 2018

@author: HP
"""

import os
import csv


def count(path):      
    ls = os.listdir(path)
    count = 0
    for i in ls:
        if os.path.isfile(os.path.join(path,i)):
            count += 1
    print(count)



def get_file_names(path):
    name_list=[]
    dirs = os.listdir(path)
    for dir in dirs:
        #print(dir)
        name_list.append(dir)
    return name_list
#print(get_file_names('D:/ZJU/test_img'))


