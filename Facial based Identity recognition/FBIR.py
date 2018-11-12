# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 16:38:10 2018

@author: kzile
"""
import os
import sys
sys.path.insert(0, './src')
from revised_inference import run_inference
from revised_classifier import train
import revised_train_softmax
import facenet
import imageio
import numpy as np
import shutil

def reframe():
    
    path_to_new_aligned_data = './ready_dataset'
    new_data_tmp = facenet.get_dataset(path_to_new_aligned_data)
    
    # Directory to access processed images
    put_into_classifier_data = './data/image'
    
    print('Number of new persons adding to classifier: ' + str(len(new_data_tmp)))
    
    for person in new_data_tmp:
        
        # Decalring name of person
        name = person.name
        
        # Declaring number of images for this person
        nrof_image = len(person.image_paths)
        
        print('Number of images for {a}: {b}'.format(a=name, b=nrof_image))
        
        # Change directory to see if there is an existing folder for this person
        os.chdir(put_into_classifier_data)
        if not os.path.exists(name):
            os.mkdir(name)
            
        # Return to base directory
        os.chdir('../..')
        
        # preprocess image
        for n in range(nrof_image):
            # Get image file name
            filename = person.image_paths[n].replace(path_to_new_aligned_data, '')
            ## \\ in replace represents only a single \
            filename = filename.replace('\\' + name + '\\', '')
            shutil.copyfile(person.image_paths[n], put_into_classifier_data + '/' + name + '/' + filename)
        
def collected():
    
    # Load dataset
    path_to_new_data = './collected_dataset'
    
    # Creating a temporary folder for mtcnn_aligned images
    os.system('python ' +  'src/align/align_dataset_mtcnn.py ' + path_to_new_data + ' ' + path_to_new_data + '_rev' + ' ' + '--image_size 182 --margin 44')
    rev_path_to_new_data = './collected_dataset_rev'
    new_data_tmp = facenet.get_dataset(rev_path_to_new_data)
    
    # Directory to access processed images
    ready_to_use = './ready_dataset'
    
    print('Number of new persons adding to dataset: ' + str(len(new_data_tmp)))
    
    for person in new_data_tmp:
        
        # Decalring name of person
        name = person.name
        
        # Declaring number of images for this person
        nrof_image = len(person.image_paths)
        
        print('Number of images for {a}: {b}'.format(a=name, b=nrof_image))
        
        # Change directory to see if there is an existing folder for this person
        os.chdir(ready_to_use)
        if not os.path.exists(name):
            os.mkdir(name)
        os.chdir('../')
        
        # preprocess image
        for n in range(nrof_image):
            # Load image
            img = facenet.load_data([person.image_paths[n]], False, False, 160)
            
            # Reformat to remove 1st dimension (which is no. of image in this case always 1)
            img = np.squeeze(img, axis=0)
            
            # Get image file name
            filename = person.image_paths[n].replace(rev_path_to_new_data, '')
            ## \\ in replace represents only a single \
            filename = filename.replace('\\' + name + '\\', '')
            
            # Change directory ready_dataset then save
            # Note! This will overwrite any image files with the same name
            os.chdir(ready_to_use + '/' + name)
            imageio.imsave(filename, img)
            os.chdir('../..')
    
    # Deletes temporary folder with mtcnn_aligned images    
    shutil.rmtree(rev_path_to_new_data, ignore_errors=None, onerror=None)
            
    
run_inference()
    