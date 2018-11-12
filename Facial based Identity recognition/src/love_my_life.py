
'''
def main():

    # Set up camera

    # Load model

    # Run camera

        # Take img

        # Process (what do I need with it)

        # Infer
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import random
from time import sleep
import cv2

def preprocess_image(src):
    # scaling img
    network_width = 160
    network_height = 160
    preprocessed_image = cv2.resize(src, (network_width,network_height))

    return preprocessed_image

def crop_my_baby(vid_image):
##    print(vid_image)
    sleep(random.random())

    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess,None)


    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709

    random_key = np.random.randint(0, high=99999)
##    bounding_boxes_filename = os.path.join(
    scaled = preprocess_image(vid_image)
    try:
##        img = misc.imread(vid_image)
        img = vid_image
    except (IOError, ValueError, IndexError) as e:
        errorMessage='{}: {}'.format(vid_image, e)
        print(errorMessage)
    else:
        if img.ndim == 2:
            img = facenet.to_rgb(img)
        img = img[:,:,0:3]
##        print(img)

        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        
        nrof_faces = bounding_boxes.shape[0]
#        print('nrof_faces')
#        print(nrof_faces)
        
        if nrof_faces>0:
            print('nrof_faces' + str(nrof_faces))
            det = bounding_boxes[:,0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces>1:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                det_arr.append(np.squeeze(det))

            for i,det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-22, 0)
                bb[1] = np.maximum(det[1]-22, 0)
                bb[2] = np.minimum(det[2]+22, img_size[1])
                bb[3] = np.minimum(det[3]+22, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                scaled = misc.imresize(cropped, (160, 160), interp='bilinear')
                print('Success aligning')
                print('scaled_type' + str(type(scaled)) + 'scaled_dim')
                print(scaled.shape)
        else:
            print('scaled_type' + str(type(scaled)) + 'scaled_dim')
            print(scaled.shape)
    return scaled
                
