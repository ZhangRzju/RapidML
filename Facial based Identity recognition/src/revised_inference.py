import sys
import cv2
import tensorflow as tf
sys.path.insert(0, './../src')
import facenet
from love_my_life import crop_my_baby
import pickle
from sklearn.svm import SVC
import numpy as np
import math
import os


CV_WINDOW_NAME = "FaceNet"

CAMERA_INDEX = 0
REQUEST_CAMERA_WIDTH = 640
REQUEST_CAMERA_HEIGHT = 480

data_dir = './../data/image'
fr_model = './../data/model/20170512-110547.pb'
classifier_filename = './../data/classifier/test.pkl'
batch_size = 1000
seed = 666
min_num_img_per_class = 1
num_train_img_per_class = 5


def preprocess_image(src):
    # scaling img
    network_width = 160
    network_height = 160
    preprocessed_image = cv2.resize(src, (network_width,network_height))

    return preprocessed_image

def split_dataset(dataset, min_num_img_per_class, num_train_img_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min no. of imgs
        if len(paths)>=min_num_img_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:num_train_img_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[num_train_img_per_class:]))

    return train_set, test_set



def fp2ce_adjusted(notice_me, embedding_size,img_placeholder,phase_train_placeholder,sess,embeddings):
        # Run forward pass to calculate embeddings
        print('Calculating features for images')
        num_img = 1
        emb_array = np.zeros((num_img, embedding_size))
        feed_dict = { img_placeholder: notice_me, phase_train_placeholder:False }
        emb_array = sess.run(embeddings, feed_dict=feed_dict)
        return emb_array 

def run_inference():

    with tf.Graph().as_default():
        with tf.Session() as sess:
            np.random.seed(seed=seed)

            # Set up the camera
            camera_device = cv2.VideoCapture(CAMERA_INDEX)
            camera_device.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_CAMERA_WIDTH)
            camera_device.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_CAMERA_HEIGHT)

            actual_width = camera_device.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = camera_device.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print('actual camera resolution [WxH]: ' + str(actual_width) + 'x' + str(actual_height))

            if ((camera_device == None) or (not camera_device.isOpened())):
                print ('Could not open camera.  Make sure it is plugged in.')
                print ('Also, if you installed python opencv via pip or pip3 you')
                print ('need to uninstall it and install from source with -D WITH_V4L=ON')
                print ('Use the provided script: install-opencv-from_source.sh')
            frame_count = 0

            cv2.namedWindow(CV_WINDOW_NAME)

            # Load model
            print('Loading feature extraction model')
            facenet.load_model(fr_model)

            # Get input / output tensors
            img_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            while True:
                ret_val, vid_image = camera_device.read()
                if not ret_val:
                    print('No image from camera, exiting')
                    break
                
                frame_count += 1
                frame_name = 'camera frame' + str(frame_count)
                print(frame_name)
                print('ok what is vid_image originally')
#                print(vid_image.shape)

                notice_me = crop_my_baby(vid_image)
                notice_me = np.expand_dims(notice_me,axis=0)
#                print(notice_me.shape)
                emb_array = fp2ce_adjusted(notice_me, embedding_size,img_placeholder,phase_train_placeholder,sess,embeddings)
                classifier_filename_exp = os.path.expanduser(classifier_filename)
                
               # Classify images
                print('Testing Classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
#                print(predictions)
                best_class_indices = np.argmax(predictions, axis=1)
#                print(best_class_indices)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                print(best_class_probabilities)
                print('len' + str(len(best_class_indices)))

                for i in range(len(best_class_indices)):
                    print('%4d %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

                prop_val = cv2.getWindowProperty(CV_WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO)
                if (prop_val < 0.0):
                    print('window closed')
                    break

                cv2.imshow(CV_WINDOW_NAME, vid_image)
                raw_key = cv2.waitKey(1)

    
def main():
    run_inference()
    
if __name__ == "__main__":
    sys.exit(main())
          
