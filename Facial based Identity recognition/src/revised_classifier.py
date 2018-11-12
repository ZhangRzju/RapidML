import sys
import cv2
import tensorflow as tf
import facenet
import pickle
from sklearn.svm import SVC
import numpy as np
import math
import os

data_dir = './../data/image'
fr_model = './../data/model/20170512-110547.pb'
classifier_filename = './../data/classifier/test.pkl'
batch_size = 1000
seed = 666
min_num_img_per_class = 1
num_train_img_per_class = 1


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

def fp2ce(paths, embedding_size,img_placeholder,phase_train_placeholder,sess,embeddings):
        # Run forward pass to calculate embeddings
        print('Calculating features for images')
        num_img = len(paths)
        num_batch_per_epoch = int(math.ceil(1.0*num_img / batch_size))
        emb_array = np.zeros((num_img, embedding_size))
        for i in range(num_batch_per_epoch):
            start_index = i*batch_size
            end_index = min((i+1)*batch_size, num_img)
            paths_batch = paths[start_index:end_index]
            img = facenet.load_data(paths_batch, False, False, 160)
            feed_dict = { img_placeholder: img, phase_train_placeholder:False }
            emb_array[start_index:end_index] = sess.run(embeddings, feed_dict=feed_dict)
        return emb_array

def train():

    with tf.Graph().as_default():
        with tf.Session() as sess:
            np.random.seed(seed=seed)

            # splitting dataset
            dataset_tmp = facenet.get_dataset(data_dir)
            print(dataset_tmp[0])
            train_set, test_set = split_dataset(dataset_tmp, min_num_img_per_class, num_train_img_per_class)

            train_paths, train_labels = facenet.get_image_paths_and_labels(train_set)
            test_paths, test_labels = facenet.get_image_paths_and_labels(test_set)
            
            print('Train cls: ' + str(len(train_set)) + 'Train img: ' + str(len(train_paths)))
            print('Test cls: ' + str(len(test_set)) + 'Test img: ' + str(len(test_paths)))
            
            # Load model
            print('Loading feature extraction model')
            facenet.load_model(fr_model)

            # Get input / output tensors
            img_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            train_emb_array = fp2ce(train_paths,embedding_size,img_placeholder,phase_train_placeholder,sess,embeddings)
            test_emb_array = fp2ce(test_paths,embedding_size,img_placeholder,phase_train_placeholder,sess,embeddings)
            
            classifier_filename_exp = os.path.expanduser(classifier_filename)

            # TRAIN classifier
            print('Training Classifier')
            model = SVC(kernel='linear', probability=True)
            model.fit(train_emb_array, train_labels)

            # Create list of class names
            class_names = [cls.name.replace('_', ' ') for cls in train_set]

            # Saving Classifier model
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('Save classifier model to file "%s"' % classifier_filename_exp)

            # Classify images
            print('Testing Classifier')
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            print('Loaded classifier model from file "%s"' % classifier_filename_exp)

            predictions = model.predict_proba(test_emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

            for i in range(len(best_class_indices)):
                print('%4d %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

            accuracy = np.mean(np.equal(best_class_indices, test_labels))
            print('Accuracy: %.3f' % accuracy)

def main():
    train()
    
if __name__ == "__main__":
    sys.exit(main())
          
