from constants import *
import csv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from emotion_recognition import EmotionRecognition
import cv2
import sys

cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

image=Image.open("recorder/1.jpg")

#format jpg/png file to fit into network size
def format_image_norm(image):
    image = np.asarray(image)
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    # None is we don't found an image
    if not len(faces) > 0:
        return None
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    # Chop image to face
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    # Resize image to network size
    try:
        image = cv2.resize(image, (SIZE_FACE, SIZE_FACE),
                           interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+] Problem during resize")
        return None

    return image

#format jpg/png file (without normalization)
def format_image(image):
    image = np.asarray(image)
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    # None is we don't found an image
    if not len(faces) > 0:
        return None
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    # Chop image to face
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    # Resize image to network size
    try:
        image = cv2.resize(image, (SIZE_FACE, SIZE_FACE),
                           interpolation=cv2.INTER_CUBIC)
    except Exception:
        print("[+] Problem during resize")
        return None

    return image

def infer():
    
    #Load model
    network = EmotionRecognition()
    network.build_network()
    
    #Predict
    result = network.predict(format_image_norm(image))
    label = np.argmax(result[0])
    print("Emotion: " + str(EMOTIONS[np.argmax(result[0])]))
    return label

def collect():
    #Get predicted label
    labell = 0  #the type of picture
    
    #Change image to string of values
    data = format_image(image)
    l=data.tolist()
    new = []
    for sublist in l:
        for item in sublist:
            new.append(item)
    stri=' '.join([str(int(i)) for i in new])
    
    #Add to csv file
    row=[str(labell), stri,'Training' ]
    with open("./collected_dataset/collect.csv", 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        
def reframe():
    
    #Get data from collect.csv and add it to dataset
    with open("./collected_dataset/collect.csv", newline = '') as f_out, open("./former_dataset/fer2013.csv", 'a', newline='') as f_in:
        reader = csv.reader(f_out)
        next(reader) #skip first row
        writer = csv.writer(f_in)
        for row in reader:
            writer.writerow(row)
            
def train():
    os.system("python3 emotion_recognition.py")
def retrain(str):
    if str=="collect":
        os.system("python3 retrain.py")
    elif str=="reframe":
        os.system("python3 retrain2.py")
    else:
        print("Wrong parameter")
