#from pix2pix import *
import os


def train():
    os.system("python pix2pix.py \
  --mode train \
  --output_dir facades_train \
  --max_epochs 1 \
  --input_dir former_dataset/train \
  --which_direction BtoA")
def infer():
    os.system("python pix2pix.py \
  --mode test \
  --output_dir facades_test \
  --input_dir former_dataset/val \
  --checkpoint facades_train")

def collect():
    os.system("cp -r recorder/* collected_dataset")

def reframe():
    os.system("cp -r recorder/* reframed_dataset")
    os.system("cp -r former_dataset/train/* reframed_dataset")







def retrain(str):
    if str=="collect":
        os.system("python pix2pix.py \
  --mode train \
  --output_dir facades_train \
  --max_epochs 1 \
  --input_dir collected_dataset \
  --which_direction BtoA")
    elif str=="reframe":
        os.system("python pix2pix.py \
  --mode train \
  --output_dir facades_train \
  --max_epochs 1 \
  --input_dir reframed_dataset \
  --which_direction BtoA")
    else:
        print("Wrong parameter")

