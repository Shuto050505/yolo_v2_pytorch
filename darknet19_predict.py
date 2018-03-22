import time
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import argparse
from darknet19 import *

# argument parse
parser = argparse.ArgumentParser(description="指定したパスの画像を読み込み、darknet19でカテゴリ分類を行う")
parser.add_argument('path', help="クラス分類する画像へのパスを指定")
args = parser.parse_args()

# hyper parameters
input_height, input_width = (224, 224)
weight_file = "./backup/darknet19_final.model"
label_file = "./data/label.txt"
image_file = args.path

# read labels
with open(label_file, "r") as f:
    labels = f.read().strip().split("\n")

# read image
print("loading image...")
img = cv2.imread(image_file)
img = cv2.resize(img, (input_height, input_width))
img = np.asarray(img, dtype=np.float32) / 255.0
img = img.transpose(2, 0, 1)

# load model
print("loading model...")
model = Darknet19()
model.load_state_dict(weight_file)
model.cuda()

# forward
x_data = img[np.newaxis, :, :, :]
x = torch.from_numpy(x)
x.cuda()
x = Variable(x)

y = model(x)
_, prob = torch.max(y.data, 1)
cls = labels[prob]
print("%16s : %.2f%%" % (cls, prob*100))
