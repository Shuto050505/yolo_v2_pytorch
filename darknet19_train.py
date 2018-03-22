import time
import cv2
import numpy as np
import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from darknet19 import *
from lib.image_generator import *


def main():
    # hyper parameters
    input_height, input_width = (224, 224)
    item_path = "./items"
    background_path = "./backgrounds"
    label_file = "./data/label.txt"
    backup_path = "backup_19"
    batch_size = 32
    max_batches = 3000
    learning_rate = 0.001
    lr_decay_power = 4
    momentum = 0.9
    weight_decay = 0.0005

    # load image generator
    print("loading image generator...")
    generator = ImageGenerator(item_path, background_path)

    with open(label_file, "r") as f:
        labels = f.read().strip().split("\n")

    # load model
    print("loading model...")
    model = Darknet19(num_classes=10, phase='train')

    # Load Weight
    # weight_path = ''
    # model.load_state_dict(weight_path)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.MSELoss()

    def lr_scheduler(optimizer, batch, lr_decay_power):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate * (1 - batch / max_batches) ** lr_decay_power
        return optimizer

    # start to train
    print("start training")
    for batch in range(max_batches):
        # generate sample
        x, t = generator.generate_samples(
            n_samples=batch_size,
            n_items=1,
            crop_width=input_width,
            crop_height=input_height,
            min_item_scale=0.3,
            max_item_scale=1.05,
            rand_angle=25,
            minimum_crop=0.8,
            delta_hue=0.01,
            delta_sat_scale=0.5,
            delta_val_scale=0.5
        )
        x = torch.from_numpy(x)
        #     x.cuda()
        x = Variable(x)

        one_hot_t = []
        for i in range(len(t)):
            one_hot_t.append(t[i][0]["one_hot_label"])
        one_hot_t = np.array(one_hot_t, dtype=np.float32)
        one_hot_t = torch.from_numpy(one_hot_t)
        #     one_hot_t.cuda()
        one_hot_t = Variable(one_hot_t)

        optimizer.zero_grad()
        y = model(x)
        loss = criterion(y, one_hot_t)
        loss.backward()
        optimizer.step()

        _, pred = torch.max(y.data, 1)
        _, answ = torch.max(one_hot_t.data, 1)
        accuracy = (pred == answ).sum() / batch_size

        print("[batch %d (%d images)] learning rate: %f, loss: %f, accuracy: %f" % (
            batch + 1, (batch + 1) * batch_size, optimizer.param_groups[0]['lr'], loss, accuracy))

        # update lr
        optimizer = lr_scheduler(optimizer, batch, lr_decay_power)  # Polynomial decay learning rate

        # save model
        if (batch + 1) % 1000 == 0:
            model_file = "%s/%s.pth" % (backup_path, batch + 1)
            print("saving model to %s" % (model_file))
            torch.save(model.state_dict(), model_file)

    print("saving model to %s/darknet_19_final.model" % (backup_path))
    torch.save(model.state_dict(), "%s/darknet_19_final.model" % (backup_path))


if __name__ == '__main__':
    main()