from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import os
import cv2

def readImagesFrom(dir):
    """ 
    Read all .png files in a directory (not include sub-dirs)
    """
    images = []
    files = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    labels = []
    for f in files:
        if f.endswith('.png'):
            im = cv2.imread(f, 0)
            images = np.append(images, im)
            labels = np.append(labels, [os.path.basename(dir)])
    return images, labels

def readDataset(dir, image_height, image_width):
    """
    Read a dataset from a two level directory, sub-directory name will be treated as labels for all images inside.

 ---data
    |----50c
    |     |---0.png
    |     |---1.png
    |
    |___20c
          |_0.png
          |_1.png
    """
    # get all sub-directories
    dirs = [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]

    dataset = []
    labels = []
    for d in dirs:
        images, lbs = readImagesFrom(os.path.join(dir,d))
        dataset = np.append(dataset, images)
        labels = np.append(labels,lbs)


    dataset = dataset.reshape([-1, image_height, image_width, 1])
    return dataset,labels
    
