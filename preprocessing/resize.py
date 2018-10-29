import numpy as np
import argparse
import cv2
import os
import os.path
import argparse

def resize_all_images_in(dir, to_width, to_height):
    files = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    dirs = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]
    for f in files:
        if f.endswith('.png'):
            print ("Processing "+ f)
            img = cv2.imread(f)
            resized = cv2.resize(img, (to_width, to_height))
            cv2.imwrite(f, resized)
    for d in dirs:
        resize_all_images_in(d, to_width, to_height)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--width", dest="width", help = "resized width", default=200)
ap.add_argument("-ht", "--height", dest="height", help = "resized height", default=200)
ap.add_argument("-d", "--dir", dest="dir", help = "Path to the directory that contains all the original photos", default="data")
args = ap.parse_args()

resize_all_images_in(args.dir, int(args.width), int(args.height))