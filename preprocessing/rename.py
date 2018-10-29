import numpy as np
import argparse
import cv2
import os
import os.path
import argparse

def resize_all_images_in(dir):
    files = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    dirs = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]
    for f in files:
        if f.endswith('.png'):
            cmd = "mv {0} {1}".format(f, f.replace(".png", "_new.png"))
            print(cmd)
            os.system(cmd)
    for d in dirs:
        resize_all_images_in(d)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", dest="dir", help = "Path to the directory that contains all the original photos", default="data")
args = ap.parse_args()

resize_all_images_in(args.dir)