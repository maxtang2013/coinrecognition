import numpy as np
import argparse
import cv2
import os
import os.path
import argparse

def delete_all_invalid_coins_in(dir):
    files = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    for f in files:
        if f.endswith('.png'):
            img = cv2.imread(f)
            if img.shape[0] > 800 or img.shape[1] > 800:
                cmd = "rm -rf " + f
                print(cmd + " with size({0},{1}".format(img.shape[0], img.shape[1]))
                os.system(cmd)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", dest="dir", required = True, help = "Path to the directory that contains all the original photos")
args = ap.parse_args()

delete_all_invalid_coins_in(args.dir)