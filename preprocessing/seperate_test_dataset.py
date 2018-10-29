import numpy as np
import argparse
import cv2
import os
import os.path
import argparse

def random_seperate(dir, to_dir, percent=0.5):
    files = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    
    np.random.shuffle(files)

    test_set_size = int(percent*len(files))
    test_set = files[:test_set_size]

    for f in test_set:
        cmd = "mv {0} {1}".format(f, to_dir)
        print (cmd)
        os.system(cmd)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", dest="dir", help = "Path to the directory that contains all the original photos", default="data")
ap.add_argument("-t", "--test-dir", dest="test_dir", help="Path to the directory where the testing images are stored", default="test_data")
args = ap.parse_args()


subs = ["onedollar", "twodollar", "tencents"]

for s in subs:
    dir = os.path.join(args.dir, s)
    test_dir = os.path.join(args.test_dir, s)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    random_seperate(dir, test_dir)