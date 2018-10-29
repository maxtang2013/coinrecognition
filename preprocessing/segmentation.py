import os
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--resize", dest="resize", help = "Resize extracted coin images to 64x64 or not", default=0)
parser.add_argument("-d", "--dir", dest="dir", help = "Path to the directory that contains all the original photos", default="data")
parser.add_argument("-o", "--out-dir", dest="out_dir", help="Path to the directory where all generated coin images will be stored", required=True)
args = parser.parse_args()

src_dir = args.dir
should_resize = (args.resize == 1 or args.resize == "True" or args.resize == '1')

files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and (f.endswith(".jpg") or f.endswith(".jpeg"))]

idx = 0

for f in files:
    print("Processing {}".format(os.path.join(src_dir, f)))

    img = cv2.imread(os.path.join(src_dir, f))
    img = cv2.resize(img, (900, 1200))

    _,binary = cv2.threshold(img,45,255,cv2.THRESH_BINARY)
    gray = cv2.cvtColor(binary, cv2.COLOR_RGB2GRAY)
    _,binary = cv2.threshold(gray, 2,255,cv2.THRESH_BINARY)

    binary = cv2.medianBlur(binary, 7)

    kernel = np.ones((6,6))
    
    # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(binary,20,50)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 50,
            param1=15,
            param2=18,
            minRadius=30,
            maxRadius=80)
    
    for i in circles[0,:]:
        minx = int(i[0]-i[2])
        maxx = int(i[0]+i[2]+1)
        miny = int(i[1]-i[2])
        maxy = int(i[1]+i[2]+1)

        sub_img = img[miny:maxy, minx:maxx, :]
        coin_img = sub_img.copy()
        for x in range(minx,maxx):
            for y in range(miny, maxy):
                if (x - i[0])*(x-i[0]) + (y-i[1])*(y-i[1]) > i[2]*i[2]:
                    coin_img[y-miny][x-minx][:] = 0
        
        if should_resize:
            coin_img = cv2.resize(coin_img, (64,64))
        
        # write out
        idx = idx + 1
        cv2.imwrite("{0}/{1}.png".format(args.out_dir,idx), coin_img)