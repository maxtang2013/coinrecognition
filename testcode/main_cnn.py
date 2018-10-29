import numpy as np
import tensorflow as tf
import dataUtil
import cv2
import pickle
from sklearn.cluster import KMeans
import argparse
import os
import coin_recognition_cnn as model

def predict_input_fn(images):
    return tf.data.Dataset.from_tensor_slices( (images, [0 for _ in range(images.shape[0])] ))


def predict_whole_images():

    classifier = tf.estimator.Estimator(
        model_fn=model.model_fn, model_dir=model._MPL_MODEL_DIR)
    

    files = ['20180416_224746.jpg']
    # files = ['20180416_225433.jpg', '20180416_225154.jpg', '20180416_223220.jpg', '20180416_225447.jpg', '20180416_225126.jpg', '20180416_225404.jpg', '20180416_225528.jpg']

    files = [f for f in os.listdir('image/multi') if os.path.isfile(os.path.join('image/multi', f)) and (f.endswith('.jpg') or f.endswith('.jpeg'))]

    for f in files:
        img = cv2.imread('image/multi/' + f)

        img = cv2.resize(img, (900, 1200))

        _,binary = cv2.threshold(img,8,255,cv2.THRESH_BINARY)
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
        idx = 0
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

            idx = idx + 1
            # cv2.imwrite("tmp/{}_ori.png".format(idx), coin_img)

            coin_img = cv2.resize(coin_img, (64,64))
            coin_img = cv2.cvtColor(coin_img, cv2.COLOR_RGB2GRAY)

            coin_img = coin_img.astype(np.float32)/255.0

            print ("going to predict")
            predictions = list(classifier.predict(input_fn=lambda:predict_input_fn(images=np.array([coin_img]))))
            print("preditions {}".format(predictions))
            predicted_classes = [p["classes"] for p in predictions]

            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (int(i[0]-15),int(i[1]-15))
            fontScale              = 1
            fontColor              = (255,0,0)
            lineType               = 2
            cv2.putText(img, '{}'.format(int(predictions[0]['classes'])), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

        for i in circles[0,:]:
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

        # cv2.imshow("edges", edges)
        # cv2.imshow("binary", binary)
        cv2.imshow("cirlces", img)
        cv2.waitKey()

def predict_good_quality_coins():

    classifier = tf.estimator.Estimator(
        model_fn=model.model_fn, model_dir=model._MPL_MODEL_DIR)

    files = ["onedollar/4.png", "onedollar/47.png", "tencents/2.png", "tencents/3.png",
        "tencents/1.png", "tencents/2.png", "tencents/3.png", "tencents/19.png",  "onedollar/7.png", "onedollar/12.png",
        "twodollar/1.png","twodollar/9.png","twodollar/13.png","twodollar/17.png",
        ]
    files = ["test3.png"]

    dir = 'model_size_64/test_data/'

    dir = 'tmp/'
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.endswith(".png") and not f.endswith("_ori.png")]

    for f in files:
        print (dir + f)
        img = cv2.imread(dir + f)
        ori_img = cv2.imread(dir + f.replace(".png", "_ori.png"))

        predictions = list(classifier.predict(input_fn=lambda:predict_input_fn(histograms=histogram)))

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (15, 35)
        fontScale              = 1
        fontColor              = (255,0,0)
        lineType               = 2
        cv2.putText(ori_img, '{}'.format(int(predictions[0]['classes'])), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

        cv2.imshow("cirlces", ori_img)
        cv2.waitKey()

# predict_good_quality_coins()
predict_whole_images()