import numpy as np
import tensorflow as tf
import dataUtil
import cv2
import pickle
from sklearn.cluster import KMeans
import argparse
import os

tf.logging.set_verbosity(tf.logging.INFO)

_NUM_FEATURE_CLUSTER = 128

_MODEL_DIR = "model_size_64_single_coins/"
_IMAGE_HEIGHT = 64
_IMAGE_WIDTH = 64

_FEATURE_LIST_FILE = os.path.join(_MODEL_DIR,"SIFT_features.pkl")
_MPL_MODEL_DIR = os.path.join(_MODEL_DIR, "MPL/coin_recognition_SIFT")
_KMEANS_RESULT_FILE=os.path.join(_MODEL_DIR, "Kmeans.pkl")
_TRAIN_SET_PATH = os.path.join(_MODEL_DIR, "data/train_dataset.pkl")
_TEST_SET_PATH = os.path.join(_MODEL_DIR, "test_data/test_dataset.pkl")

def model_fn(features, labels, mode):
    """
    Model function for the MPL classifier
    Just a simple MPL with 2 hidden layers.
    """
    # Input Layer

    input_layer = tf.reshape(features, [-1, _NUM_FEATURE_CLUSTER])

    dense = tf.layers.dense(inputs=input_layer, units=256, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense = tf.layers.dense(inputs=input_layer, units=128, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=3)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.Print(loss, ["classes", labels, tf.argmax(input=logits, axis=1)])

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def train_input_fn(train_data, train_labels, num_epochs, batch_size=128, shuffle=True):
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    if shuffle:
        train_dataset = train_dataset.shuffle(512)
    train_dataset = train_dataset.repeat(num_epochs).batch(batch_size)
    return train_dataset

class BagOfWords:

    def build_model(self, train_images, train_labels):

        # extract SIFT features for all images
        if os.path.exists(_FEATURE_LIST_FILE):
            with open(_FEATURE_LIST_FILE, "rb") as f:
                allSIFTFeatures = pickle.load(f)
        else:
            allSIFTFeatures = self.extractSIFTFeatrues(train_images)
            with open(_FEATURE_LIST_FILE, "wb") as f:
                pickle.dump(allSIFTFeatures, f)
        
        flatSIFTFeatures = []
        for descriptor in allSIFTFeatures:
            flatSIFTFeatures = np.append(flatSIFTFeatures, descriptor)
        flatSIFTFeatures = flatSIFTFeatures.reshape([-1, 128])

        print("Total number of images: {}".format(allSIFTFeatures.shape[0]))
        print ("Total number of SIFT features: {}".format(flatSIFTFeatures.shape[0]))

        # clusterring SIFT descriptors into _NUM_FEATURE_CLUSTER clusters.
        if not os.path.exists(_KMEANS_RESULT_FILE):
            kmeans = KMeans(n_clusters=_NUM_FEATURE_CLUSTER)
            res = kmeans.fit(flatSIFTFeatures)
            self.kmeans = kmeans
            self.centroids = res.cluster_centers_
            self.train_labels = train_labels
            with open(_KMEANS_RESULT_FILE, "wb") as f:
                pickle.dump((self.kmeans, self.centroids, self.train_labels), f)
        else:
            with open(_KMEANS_RESULT_FILE, "rb") as f:
                self.kmeans, self.centroids, self.train_labels = pickle.load(f)

        # create histograms for each training image
        histograms = []
        for descriptors in allSIFTFeatures:
            feature_vector = self._construct_histgram_from_descriptors(descriptors)
            histograms.append(feature_vector)
        self.train_features = np.array(histograms)

    def save_model(self, to_file):
        with open(to_file, 'wb') as f:
            pickle.dump((self.kmeans, self.centroids, self.train_features, self.train_labels), f)

    def load_model(self, from_file):
        with open(from_file, 'rb') as f:
            self.kmeans, self.centroids, self.train_features, self.train_labels = pickle.load(f)

    def extractSIFTFeatrues(self, images):
        sift = cv2.xfeatures2d.SIFT_create()
        features = []
        for image in images:
            image = image.astype(np.uint8)
            keypoints,des = sift.detectAndCompute(image, None)
            if len(keypoints) > 0:
                features.append(des)
            else:
                features.append(np.array([]))
        return np.array(features)

    def _normalize(self, input_data):
        sum_input = np.sum(input_data)

        if sum_input > 0:
            return input_data / sum_input
        else:
            return input_data

    def _construct_histgram_from_descriptors(self, descriptors):
        if (descriptors.shape[0] == 0):
            feature_vector = np.zeros(_NUM_FEATURE_CLUSTER)
            return np.reshape(feature_vector, ((1, feature_vector.shape[0])))

        labels = self.kmeans.predict(descriptors)
        feature_vector = np.zeros(_NUM_FEATURE_CLUSTER)

        for i, item in enumerate(descriptors):
            feature_vector[labels[i]] += 1

        feature_vector_img = np.reshape(feature_vector, 
                ((1, feature_vector.shape[0])))
        return self._normalize(feature_vector_img)

    def construct_feature_map(self, images):
        """
        Construct histogram of bag of words for a list images.
        """
        descriptors_list = self.extractSIFTFeatrues(images)
        features = []
        for descriptors in descriptors_list:
            feature_vector = self._construct_histgram_from_descriptors(descriptors)
            features.append(feature_vector)
        return np.array(features)

def labelToInt(label):
    return {
        "tencents": 0,
        "onedollar": 1,
        "twodollar": 2
    }[label]

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Extract features from a given \
            set of images')

    parser.add_argument("--data-folder", dest="train_folder",
            help="Folder containing the training images organized in subfolders", default=os.path.join(_MODEL_DIR, "data"))

    parser.add_argument("--test-folder", dest="test_folder",
            help="Folder containing the testing images organized in subfolders", default=os.path.join(_MODEL_DIR,"test_data"))

    parser.add_argument("--model-file", dest='model_file',
            help="Output file where the model will be stored", default=os.path.join(_MODEL_DIR, "SIFT_bagofwords.pkl"))

    parser.add_argument("--evaluate", dest="evaluate", default=False)

    args = parser.parse_args()

    return args

def load_cache_if_exist(filepath, original_dir):
    if not os.path.exists(filepath):
        train_images, train_labels = dataUtil.readDataset(original_dir, _IMAGE_HEIGHT, _IMAGE_WIDTH)
        with open(filepath, 'wb') as f:
            pickle.dump((train_images, train_labels), f)
    else:
        with open(filepath, 'rb') as f:
            train_images, train_labels = pickle.load(f)
    return train_images, train_labels



def main(unused_argv):

    parser = build_arg_parser()

    bagOfWords = BagOfWords()

    func = np.vectorize(labelToInt)

    is_training = (parser.evaluate == False or parser.evaluate == '0' or parser.evaluate == "False")

    if is_training:
        print ("=====================Training Mode===================")
    else:
        print ("=====================Evaluation Mode===================")

    # Step 1. Build or load the BagOfWords Model.
    if is_training:
        
        train_images, train_labels = load_cache_if_exist(_TRAIN_SET_PATH, parser.train_folder)
        print (train_images.shape)
        train_labels = func(train_labels)
        print (np.bincount(train_labels))

        if os.path.exists(parser.model_file):
            bagOfWords.load_model(parser.model_file)
        else:
            bagOfWords.build_model(train_images, train_labels)
            bagOfWords.save_model(parser.model_file)
        
        train_data = bagOfWords.train_features
    else:
        bagOfWords.load_model(parser.model_file)

    eval_images, eval_labels = load_cache_if_exist(_TEST_SET_PATH, parser.test_folder)
    eval_labels = func(eval_labels)
    print("evaluation data distribution:")
    print (np.bincount(eval_labels))
    eval_data = bagOfWords.construct_feature_map(eval_images)

    # Step 2. Train or evaluate the MPL classifier.
    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=_MPL_MODEL_DIR)
    
    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    
    if is_training:
        # Train the model
        print("Start to train the MLP...")

        classifier.train(
            input_fn=lambda: train_input_fn(train_data, train_labels, num_epochs=40000),
            steps=None)
            # hooks=[logging_hook])

        print("Calculating training accuracy ...")
        eval_results = classifier.evaluate(input_fn=lambda: train_input_fn(train_data, train_labels, num_epochs=1,shuffle=False, batch_size=1))
        print(eval_results)
        
        print("Evaluating ...")
        eval_results = classifier.evaluate(input_fn=lambda: train_input_fn(eval_data, eval_labels, num_epochs=1,shuffle=False, batch_size=1))
        print(eval_results)
    else:
        eval_results = classifier.evaluate(input_fn=lambda: train_input_fn(eval_data, eval_labels, num_epochs=1,shuffle=False, batch_size=1))
        print(eval_results)

    if is_training:
        print ("=====================Training Mode===================")
    else:
        print ("=====================Evaluation Mode===================")

if __name__ == "__main__":

    tf.app.run()

