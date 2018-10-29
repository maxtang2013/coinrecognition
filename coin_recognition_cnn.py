import numpy as np
import tensorflow as tf
import dataUtil
import pickle
from sklearn.cluster import KMeans
import argparse
import os

tf.logging.set_verbosity(tf.logging.INFO)

_MODEL_DIR = "model_size_64_single_coins/"
_IMAGE_HEIGHT = 64
_IMAGE_WIDTH = 64

_MPL_MODEL_DIR = os.path.join(_MODEL_DIR, "CNN/coin_recognition_cnn")
_TRAIN_SET_PATH = os.path.join(_MODEL_DIR, "data/train_dataset.pkl")
_TEST_SET_PATH = os.path.join(_MODEL_DIR, "test_data/test_dataset.pkl")

def learning_rate_with_decay(
    batch_size, batch_denom, num_images, boundary_epochs, decay_rates):
    """Get a learning rate that decays step-wise as training progresses.

    Args:
    batch_size: the number of examples processed in each training batch.
    batch_denom: this value will be used to scale the base learning rate.
        `0.1 * batch size` is divided by this number, such that when
        batch_denom == batch_size, the initial learning rate will be 0.1.
    num_images: total number of images that will be used for training.
    boundary_epochs: list of ints representing the epochs at which we
        decay the learning rate.
    decay_rates: list of floats representing the decay rates to be used
        for scaling the learning rate. Should be the same length as
        boundary_epochs.

    Returns:
    Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
    """
    initial_learning_rate = 0.1 * batch_size / batch_denom
    batches_per_epoch = num_images / batch_size

    # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(global_step):
        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step, boundaries, vals)

    return learning_rate_fn

def model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features, [-1, 64, 64, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1, outputs 32x32, 32 channels
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    
    # Pooling layer #2, outputs 16x16, 64 channels
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling layer #3, outputs 8x8, 128 channels
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool3_flat = tf.reshape(pool3, [-1, 8 * 8 * 64])
    dense = tf.layers.dense(inputs=pool3_flat, units=2048, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # dense1 = tf.layers.dense(inputs=dropout, units=512, activation=tf.nn.relu)
    # dropout1 = tf.layers.dropout(
    #     inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=3)

    classes = tf.argmax(input=logits, axis=1)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": classes,
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
    
    learning_rate_fn = learning_rate_with_decay(
        batch_size=128, batch_denom=128,
        num_images=381, boundary_epochs=[100, 150, 200],
        decay_rates=[1, 0.1, 0.01, 0.001])

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        # loss = tf.Print(loss, [global_step, learning_rate_fn(global_step)])
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_fn(global_step))
        train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step)
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
        train_images = train_images.astype(np.float32) / 255.0
        print (train_images[0])
    else:
        eval_images, eval_labels = load_cache_if_exist(_TEST_SET_PATH, parser.test_folder)
        eval_labels = func(eval_labels)
        print("evaluation data distribution:")
        print (np.bincount(eval_labels))
        eval_images = eval_images.astype(np.float32) / 255.0

    # Step 2. Train or evaluate the MPL classifier.
    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=_MPL_MODEL_DIR)
    
    # Set up logging for predictions
    tensors_to_log = {
        "probabilities": "softmax_tensor"
    }
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    
    if is_training:
        # Train the model
        print("Start to train the MLP ...")
        for i in range(100):
            classifier.train(
                input_fn=lambda: train_input_fn(train_images, train_labels, num_epochs=20),
                steps=None,
                hooks=[logging_hook])
            eval_results = classifier.evaluate(input_fn=lambda: train_input_fn(train_images, train_labels, num_epochs=1,shuffle=False, batch_size=1))
            print(eval_results)
    else:
        eval_results = classifier.evaluate(input_fn=lambda: train_input_fn(eval_images, eval_labels, num_epochs=1,shuffle=False, batch_size=1))
        print(eval_results)

    if is_training:
        print ("=====================Training Mode===================")
    else:
        print ("=====================Evaluation Mode===================")

if __name__ == "__main__":

    tf.app.run()

