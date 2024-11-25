from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from tensorflowonspark import TFCluster
import tensorflow as tf
import argparse
from datetime import datetime
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", help="HDFS path to save the model", default="/path/to/save/model")
    parser.add_argument(
        "--num_epochs", help="Number of epochs", type=int, default=10)
    return parser.parse_args()


def parse_mnist_data(iterator):
    for x in iterator:
        image, label = x
        image = tf.reshape(tf.cast(image, tf.float32),
                           shape=[-1, 28, 28]) / 255.0
        label = tf.one_hot(label, 10)
        yield (image, label)


def main_fun(args, ctx):
    from tensorflowonspark import TFNode

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])

    tf_feed = TFNode.DataFeed(ctx.mgr)

    while not tf_feed.should_stop():
        batch = tf_feed.next_batch(1)
        if len(batch) > 0:
            x, y = zip(*batch)
            # Use args.num_epochs to use the command line argument
            model.fit(x, y, epochs=1)
        else:
            break

    # Save the model only if this is the chief worker
    if ctx.job_name == "chief":
        model.save(args.output)

    tf_feed.terminate()


if __name__ == '__main__':
    sc = SparkContext(conf=SparkConf().setAppName("tensorflow_spark_mnist").setMaster("spark://192.168.1.28:7077"))
    args = parse_args()
    executors = int(sc.getConf().get("spark.executor.instances", "1"))
    num_ps = 1
    tensorboard = False

    # Load MNIST using TensorFlow
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    # Create a Spark RDD for MNIST
    mnist_rdd = sc.parallelize(
        list(zip(x_train, y_train))).mapPartitions(parse_mnist_data)

    num_ps = 1
    executors = 3

    cluster = TFCluster.run(sc, main_fun, args, num_executors=executors, num_ps=num_ps,
                            tensorboard=tensorboard, input_mode=TFCluster.InputMode.TENSORFLOW)
    cluster.train(mnist_rdd, num_epochs=args.num_epochs)
    cluster.shutdown()

    # Close the Spark context
    sc.stop()
