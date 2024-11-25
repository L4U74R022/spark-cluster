from pyspark import SparkConf
from pyspark.sql import SparkSession
from tensorflowonspark import TFCluster
import tensorflow as tf
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", help="HDFS path to save the model", default="/path/to/save/model")
    return parser.parse_args()


def mnist_model(args, ctx):
    from tensorflowonspark import TFNode
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with strategy.scope():
        # Define a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')

    # Dummy data for testing
    x_train = np.array([[i] for i in range(100)], dtype=np.float32)
    y_train = np.array([[2 * i] for i in range(100)], dtype=np.float32)

    # Train the model
    model.fit(x_train, y_train, epochs=5, steps_per_epoch=5)

    # Save the model only if this is the chief worker
    if ctx.job_name == "chief":
        model.save(args.output)

    return model.history


if __name__ == '__main__':
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("tensorflow_spark_test") \
        .master("spark://192.168.1.28:7077") \
        .config(conf=SparkConf()) \
        .config("spark.executor.instances", "3") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()

    args = parse_args()

    # Test simple Spark operation
    rdd = spark.sparkContext.parallelize(range(10))
    print("Count of RDD:", rdd.count())

    # Set up TensorFlowOnSpark
    num_ps = 1  # Number of parameter servers
    cluster = TFCluster.run(spark.sparkContext, mnist_model, args, num_executors=3,
                            num_ps=num_ps, tensorboard=False, input_mode=TFCluster.InputMode.TENSORFLOW)
    cluster.train(rdd, num_epochs=5)
    cluster.shutdown()

    spark.stop()
