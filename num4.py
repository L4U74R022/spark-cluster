from pyspark.sql import SparkSession
from tensorflowonspark import TFCluster, TFNode
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", help="HDFS path to save the model", default="/path/to/save/model")
    return parser.parse_args()


def main_fun(args, ctx):
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with strategy.scope():
        # ConstrucciÃ³n del modelo usando Keras
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(10, activation='softmax')
        ])
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

    # Obtener el TFNode DataFeed
    tf_feed = TFNode.DataFeed(ctx.mgr, is_chief=(ctx.job_name == "chief"))

    # Cargar datos de MNIST utilizando Keras
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalizar los datos

    # Entrenamiento del modelo
    while not tf_feed.should_stop():
        model.fit(x_train, y_train, epochs=5, validation_data=(
            x_test, y_test), steps_per_epoch=10)

    if ctx.job_name == "chief":
        model.save(args.output)


if __name__ == '__main__':
    spark = SparkSession.builder.appName(
        "MNIST TensorFlowOnSpark").getOrCreate()
    args = parse_args()

    # Configurar TensorFlowOnSpark
    num_executors = 3
    num_ps = 1
    cluster = TFCluster.run(spark.sparkContext, main_fun, args, num_executors,
                            num_ps, input_mode=TFCluster.InputMode.TENSORFLOW, master_node='chief')
    cluster.shutdown()

    spark.stop()