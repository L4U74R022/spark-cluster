from sparkflow.graph_utils import build_graph
from sparkflow.tensorflow_async import SparkAsyncDL
import tensorflow.compat.v1 as tf
from pyspark.ml.feature import VectorAssembler, OneHotEncoder
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import SparkSession
import numpy as np
import pandas as pd

tf.disable_v2_behavior()

# Crear la SparkSession
spark = SparkSession.builder \
    .appName("Spark TensorFlow Example") \
    .getOrCreate()

# Simple tensorflow network
@tf.function
def small_model():
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y = tf.placeholder(tf.float32, shape=[None, 10], name='y')
    layer1 = tf.keras.layers.Dense(x, 256, activation=tf.nn.relu)
    layer2 = tf.keras.layers.Dense(layer1, 256, activation=tf.nn.relu)
    out = tf.keras.layers.Dense(layer2, 10)
    z = tf.argmax(out, 1, name='out')
    loss = tf.losses.softmax_cross_entropy(y, out)
    return loss

# Cargar el dataset de MNIST desde TensorFlow
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

# Preprocesar los datos: aplanar las imágenes y normalizar
x_train = x_train.reshape(-1, 784) / 255.0

# Convertir etiquetas a one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)

# Crear un DataFrame de Pandas y convertirlo a Spark DataFrame
df_pandas = pd.DataFrame(np.hstack((y_train, x_train)))
spark_df = spark.createDataFrame(df_pandas)

# Especificar las columnas
feature_cols = [str(i) for i in range(784)]
label_cols = [str(i) for i in range(784, 794)]

# Convertir el grafo de TensorFlow a JSON
tensorflow_graph = build_graph(small_model)

# VectorAssembler para combinar columnas de características
va = VectorAssembler(inputCols=feature_cols, outputCol='features')

# OneHotEncoder para las etiquetas
encoded = OneHotEncoder(inputCol='labels', outputCol='encoded_labels', dropLast=False)

# Crear el modelo de SparkAsyncDL
spark_model = SparkAsyncDL(
    inputCol='features',
    tensorflowGraph=tensorflow_graph,
    tfInput='x:0',
    tfLabel='y:0',
    tfOutput='out:0',
    tfLearningRate=.001,
    iters=20
)

# Construir y ajustar el pipeline
p = Pipeline(stages=[va, encoded, spark_model]).fit(spark_df)

