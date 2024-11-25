from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, OneHotEncoder
from pyspark.ml.pipeline import Pipeline
from sparkflow.graph_utils import build_graph
from sparkflow.tensorflow_async import SparkAsyncDL
import tensorflow as tf
import numpy as np
import pandas as pd

# Crear la SparkSession
spark = SparkSession.builder.appName("Distributed MNIST with SparkFlow").getOrCreate()

# Definir el modelo de TensorFlow
def create_model():
    x = tf.keras.Input(dtype=tf.float32, shape=(None, 784), name='x')  # Entrada
    y = tf.keras.Input(dtype=tf.float32, shape=(None, 10), name='y')  # Etiquetas

    # Modelo secuencial equivalente
    layer1 = tf.layers.dense(x, 128, activation=tf.nn.relu)
    dropout = tf.layers.dropout(layer1, rate=0.5)
    logits = tf.layers.dense(dropout, 10)
    output = tf.nn.softmax(logits, name='output')  # Salida

    # Función de pérdida y optimizador
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

    return loss

# Cargar y procesar datos usando Spark
# Dataset MNIST cargado manualmente
mnist_file = tf.keras.utils.get_file(
    fname="mnist.npz",
    origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
)

with np.load(mnist_file) as data:
    x_train = data['x_train']
    y_train = data['y_train']

# Preprocesar: aplanar y normalizar las imágenes
x_train = x_train.reshape(-1, 784) / 255.0
y_train_one_hot = np.eye(10)[y_train]  # Convertir etiquetas a one-hot encoding

# Crear un DataFrame de Pandas y luego convertirlo a Spark DataFrame
df_pandas = pd.DataFrame(np.hstack((y_train_one_hot, x_train)))
spark_df = spark.createDataFrame(df_pandas)

# Configurar pipeline para SparkFlow
feature_cols = [str(i) for i in range(784)]  # Columnas de características
label_cols = [str(i) for i in range(784, 794)]  # Columnas de etiquetas (one-hot)

# VectorAssembler para características
va = VectorAssembler(inputCols=feature_cols, outputCol='features')

# Convertir el grafo de TensorFlow a JSON para SparkFlow
tensorflow_graph = build_graph(create_model)

# Configurar SparkAsyncDL para entrenamiento distribuido
spark_model = SparkAsyncDL(
    inputCol='features',
    tensorflowGraph=tensorflow_graph,
    tfInput='x:0',
    tfLabel='y:0',
    tfOutput='output:0',
    tfLearningRate=0.0001,
    iters=100,
    verbose=1
)

# Crear y ajustar el pipeline
pipeline = Pipeline(stages=[va, spark_model])
pipeline_model = pipeline.fit(spark_df)

# Predicciones
predictions = pipeline_model.transform(spark_df)
predictions.select("features", "prediction").show(5)

# Detener SparkSession
spark.stop()

