from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import tensorflow_datasets as tfds

# Inicializar SparkSession
spark = SparkSession.builder \
    .appName("NumberRecognitionWithMNIST8") \
    .config("spark.executor.memory", "1g") \
    .config("spark.executor.cores", 8) \
    .config("spark.task.cpus", 1) \
    .getOrCreate()

# 1. Cargar el dataset MNIST usando tensorflow_datasets


def load_mnist_data():
    # Reducir el tamaño del conjunto de datos para pruebas
    mnist_data = tfds.load('mnist', split='train[:15%]', as_supervised=True)
    data = []
    for image, label in tfds.as_numpy(mnist_data):
        features = image.flatten() / 255.0  # Normalizar los valores de los píxeles
        data.append((list(features), int(label)))
    return data


# 2. Crear un esquema para el DataFrame
schema = StructType([
    StructField("features", FloatType(), True),
    StructField("label", IntegerType(), True)
])

# 3. Crear un DataFrame con los datos del MNIST
mnist_data = load_mnist_data()
data_df = spark.createDataFrame(
    [(Vectors.dense(row[0]), row[1]) for row in mnist_data],
    schema=["features", "label"]
    ).repartition(5*8*2).rdd.map(lambda row: LabeledPoint(row['label'], row['features']))

# 4. Crear y entrenar un modelo de ejemplo (mejorar la configuración del modelo)
# Reducir maxIter para pruebas más rápidas
lr = LogisticRegressionWithLBFGS()

# Dividir datos en entrenamiento y prueba
train, test = data_df.randomSplit([0.8, 0.2], seed=12345)

# Entrenar modelo
model = lr.train(train, iterations=25, regParam=0.01)

# Guardamos el modelo para ser usado por otros servicios
# if spark.sparkContext.getConf().get("spark.executor.id") == "driver":
#     model.write().overwrite().save("/home/labsis28/spark-cluster/modelito")
#     print("MODELO GUARDADO")

# 5. Realizar predicciones
predictions = model.transform(test)

# 6. Mostrar las predicciones
predictions.select("features", "label", "prediction").show(10)

# 7. Evaluar el modelo
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Cerrar la sesión de Spark
spark.stop()