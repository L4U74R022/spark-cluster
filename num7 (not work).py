from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import tensorflow_datasets as tfds

# Inicializar SparkSession
spark = SparkSession.builder \
    .appName("NumberRecognitionWithMNIST") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.executor.memory", "12g") \
    .getOrCreate()

# Cargar el dataset MNIST usando tensorflow_datasets


def load_mnist_data():
    mnist_data = tfds.load('mnist', split='train[:10%]', as_supervised=True)
    data = []
    for image, label in tfds.as_numpy(mnist_data):
        features = image.flatten() / 255.0  # Normalizar los valores de los píxeles
        data.append((Vectors.dense(features), int(label)))
    return data


# Crear un esquema para el DataFrame
schema = StructType([
    StructField("features", FloatType(), True),
    StructField("label", IntegerType(), True)
])

# Crear un DataFrame con los datos del MNIST
mnist_data = load_mnist_data()
data_df = spark.createDataFrame(mnist_data, ["features", "label"])

# Configurar y entrenar el modelo de regresión logística
lr = LogisticRegression(maxIter=10, regParam=0.01, elasticNetParam=0.5)

# Dividir datos en entrenamiento y prueba
train, test = data_df.randomSplit([0.8, 0.2], seed=12345)

# Entrenar el modelo
model = lr.fit(train)

# Guardar el modelo entrenado
model_path = "./modelito"
model.write().overwrite().save(model_path)

# Cargar el modelo
model_loaded = LogisticRegressionModel.load(model_path)

# Hacer predicciones con el modelo cargado
predictions = model_loaded.transform(test)

# Mostrar las predicciones
predictions.select("features", "label", "prediction").show(10)

# Evaluar el modelo
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Cerrar la sesión de Spark
spark.stop()
