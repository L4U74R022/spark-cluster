from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

def resize_image(file_storage):
    """
    Redimensiona una imagen dada como array de numpy a las dimensiones especificadas.
    """
    img = Image.open(file_storage.stream)
    img = img.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = 1.0 - img_array
    img_array = img_array.flatten()

    return Vectors.dense(img_array)


# Inicializar SparkSession
spark = SparkSession.builder \
    .appName("NumberRecognitionWithMNIST6") \
    .config("spark.executor.memory", "1g") \
    .config("spark.executor.cores", 8) \
    .config("spark.task.cpus", 1) \
    .getOrCreate()

# 1. Cargar el dataset MNIST usando tensorflow_datasets


def load_mnist_data():
    # Reducir el tamaño del conjunto de datos para pruebas
    mnist_data = tfds.load('mnist', split='train[:35%]', as_supervised=True)
    data = []
    for image, label in tfds.as_numpy(mnist_data):
        features = image.flatten() / 255.0  # Normalizar los valores de los píxeles
        data.append((list(features), int(label)))
    return data


model_path = "/home/labsis28/spark-cluster/modelito"
if __name__ == "__main__":
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
    ).repartition(40)

    # 3.5. En cada nodo guardamos el dataframe en cache
    data_df.cache()

    # 4. Crear y entrenar un modelo de ejemplo (mejorar la configuración del modelo)
    # Reducir maxIter para pruebas más rápidas
    lr = LogisticRegression(featuresCol='features', labelCol='label', regParam=0.001, elasticNetParam=0.5)

    # Dividir datos en entrenamiento y prueba
    train, test = data_df.randomSplit([0.7, 0.3], seed=12345)

    # Entrenar modelo
    model = lr.fit(train)

    # Guardamos el modelo para ser usado por otros servicios
    if spark.sparkContext.getConf().get("spark.executor.id") == "driver":
        model.write().overwrite().save(model_path)
        print("MODELO GUARDADO")

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

def predict(image):
    image = resize_image(image)
    model = LogisticRegressionModel.load(model_path)
    test_data = spark.createDataFrame([(image,)], ["features"])
    prediction = model.transform(test_data)
    return int(prediction.collect()[0]['prediction'])
