from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy as np

# Inicializar SparkSession
spark = SparkSession.builder \
    .appName("NumberRecognition") \
    .getOrCreate()

# 1. Generar datos de ejemplo (simulando imágenes aplanadas de 28x28)
def generate_dummy_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        features = np.random.rand(28 * 28)  # Simula datos de imágenes
        label = np.random.randint(0, 10)  # Etiqueta entre 0 y 9
        data.append((list(features), label))
    return data

# Crear esquema para DataFrame
schema = StructType([
    StructField("features", StructType([
        StructField("values", FloatType(), True)
    ]), True),
    StructField("label", IntegerType(), True)
])

# 2. Crear un DataFrame con los datos simulados
dummy_data = generate_dummy_data()
data_df = spark.createDataFrame(
    [(Vectors.dense(row[0]), row[1]) for row in dummy_data],
    schema=["features", "label"]
)

# 3. Crear y entrenar un modelo de ejemplo
lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Dividir datos en entrenamiento y prueba
train, test = data_df.randomSplit([0.8, 0.2], seed=12345)

# Entrenar modelo
model = lr.fit(train)

# 4. Realizar predicciones
predictions = model.transform(test)

# 5. Mostrar las predicciones
predictions.select("features", "label", "prediction").show(10)

# 6. Evaluar el modelo
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Cerrar la sesión de Spark
spark.stop()
