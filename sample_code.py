from pyspark import SparkConf
from pyspark import SparkContext

conf = SparkConf()
#conf.setMaster('spark://<MASTER-HOSTNAME>:7077')
conf.setMaster('local[*]')
conf.setAppName('spark-basic')
sc = SparkContext(conf=conf)

def mod(x):
    import numpy as np
    return (x, np.mod(x, 2))

rdd = sc.parallelize(range(1000)).map(mod).take(10)
print(rdd)