# import findspark
# findspark.init("/opt/spark-2")
import random
from pyspark import SparkContext

sc = SparkContext(appName="EstimatePi")

def inside(p):
    """function inside.
    Doc::
            
            Args:
                p:   
            Returns:
                
    """
    x, y = random.random(), random.random()
    return x*x + y*y < 1

NUM_SAMPLES = 1000000

count = sc.parallelize(range(0, NUM_SAMPLES)) \
             .filter(inside).count()

print("Pi is roughly %f" % (4.0 * count / NUM_SAMPLES))

sc.stop()
