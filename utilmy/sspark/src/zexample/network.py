from __future__ import print_function
import os
import graphframes
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T
import re
from pyspark.sql.window import Window
 
 
def graph_coalesce(g, numPartitions):
    return graphframes.GraphFrame(
        g.vertices.coalesce(numPartitions),
        g.edges.coalesce(numPartitions)
    )
 
def load_graphframe(sqlContext, dir_name, numPartitions=None):
    fn_vertices = os.path.join(dir_name, 'vertices.parquet')
    fn_edges = os.path.join(dir_name, 'edges.parquet')
    vertices = sqlContext.read.parquet(fn_vertices)
    edges = sqlContext.read.parquet(fn_edges)
    ret = graphframes.GraphFrame(vertices, edges)
    if numPartitions is not None:
        ret = graph_coalesce(ret, numPartitions)
    return ret
 
def getOrCreateSparkContext(conf=None):
    # Based on
    # <a href="https://href.li/?http://www.eecs.berkeley.edu/~jegonzal/pyspark/_modules/pyspark/context.html" rel="nofollow noreferrer">http://www.eecs.berkeley.edu/~jegonzal/pyspark/_modules/pyspark/context.html</a>
    # pyspark version that we currently use (1.5) doesn't provide this method.
    # Thus, implementing it here.
    # Note: If we use `with pyspark.SparkContext._lock:`, as in the linked code,
    # the program freezes infinitely. Right now, we don't create threads within the
    # main script. Thus, this code seems to be pretty safe. In the future, we will
    # have to deal with the locking issue
 
    if pyspark.SparkContext._active_spark_context is None:
        pyspark.SparkContext(conf=conf or pyspark.SparkConf())
    return pyspark.SparkContext._active_spark_context
 
 
 
sc = getOrCreateSparkContext()
sqlContext = pyspark.HiveContext(sc)
path_input_graph = '/user/anandnalya/network'
grph = load_graphframe(sqlContext, path_input_graph, 128)
vertices = grph.vertices.withColumn('cost', F.pow(F.col('pagerank'), -1.0))
edges = grph.edges.withColumn('cost', F.pow(F.col('count'), -1.0))
grph = graphframes.GraphFrame(vertices, edges)
 
path_search_query = '''(u0)-[likes_post00]->(p0);
        (a0)-[is_author0]->(p0);
        (u1)-[likes_post10]->(p0);
        (u1)-[likes_post11]->(p1);
        (a1)-[is_author1]->(p1)
'''
path_search_filter_statement  = """u0.id IN ('1,2,3,...') AND
            is_author0.edge_type = 'IS_AUTHOR' AND
            is_author1.edge_type = 'IS_AUTHOR' AND
            likes_post00.edge_type = 'LIKES_POST' AND
            likes_post10.edge_type = 'LIKES_POST' AND
            likes_post11.edge_type = 'LIKES_POST' AND
 
            a0.node_type = 'USER' AND
            a1.node_type = 'USER' AND
            u0.node_type = 'USER' AND
            u1.node_type = 'USER' AND
            p0.node_type = 'POST' AND
            p1.node_type = 'POST' AND
 
            a0.id != u0.id AND
            a0.id != u1.id AND
            a1.id != u0.id AND
            a1.id != u1.id AND
            a0.id != a1.id AND
            p0.id != p1.id AND
            a0.id != 'USER__' AND
            a1.id != 'USER__'"""
path_search = grph.find(
    path_search_query
).filter(
    path_search_filter_statement
)
 
 