import os
import subprocess
from pyspark import SparkConf
from pyspark.sql import SparkSession

def get_spark_session(**kwargs):
    """Initializing Spark context

    Parameters
    ----------
    kwargs : dict
        Variable number of keyword arguments to initialize the SparkContext
        object

    Returns
    -------
     : SparkContext object

    """
    os.environ['PYSPARK_SUBMIT_ARGS'] = \
        '--conf spark.driver.memory=16g pyspark-shell'

    conf = SparkConf() \
        .setAppName(kwargs.get('app_name', 'anonymous')) \
        .set("spark.executor.memory", kwargs.get('executor_memory', '10g')) \
        .set("spark.driver.maxResultSize",
             kwargs.get('max_result_size', '25g')) \
        .set("spark.executor.instances", kwargs.get('num_executors', '20')) \
        .set("spark.executor.cores", kwargs.get('num_cores', '3')) \
        .set("spark.memory.fraction", kwargs.get('spark_mem_frac', '0.2')) \
        .set("spark.sql.crossJoin.enabled", True)
    
    spark_session = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
    return spark_session


def write_csv(df, local_filename, username):
    """ Write a dataframe into a CSV file
        # df: the dataframe
        # local_filename: Name of the file you want to save
        # username: username for Spark (not local username)
    """
    
    # On hadoop cluster, you can only access your own directory
    HDFS_FILENAME = 'hdfs://10.65.12.3:9000/user/{0}/{1}'.format(username, os.path.basename(local_filename)) # specify a fil    e on hadoop
  
    df.write.format('csv').option("sep" , "\t").option("header", "false").mode('overwrite').save(HDFS_FILENAME)
  
    # Merge tmp files
    subprocess.call('hadoop fs -getmerge %s %s' % (HDFS_FILENAME, local_filename + "_tmp"), shell=True)
    subprocess.call('hadoop fs -rm -r %s' % HDFS_FILENAME, shell=True)
  
    dbinputfile = open(local_filename + "_tmp", 'rb')
    dboutputfile = open(local_filename, 'wb')
  
    # Specify a header if needed
    header = ",".join(df.columns)
    dboutputfile.write(header + "\n")
    for line in dbinputfile:
        dboutputfile.write(line)
    dboutputfile.close()
  
    # Remove tmp files on local computer 
    os.remove(local_filename + "_tmp")
    os.remove(os.path.dirname(local_filename) + '/.' + os.path.basename(local_filename) + "_tmp.crc")
