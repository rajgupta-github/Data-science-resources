if not ('sc' in locals() or 'sc' in globals()):
    print(
        'It seems you are note running in a IBM Watson Studio Apache Spark Notebook. You might be running in a IBM Watson Studio Default Runtime or outside IBM Waston Studio. Therefore installing local Apache Spark environment for you. Please do not use in Production')

    from pip import main

    main(['install', 'pyspark==2.4.5'])

    from pyspark import SparkContext, SparkConf
    from pyspark.sql import SparkSession

    sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

    spark = SparkSession \
        .builder \
        .getOrCreate()

def assignment1(sc):
    rdd = sc.parallelize(list(range(100)))
    return rdd.count()

print(assignment1(sc))