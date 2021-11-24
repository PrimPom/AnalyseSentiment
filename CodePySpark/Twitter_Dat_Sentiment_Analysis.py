# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:06:18 2021

@author: user
"""
import re


from pyspark.ml.feature import StopWordsRemover
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel

def f(line):
    review = re.sub('[^a-zA-Z]', ' ', line[0])    
    review = review.lower()
    review = review.split()    
    return [review, line[1]]


def g(line):
    label = line[0][1]
    return LabeledPoint(label,line[1])



if __name__=="__main__":
    import findspark

    findspark.init()
    
    import pyspark
    
    from pyspark.sql import *
    
    spark = SparkSession.builder.getOrCreate()
    
    """
    spark = SparkSession.builder\
            .master("local[*]")\
            .appName("Ilyas")\
            .getOrCreate()
    """
    sc = spark.sparkContext 
    
    from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType
    
    schema = StructType([ \
        StructField("target", IntegerType(), True), \
        StructField("ID", IntegerType(), True), \
        StructField("date ", StringType(), True), \
        StructField("query", StringType(), True), \
        StructField("user", StringType(), True), \
        StructField("text", StringType(), True)])
    
    trainfile = "file:/D:/ilyas/Apprentissage automatique pour le Big Data/Projet/training.1600000.processed.noemoticon.csv"    
    testfile = "file:/D:/ilyas/Apprentissage automatique pour le Big Data/Projet/testdata.manual.2009.06.14.csv"
    print("\n\nStart importing data\n\n")
        
    data = spark.read.format('csv').load(trainfile, schema = schema)
    data = data.na.drop()
    newSchema = ["text","target"]
    
    print("Start cleaning data")
    df = data.select("text","target").rdd.map(f).toDF(schema = ["text","target"])
    
    # Define a list of stop words or use default list
    remover = StopWordsRemover()
    remover.setInputCol("text")
    remover.setOutputCol("text_no_stopw")
    
    # Transform existing dataframe with the StopWordsRemover
    df = remover.transform(df).select("text_no_stopw","target")
    
    from pyspark.mllib.feature import HashingTF, IDF
    
    print("Start HashingTF")
    hashingTf = HashingTF()
    tf = hashingTf.transform(df.rdd.map(lambda line : [' '.join(line[0]),line[1]]))
    tf.cache()
    
    print("Start IDF")   
    idf = IDF().fit(tf)
    tfidf = idf.transform(tf)
    tfidf.cache()
    
    print("Start ZIP")
    #schema = ['ancien',"features"]
    xformedData = df.rdd.zip(tfidf).toDF(schema = ['ancien',"features"])
    xformedData.cache()
    
    
    print("Start mapping")
    newData = xformedData.rdd.map(g)
    #.toDF(schema = ["ft","target"])
        
    train, test = newData.randomSplit([0.9, 0.1], seed=2000)
    
    print("Start Naive Bayes")
    
    nbModel = NaiveBayes.train(train)
    
    predictionAndLabel = test.map(lambda p : (nbModel.predict(p.features), p.label))
    
    accuracy = predictionAndLabel.filter(lambda x : x[0] == x[1]).count() / test.count()
    
    print(accuracy)
    