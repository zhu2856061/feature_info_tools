#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: zhuzhipeng
# @Time: 2020/2/3|3:43 下午
# @Motto： Knowledge comes from decomposition

from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType


spark = SparkSession.builder.appName("gen_sample") \
    .enableHiveSupport() \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()

rdd = spark.sparkContext.textFile("hdfs://emr-cluster/user/dnn_1537324485/midu_dnn/20200202/22_23/*")

rdd = rdd.sample(False, 0.001)
rdd.repartition(1).saveAsTextFile('sample')




