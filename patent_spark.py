from collections import Counter
import json
#import nltk
#from nltk.corpus import stopwords
from pyspark import SparkConf, SparkContext
import re

conf = SparkConf().setMaster('local[1]').setAppName('logs')
sc = SparkContext(conf=conf)

test = sc.textFile('02444358.nlp')

lines = test.map(lambda x: str(x.split('\t')[1]) if len(x.split('\t')) > 1 else '') 

lines.saveAsTextFile("output.txt")
