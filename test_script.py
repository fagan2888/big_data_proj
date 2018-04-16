from operator import add
import pyspark.sql
from pyspark.ml.feature import CountVectorizer, NGram
import numpy as np
import pandas as pd
import sys
from pyspark.sql import SparkSession


# 1. Blacklist
CASE_SENSITIVE_BLACKLIST = {
    "",
}
CASE_INSENSITIVE_BLACKLIST = {
    "blank",
    "null",
    "nan",
}
def filter_blacklist(i, x, cs_blacklist, ci_blacklist):
    if x in cs_blacklist:
        return i, x, True, "'{}' is in the blacklist".format(x)
    x = x.lower()
    if x in ci_blacklist:
        return i, x, True, "'{}' is in the blacklist".format(x)
    return i, x, False, None

# 2. Lengths
LOW_QUANTILE = 0.01
HIGH_QUANTILE = 0.99

# 3. 
N_GRAMS = [1, 2]
LOGPROB_QUANTILE_CUTOFF = 0.01

input_path = "/user/zp489/reddit_sarcasm_small.txt"


spark = SparkSession \
    .builder \
    .appName("mytask") \
    .getOrCreate()
sc = spark.sparkContext
raw_data = sc.textFile(input_path)
filtered_rdds = []

original_data = raw_data \
    .zipWithIndex() \
    .map(lambda _: (_[1], _[0]))
data = original_data

blacklist_inter1 = data \
    .map(lambda _: filter_blacklist(
        _[0], _[1],
        CASE_SENSITIVE_BLACKLIST,
        CASE_INSENSITIVE_BLACKLIST,
    ))
filtered_blacklist = blacklist_inter1 \
    .filter(lambda _: _[2]) \
    .map(lambda _: (_[0], _[3]))
filtered_rdds.append(filtered_blacklist)
data = blacklist_inter1 \
    .filter(lambda _: not _[2]) \
    .map(lambda _: (_[0], _[1]))



length_inter1 = data \
    .map(lambda _: (len(_[1]), _[0], _[1]))
length_counts = length_inter1 \
    .map(lambda _: (_[0], 1)) \
    .reduceByKey(add)

length_counts_srs = pd.Series(
    dict(length_counts.collect())
).sort_values()
low_q = length_counts_srs.quantile(LOW_QUANTILE)
high_q = length_counts_srs.quantile(HIGH_QUANTILE)
filtered_length = length_inter1 \
    .filter(lambda _: not(low_q < _[0] < high_q)) \
    .map(lambda _: (_[1], "Length {} not within ({}, {})".format(
        _[0], int(low_q), int(high_q))
    ))
import logging
logger = logging.getLogger('py4j')
print(filtered_length.count())



for ngram_n in N_GRAMS:
    sql_context = pyspark.sql.SQLContext(sc)
    data_df = data.map(lambda _: (_[0], _[1], list(_[1]))).toDF(["id", "text", "splittext"])
    ngram = NGram(n=2, inputCol="splittext", outputCol="ngrams")
    data_df_ngram = ngram.transform(data_df)
    flat_ngram = data_df_ngram \
        .select(["id", "ngrams", "text"]).rdd \
        .flatMap(lambda _: [(_[0], _1, _[2]) for _1 in _.ngrams])
    all_ngram_counts = flat_ngram \
        .map(lambda _: (_[1], 1)) \
        .reduceByKey(add)
    total_count = all_ngram_counts \
        .map(lambda _: _[1]) \
        .reduce(add)
    all_ngram_dist = all_ngram_counts \
        .map(lambda _: (_[0], _[1]/total_count))
    """
    x = flat_ngram \
        .map(lambda _: (_[1], _[0])) \
        .join(all_ngram_dist) \
        .map(lambda _: (_[1][0], np.log(_[1][1]))) \
        .groupByKey() \
        .map(lambda _: (_[0], sum(_[1]) / len(_[1])))
    """
    logprob = flat_ngram \
        .map(lambda _: (_[1], _[0])) \
        .join(all_ngram_dist) \
        .map(lambda _: (_[1][0], np.log(_[1][1]))) \
        .aggregateByKey(
                (0, 0),
                lambda a, b: (a[0] + b,    a[1] + 1),
                lambda a, b: (a[0] + b[0], a[1] + b[1])
            ) \
        .map(lambda _: (_[0], float(_[1][0] / _[1][1])))
    logprob_df = logprob.toDF(["id", "logprob"])
    logprob_cutoff = logprob_df.stat.approxQuantile("logprob", 
        [LOGPROB_QUANTILE_CUTOFF], 0.0001)[0]
    filtered_ngrams = logprob \
        .filter(lambda _: _[1] < logprob_cutoff) \
        .map(lambda _: (_[0], "{}-gram logprob smaller than {} quantile at {}".format(
            ngram_n, LOGPROB_QUANTILE_CUTOFF, _[1]
        )))
    print(filtered_ngrams.count())
    filtered_rdds.append(filtered_ngrams)
    data = logprob \
        .filter(lambda _: _[1] < logprob_cutoff) \
        .join(data) \
        .map(lambda _: (_[0], _[1][1]))

filtered_rdd = sc.union(filtered_rdds)
print(filtered_rdd.count())
outliers = filtered_rdd \
    .leftOuterJoin(original_data) \
    .map(lambda _: (_[0], _[1][1], _[1][0]))\
    .sortByKey()
print(outliers.count())
