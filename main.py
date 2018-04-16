import argparse
import json
import numpy as np
import pandas as pd
import sys

from pyspark.sql import SparkSession


from filters import blacklist, length, ngram_dist


def main(input_hfs_path, output_hfs_path, config):
    spark = SparkSession \
        .builder \
        .appName("TextOutlier") \
        .getOrCreate()
    sc = spark.sparkContext

    raw_data = sc.textFile(input_hfs_path)
    print("Read from {}".format(input_hfs_path))
    filtered_rdds = []

    original_data = raw_data \
        .zipWithIndex() \
        .map(lambda _: (_[1], _[0]))
    data = original_data

    # 1. Blacklist
    data, filtered_data = blacklist.filter_blacklist(
        data=data,
        cs_blacklist=config["blacklist_case_sensitive"],
        ci_blacklist=config["blacklist_case_insensitive"],
    )
    filtered_rdds.append(filtered_data)

    # 2. Lengths
    data, filtered_data = length.filter_length(
        data=data,
        low_quantile=config["length_low_quantile"],
        high_quantile=config["length_high_quantile"],
    )
    filtered_rdds.append(filtered_data)

    # 3. N-gram Distribution
    for ngram_n in config["ngram_list"]:
        data, filtered_data = ngram_dist.filter_ngrams(
            data=data,
            ngram_n=ngram_n,
            logprob_quantile_cutoff=config["ngram_logprob_quantile_cutoff"],
        )
        filtered_rdds.append(filtered_data)

    filtered_rdd = sc.union(filtered_rdds)
    outliers = filtered_rdd \
        .leftOuterJoin(original_data) \
        .map(lambda _: (_[0], _[1][1], _[1][0]))
    outliers.saveAsTextFile(output_hfs_path)
    print("Wrote output to {}".format(output_hfs_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Textual Outlier Detection')
    parser.add_argument("input_hfs_path", type=str, required=True)
    parser.add_argument("output_hfs_path", type=str, required=True)
    parser.add_argument("config_json_path", type=str, required=True)
    args = parser.parse_args()

    config = json.loads(args.config_json_path)
    main(
        input_hfs_path=args.input_hfs_path,
        output_hfs_path=args.output_hfs_path,
        config=config,
    )
