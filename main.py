import argparse
import json

from pyspark.sql import SparkSession


def main(input_hfs_path, output_hfs_path, config):
    from filters.api import resolve_filter
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

    for filter_index, filter_config in enumerate(config):
        filter_instance = resolve_filter(config)
        print("Running [{}] {}".format(
            filter_index, filter_instance.short_name
        ))
        data, filtered_data = filter_instance.run_filter(
            data=data,
            filter_index=filter_index,
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
    parser.add_argument("--input_hfs_path", type=str, required=True)
    parser.add_argument("--output_hfs_path", type=str, required=True)
    parser.add_argument("--config_json_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.config_json_path, "r") as f:
        config = json.loads(f.read())
    main(
        input_hfs_path=args.input_hfs_path,
        output_hfs_path=args.output_hfs_path,
        config=config,
    )
