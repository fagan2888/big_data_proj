import argparse
import json

from pyspark.sql import SparkSession


def main(input_hfs_path,
         outliers_output_hfs_path,
         clean_output_hfs_path,
         config):
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

    if config["show_counts"]:
        total_count = data.count()
        remaining_count = total_count
    else:
        total_count = remaining_count = -1

    total_filtered_count = 0
    for filter_index, filter_config in enumerate(config["filters"]):
        filter_instance = resolve_filter(filter_config)
        print("Running [{}] {}".format(
            filter_index, filter_instance.short_name
        ))
        data, filtered_data = filter_instance.run_filter(
            data=data,
            filter_index=filter_index,
        )
        filtered_rdds.append(filtered_data)
        if config["show_counts"]:
            filtered_count = filtered_data.count()
            total_filtered_count += filtered_count
            print("  Filtered out {} observation{} -- "
                  "{:.2f} of total, {:.2f} of remainder".format(
                        filtered_count,
                        "" if filtered_count == 1 else "s",
                        filtered_count / total_count * 100,
                        filtered_count / remaining_count * 100
                    ))
            remaining_count -= filtered_count

    if config["show_counts"]:
        print("ORIGINAL:  {} ({:.2f}%)".format(
            total_count, 100,
        ))
        print("FILTERED:  {} ({:.2f}%)".format(
            total_filtered_count, total_filtered_count / total_count * 100,
        ))
        print("REMAINING: {} ({:.2fou}%)".format(
            remaining_count, remaining_count / total_count * 100,
        ))

    filtered_rdd = sc.union(filtered_rdds)
    outliers = filtered_rdd \
        .leftOuterJoin(original_data) \
        .map(lambda _: (_[0], _[1][1], _[1][0]))

    print("Writing outliers to {}".format(outliers_output_hfs_path))
    outliers.saveAsTextFile(outliers_output_hfs_path)

    if clean_output_hfs_path:
        print("Writing clean output to {}".format(clean_output_hfs_path))
        data.saveAsTextFile(clean_output_hfs_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Textual Outlier Detection')
    parser.add_argument("--input_hfs_path", type=str, required=True)
    parser.add_argument("--outliers_output_hfs_path", type=str, required=True)
    parser.add_argument("--clean_output_hfs_path",
                        type=str, required=False, default="")
    parser.add_argument("--config_json_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.config_json_path, "r") as f:
        config_ = json.loads(f.read())
    main(
        input_hfs_path=args.input_hfs_path,
        outliers_output_hfs_path=args.outliers_output_hfs_path,
        clean_output_hfs_path=args.clean_output_hfs_path,
        config=config_,
    )
