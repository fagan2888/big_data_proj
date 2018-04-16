import pandas as pd
import numpy as np

from operator import add


def filter_length(data, low_quantile, high_quantile):
    length_inter1 = data \
        .map(lambda _: (len(_[1]), _[0], _[1]))
    length_counts = length_inter1 \
        .map(lambda _: (_[0], 1)) \
        .reduceByKey(add)

    length_counts_srs = pd.Series(
        dict(length_counts.collect())
    ).sort_values()

    if low_quantile == 0:
        low_q = 0
    else:
        low_q = length_counts_srs.quantile(low_quantile)

    if high_quantile == 1.0:
        high_q = np.inf
    else:
        high_q = length_counts_srs.quantile(high_quantile)

    filtered_data = length_inter1 \
        .filter(lambda _: not (low_q < _[0] < high_q)) \
        .map(lambda _: (_[1], "Length {} not within ({}, {})".format(
        _[0], int(low_q), int(high_q))
                        ))
    remaining_data = length_inter1 \
        .filter(lambda _: low_q < _[0] < high_q) \
        .map(lambda _: (_[1], _[2]))
    return (
        remaining_data,
        filtered_data,
    )
