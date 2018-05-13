import numpy as np
import pandas as pd
import sys

from operator import add

from filters.core import AbstractFilter


class LengthFilter(AbstractFilter):

    def __init__(self, low_quantile=0.001, high_quantile=1.0):
        super(LengthFilter, self).__init__()
        self.low_quantile = low_quantile
        self.high_quantile = high_quantile

    @classmethod
    def initialize_from_config(cls, config):
        return cls(**config)

    @property
    def short_name(self):
        return "Length Filter with quantiles [{}, {}]".format(
            self.low_quantile, self.high_quantile,
        )

    def run_filter(self, data, filter_index=None):
        length_inter1 = data \
            .map(lambda _: (len(_[1]), _[0], _[1]))
        length_counts = length_inter1 \
            .map(lambda _: (_[0], 1)) \
            .reduceByKey(add)

        length_counts_map = pd.Series(
            dict(length_counts.collect())
        ).sort_index()
        running_sum = length_counts_map.cumsum()
        total_count = length_counts_map.sum()

        if self.low_quantile == 0:
            low_q_n = 0
        else:
            low_q_n = int(np.floor(total_count * self.low_quantile))

        if self.high_quantile == 1.0:
            high_q_n = total_count
        else:
            high_q_n = int(np.ceil(total_count * self.high_quantile))

        low_q = running_sum[running_sum >= low_q_n].index[0]
        high_q = running_sum[running_sum <= high_q_n].index[-1]

        filtered_data = length_inter1 \
            .filter(lambda _: not (low_q < _[0] < high_q)) \
            .map(lambda _: (
                _[1],
                "{}Length {} not within ({}, {})".format(
                    "[{}] ".format(filter_index) if filter_index else "",
                    _[0], int(low_q), int(high_q)
                )))
        remaining_data = length_inter1 \
            .filter(lambda _: low_q < _[0] < high_q) \
            .map(lambda _: (_[1], _[2]))
        return (
            remaining_data,
            filtered_data,
        )
