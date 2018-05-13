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

        length_counts_srs = pd.Series(
            dict(length_counts.collect())
        ).sort_values()

        if self.low_quantile == 0:
            low_q = 0
        else:
            low_q = length_counts_srs.quantile(self.low_quantile)

        if self.high_quantile == 1.0:
            high_q = sys.maxsize
        else:
            high_q = length_counts_srs.quantile(self.high_quantile)

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
