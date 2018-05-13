import collections as col
import numpy as np

from operator import add
from pyspark.ml.feature import CountVectorizer, NGram
import pyspark.sql.functions

from filters.core import AbstractFilter


class NGramFilter(AbstractFilter):

    def __init__(self, ngram_n, score_type, score_quantile_cutoff,
                 accuracy=0.001):
        super(NGramFilter, self).__init__()
        self.ngram_n = ngram_n
        self.score_type = score_type
        self.score_quantile_cutoff = score_quantile_cutoff
        self.accuracy = accuracy

    @property
    def short_name(self):
        return "{}-gram Filter by {} (cutoff={})".format(
            self.ngram_n, self.score_type, self.score_quantile_cutoff
        )

    @classmethod
    def initialize_from_config(cls, config):
        return cls(**config)

    def run_filter(self, data, filter_index=None):
        data_df = data.toDF(["id", "text"])
        data_df = data_df.withColumn(
            'splittext', pyspark.sql.functions.split(data_df['text'], ''))

        ngram = NGram(n=self.ngram_n, inputCol="splittext", outputCol="ngrams")
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
            .map(lambda _: (_[0], _[1] / total_count))

        if self.score_type == "logprob":
            logprob_score = flat_ngram \
                .map(lambda _: (_[1], _[0])) \
                .join(all_ngram_dist) \
                .map(lambda _: (_[1][0], np.log(_[1][1]))) \
                .aggregateByKey(
                    (0, 0),
                    lambda a, b: (a[0] + b, a[1] + 1),
                    lambda a, b: (a[0] + b[0], a[1] + b[1])
                ) \
                .map(lambda _: (_[0], float(_[1][0] / _[1][1])))
            score = logprob_score
        elif self.score_type == "kl":
            ngrams_rdd = data_df_ngram \
                .select(["id", "ngrams", "text"]).rdd

            ngram_counts_within_row = ngrams_rdd \
                .map(lambda _: (_[0], dict(col.Counter(_[1])), len(_[1]))) \
                .flatMap(lambda _: [(k, (_[0], v, _[2])) for k, v in _[1].items()])

            ngram_row_data = ngram_counts_within_row \
                .join(all_ngram_dist)

            negative_kl_ngram = ngram_row_data \
                .map(lambda _: (
                    _[1][0][0],
                    float(_[1][1] * (np.log(_[1][0][1] / _[1][0][2]) - np.log(_[1][1]))))
                     )

            negative_kl_score = negative_kl_ngram \
                .reduceByKey(add)
            score = negative_kl_score
        else:
            raise KeyError(self.score_type)

        score_df = score.toDF(["id", "score"])
        score_cutoff = score_df.stat.approxQuantile(
            "score", [self.score_quantile_cutoff], self.accuracy)[0]

        filtered_ngrams = score \
            .filter(lambda _: _[1] < score_cutoff) \
            .map(lambda _: (
                _[0],
                "{}{}-gram {} smaller than {} quantile at {}".format(
                    "[{}] ".format(filter_index) if filter_index else "",
                    self.ngram_n, self.score_type, self.score_quantile_cutoff,
                    _[1]
                )
            ))
        remaining_data = score \
            .filter(lambda _: _[1] > score_cutoff) \
            .join(data) \
            .map(lambda _: (_[0], _[1][1]))
        return (
            remaining_data,
            filtered_ngrams,
        )
