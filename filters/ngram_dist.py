import numpy as np

from operator import add
from pyspark.ml.feature import CountVectorizer, NGram


def filter_ngrams(data, ngram_n, logprob_quantile_cutoff):
    data_df = data.map(lambda _: (_[0], _[1], list(_[1]))).toDF(
        ["id", "text", "splittext"])
    ngram = NGram(n=ngram_n, inputCol="splittext", outputCol="ngrams")
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

    logprob = flat_ngram \
        .map(lambda _: (_[1], _[0])) \
        .join(all_ngram_dist) \
        .map(lambda _: (_[1][0], np.log(_[1][1]))) \
        .aggregateByKey(
        (0, 0),
        lambda a, b: (a[0] + b, a[1] + 1),
        lambda a, b: (a[0] + b[0], a[1] + b[1])
    ) \
        .map(lambda _: (_[0], float(_[1][0] / _[1][1])))
    logprob_df = logprob.toDF(["id", "logprob"])
    logprob_cutoff = \
    logprob_df.stat.approxQuantile(
        "logprob", [logprob_quantile_cutoff], 0.001)[0]
    filtered_ngrams = logprob \
        .filter(lambda _: _[1] < logprob_cutoff) \
        .map(lambda _: (
            _[0], "{}-gram logprob smaller than {} quantile at {}"
                .format(ngram_n, logprob_quantile_cutoff, _[1])
        ))
    remaining_data = logprob \
        .filter(lambda _: _[1] < logprob_cutoff) \
        .join(data) \
        .map(lambda _: (_[0], _[1][1]))
    return (
        remaining_data,
        filtered_ngrams,
    )
