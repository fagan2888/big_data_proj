from filters.blacklist import BlacklistFilter
from filters.length import LengthFilter
from filters.ngram_dist import NGramFilter

FILTER_MAP = {
    "Blacklist": BlacklistFilter,
    "Length": LengthFilter,
    "NGram": NGramFilter,
}


def resolve_filter(filter_config):
    filter_type = filter_config["filter"]
    return FILTER_MAP[filter_type].initialize_from_config(
        filter_config["config"]
    )
