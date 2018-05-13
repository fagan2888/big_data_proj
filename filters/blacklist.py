from filters.core import AbstractFilter


class BlacklistFilter(AbstractFilter):

    def __init__(self,
                 case_sensitive_blacklist=None,
                 case_insensitive_blacklist=None):
        super(BlacklistFilter, self).__init__()
        self.case_sensitive_blacklist = case_sensitive_blacklist if \
            case_sensitive_blacklist else []
        self.case_insensitive_blacklist = case_insensitive_blacklist if \
            case_insensitive_blacklist else []

    @classmethod
    def initialize_from_config(cls, config):
        return cls(**config)

    @property
    def short_name(self):
        return "Blacklist Filter: CS (), CI ()".format(
            len(self.case_sensitive_blacklist),
            len(self.case_insensitive_blacklist),
        )

    def run_filter(self, data, filter_index=None):
        blacklist_inter1 = data \
            .map(lambda _: filter_operation(
                i=_[0], x=_[1],
                case_sensitive_blacklist=self.case_sensitive_blacklist,
                case_insensitive_blacklist=self.case_insensitive_blacklist,
                filter_index=filter_index,
            ))
        filtered_data = blacklist_inter1 \
            .filter(lambda _: _[2]) \
            .map(lambda _: (_[0], _[3]))
        remaining_data = blacklist_inter1 \
            .filter(lambda _: not _[2]) \
            .map(lambda _: (_[0], _[1]))
        return (
            remaining_data,
            filtered_data,
        )


def filter_operation(i, x,
                     case_sensitive_blacklist, case_insensitive_blacklist,
                     filter_index=None):
    index_string = "[{}] ".format(filter_index) if filter_index else ""
    if x in case_sensitive_blacklist:
        return i, x, True, "{}'{}' is in the blacklist".format(index_string, x)
    x = x.lower()
    if x in case_insensitive_blacklist:
        return i, x, True, "{}'{}' is in the blacklist".format(index_string, x)
    return i, x, False, None
