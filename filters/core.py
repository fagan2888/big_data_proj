class AbstractFilter:

    @classmethod
    def initialize_from_config(cls, config):
        raise NotImplementedError

    @property
    def short_name(self):
        raise NotImplementedError

    def run_filter(self, data, filter_index=None):
        """ returns (remaining_data, filtered_data)"""
        raise NotImplementedError
