def filter_blacklist(data, cs_blacklist, ci_blacklist):
    blacklist_inter1 = data \
        .map(lambda _: filter_operation(
            i=_[0], x=_[1],
            cs_blacklist=cs_blacklist,
            ci_blacklist=ci_blacklist,
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


def filter_operation(i, x, cs_blacklist, ci_blacklist):
    if x in cs_blacklist:
        return i, x, True, "'{}' is in the blacklist".format(x)
    x = x.lower()
    if x in ci_blacklist:
        return i, x, True, "'{}' is in the blacklist".format(x)
    return i, x, False, None
