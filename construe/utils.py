"""
Utilities for construe
"""

from typing import Iterable, Set


def resolve_exclude(
    exclude: Iterable[str] = None,
    include: Iterable[str] = None,
    all: Iterable[str] = None,
) -> Set[str]:
    """
    Given an exclusion list and an inclusion list, merge them to produce a definitive
    exclusion list such that if there are any inclusions, then everything not in the
    inclusion list from all is added to the exclusion list. If an item is both in the
    exclusion list and the inclusion list it is excluded.
    """
    exclude = exclude or []
    exclude = set([item.strip().lower() for item in exclude])

    include = include or []
    include = set([
        item.strip().lower() for item in include
    ])

    if include:
        for item in all:
            if item not in include:
                exclude.add(item)

    return exclude
