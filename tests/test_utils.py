import pytest

from construe.utils import resolve_exclude


@pytest.mark.parametrize("exclude,include,expected", [
    (
        None, None, set(),
    ),
    (
        ["red", "orange", "black"], None, {"red", "orange", "black"},
    ),
    (
        None, ["red", "orange", "black"], {"blue", "green", "pink", "purple", "violet"},
    ),
    (
        ["red", "orange"], ["purple", "violet"], {"red", "blue", "orange", "green", "pink", "black"},
    ),
    (
        ["red", "orange"], ["red", "purple", "violet"], {"red", "blue", "orange", "green", "pink", "black"},
    )
])
def test_resolve_exclude(exclude, include, expected):
    """
    Test that exclusions are resolved correctly
    """
    ALL = [
        "red", "blue", "green", "pink", "purple", "orange", "violet", "black",
    ]
    resolved = resolve_exclude(exclude, include, ALL)
    assert resolved == expected
