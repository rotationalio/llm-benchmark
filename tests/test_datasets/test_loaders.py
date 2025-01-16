"""
Tests the data set loaders and downloaders to make sure they work as expected.
"""

import os
import pytest

from construe.datasets.loaders import *


# Data loader tests required downloading data from the bucket; skipping them by default
# unless the environment variable is set will make the tests run faster. In CI this
# environment variable can be set to protect from regressions.
CONSTRUE_TEST_DATA_LOADERS = os.environ.get("CONSTRUE_TEST_DATA_LOADERS", False)
CONSTRUE_TEST_SAMPLE_LOADERS = os.environ.get("CONSTRUE_TEST_SAMPLE_LOADERS", False)


@pytest.mark.skipif(not CONSTRUE_TEST_DATA_LOADERS, reason="skipping data loaders tests")
def test_load_dialects(tmpdir):
    count = 0
    for dialect in load_dialects(data_home=tmpdir, sample=False):
        count += 1
        assert isinstance(dialect, str) and dialect.endswith(".wav")

    assert count == 17877
    cleanup_dialects(data_home=tmpdir)


@pytest.mark.skipif(not CONSTRUE_TEST_SAMPLE_LOADERS, reason="skipping data loaders tests")
def test_load_dialects_sample(tmpdir):
    count = 0
    for dialect in load_dialects(data_home=tmpdir, sample=True):
        count += 1
        assert isinstance(dialect, str) and dialect.endswith(".wav")

    assert count == 1785
    cleanup_dialects(data_home=tmpdir)


@pytest.mark.skipif(not CONSTRUE_TEST_DATA_LOADERS, reason="skipping data loaders tests")
def test_load_lowlight(tmpdir):
    count = 0
    for image in load_lowlight(data_home=tmpdir, sample=False):
        count += 1
        assert isinstance(image, str) and image.endswith(".png")

    assert count == 1000
    cleanup_dialects(data_home=tmpdir)


@pytest.mark.skipif(not CONSTRUE_TEST_SAMPLE_LOADERS, reason="skipping data loaders tests")
def test_load_lowlight_sample(tmpdir):
    count = 0
    for image in load_lowlight(data_home=tmpdir, sample=True):
        count += 1
        assert isinstance(image, str) and image.endswith(".png")

    assert count == 475
    cleanup_lowlight(data_home=tmpdir)


@pytest.mark.skipif(not CONSTRUE_TEST_DATA_LOADERS, reason="skipping data loaders tests")
def test_load_reddit(tmpdir):
    count = 0
    for row in load_reddit(data_home=tmpdir, sample=False):
        count += 1
        assert isinstance(row, dict) and "comment" in row

    assert count == 3844
    cleanup_reddit(data_home=tmpdir)


@pytest.mark.skipif(not CONSTRUE_TEST_SAMPLE_LOADERS, reason="skipping data loaders tests")
def test_load_reddit_sample(tmpdir):
    count = 0
    for row in load_reddit(data_home=tmpdir, sample=True):
        count += 1
        assert isinstance(row, dict) and "comment" in row

    assert count == 957
    cleanup_reddit(data_home=tmpdir)


@pytest.mark.skipif(not CONSTRUE_TEST_DATA_LOADERS, reason="skipping data loaders tests")
def test_load_movies(tmpdir):
    count = 0
    for image in load_movies(data_home=tmpdir, sample=False):
        count += 1
        assert isinstance(image, str) and image.endswith(".jpg")

    assert count == 106844
    cleanup_movies(data_home=tmpdir)


@pytest.mark.skipif(not CONSTRUE_TEST_SAMPLE_LOADERS, reason="skipping data loaders tests")
def test_load_movies_sample(tmpdir):
    count = 0
    for image in load_movies(data_home=tmpdir, sample=True):
        count += 1
        assert isinstance(image, str) and image.endswith(".jpg")

    assert count == 5465
    cleanup_movies(data_home=tmpdir)


@pytest.mark.skipif(not CONSTRUE_TEST_DATA_LOADERS, reason="skipping data loaders tests")
def test_load_essays(tmpdir):
    count = 0
    for row in load_essays(data_home=tmpdir, sample=False):
        count += 1
        assert isinstance(row, dict) and "essay" in row

    assert count == 2078
    cleanup_essays(data_home=tmpdir)


@pytest.mark.skipif(not CONSTRUE_TEST_SAMPLE_LOADERS, reason="skipping data loaders tests")
def test_load_essays_sample(tmpdir):
    count = 0
    for row in load_essays(data_home=tmpdir, sample=True):
        count += 1
        assert isinstance(row, dict) and "essay" in row

    assert count == 512
    cleanup_essays(data_home=tmpdir)


@pytest.mark.skipif(not CONSTRUE_TEST_DATA_LOADERS, reason="skipping data loaders tests")
def test_load_aegis(tmpdir):
    count = 0
    for row in load_aegis(data_home=tmpdir, sample=False):
        count += 1
        assert isinstance(row, dict) and "labels_0" in row and "text" in row

    assert count == 11997
    cleanup_aegis(data_home=tmpdir)


@pytest.mark.skipif(not CONSTRUE_TEST_SAMPLE_LOADERS, reason="skipping data loaders tests")
def test_load_aegis_sample(tmpdir):
    count = 0
    for row in load_aegis(data_home=tmpdir, sample=True):
        count += 1
        assert isinstance(row, dict) and "labels_0" in row and "text" in row

    assert count == 3030
    cleanup_aegis(data_home=tmpdir)


@pytest.mark.skipif(not CONSTRUE_TEST_DATA_LOADERS, reason="skipping data loaders tests")
def test_load_nsfw(tmpdir):
    count = 0
    for image in load_nsfw(data_home=tmpdir, sample=False):
        count += 1
        assert isinstance(image, str) and image.endswith(".jpg")

    assert count == 215
    cleanup_nsfw(data_home=tmpdir)


@pytest.mark.skipif(not CONSTRUE_TEST_SAMPLE_LOADERS, reason="skipping data loaders tests")
def test_load_nsfw_sample(tmpdir):
    count = 0
    for image in load_nsfw(data_home=tmpdir, sample=True):
        count += 1
        assert isinstance(image, str) and image.endswith(".jpg")

    assert count == 53
    cleanup_nsfw(data_home=tmpdir)
