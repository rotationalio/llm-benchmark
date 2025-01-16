"""
Manifest handlers for datasets
"""

import os
import glob
import json
import shutil
import tempfile

from collections import defaultdict

from .path import FIXTURES, MANIFEST
from ..cloud.manifest import DATASETS
from ..cloud.manifest import load_manifest as _load_manifest
from ..cloud.manifest import generate_manifest as _generate_manifest


def load_manifest(path=MANIFEST):
    return _load_manifest(path)


def generate_manifest(fixtures=FIXTURES, out=MANIFEST):
    out = out or MANIFEST
    return _generate_manifest(fixtures, out, DATASETS, extra=dataset_extra)


def dataset_extra(path, name, **kwargs):
    """
    Count the number of instances in each class in the dataset
    """
    with tempfile.TemporaryDirectory() as data_home:
        basename = os.path.basename(path)
        archive = os.path.join(data_home, basename)
        datadir = os.path.join(data_home, name)

        # Copy and extract the archive
        shutil.copy(path, archive)
        shutil.unpack_archive(archive, datadir)

        extra = {"instances": 0, "classes": defaultdict(int)}
        name = name.removesuffix("-sample")

        # Count the number of instances
        {
            "dialects": _count_files,
            "lowlight": _count_files,
            "reddit": _count_jsonl,
            "movies": _count_files,
            "essays": _count_jsonl,
            "aegis": _count_jsonl,
            "nsfw": _count_files,
        }[name](name, datadir, extra)

        return extra


def _count_files(name, datadir, extra):
    patterns = {
        "lowlight": "lowlight/**/*.png",
        "nsfw": "nsfw/**/*.jpg",
    }

    if name in patterns:
        pattern = patterns[name]
    else:
        pattern = "**/*"

    for path in glob.glob(os.path.join(datadir, pattern)):
        if os.path.isdir(path):
            continue

        label = os.path.basename(os.path.dirname(path))
        extra["instances"] += 1
        extra["classes"][label] += 1


def _count_jsonl(name, datadir, extra):
    for path in glob.glob(os.path.join(datadir, "*.jsonl")):
        with open(path, "r") as f:
            for line in f:
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    continue

                extra["instances"] += 1

    del extra["classes"]
