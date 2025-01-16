"""
Manifest handlers for downloading cloud resources and checking signatures.
"""

import os
import json
import glob
import zipfile

from urllib.parse import urljoin

from .signature import sha256sum
from ..version import get_version


BUCKET = "construe"
BASE_URL = "https://storage.googleapis.com/"

MODELS = "models"
DATASETS = "datasets"


def load_manifest(path):
    with open(path, "r") as f:
        return json.load(f)


def generate_manifest(fixtures, out, upload_type, extra=None):
    manifest = {}
    version = get_version(short=True)

    # Sort the list of paths by name
    paths = list(glob.glob(os.path.join(fixtures, "*.zip")))
    paths.sort()

    for path in paths:
        fname = os.path.basename(path)
        name, _ = os.path.splitext(fname)

        manifest[name] = {
            "url": make_fixture_url(fname, upload_type=upload_type, version=version),
            "signature": sha256sum(path),
            "size": {
                "compressed": os.path.getsize(path),
                "decompressed": get_uncompressed_size(path),
            },
        }

        if extra is not None:
            if callable(extra):
                manifest[name].update(extra(path=path, name=name, **manifest[name]))
            else:
                manifest[name].update(extra)

    with open(out, "w") as o:
        json.dump(manifest, o, indent=2)


def make_fixture_url(fname, upload_type, version=None):
    # Bucket must be joined here and not make_fixture_path to support uploading
    path = make_fixture_path(fname, upload_type, version)
    path = os.path.join(BUCKET, path)
    return urljoin(BASE_URL, path)


def make_fixture_path(fname, upload_type, version=None):
    version = version or get_version(short=True)
    return os.path.join(f"v{version}", upload_type, fname)


def get_uncompressed_size(path: str) -> int:
    bytes = 0
    with zipfile.ZipFile(path, 'r') as zf:
        for info in zf.infolist():
            bytes += info.file_size
    return bytes
