"""
Manifest handlers for downloading and checking signatures
"""

import os
import json
import glob

from urllib.parse import urljoin

from .signature import sha256sum
from ..version import get_version
from .path import FIXTURES, MANIFEST


BASE_URL = "https://storage.googleapis.com/"
DATASETS = "datasets"
APPLICATION = "construe"


def load_manifest(path=MANIFEST):
    with open(MANIFEST, "r") as f:
        return json.load(f)


def generate_manifest(fixtures=FIXTURES, out=MANIFEST):
    out = out or MANIFEST

    manifest = {}
    version = get_version(short=True)

    for path in glob.glob(os.path.join(fixtures, "*.zip")):
        fname = os.path.basename(path)
        name, _ = os.path.splitext(fname)

        manifest[name] = {
            "url": make_fixture_url(fname, version=version),
            "signature": sha256sum(path),
        }

    with open(out, "w") as o:
        json.dump(manifest, o, indent=2)


def make_fixture_url(fname, app=APPLICATION, upload_type=DATASETS, version=None):
    version = version or get_version(short=True)
    path = os.path.join(app, f"v{version}", upload_type, fname)
    return urljoin(BASE_URL, path)