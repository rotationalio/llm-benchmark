"""
Upload models to the google cloud bucket (developers only).
"""

import os
import glob
import json

from ..version import get_version
from ..models.source import SOURCE_MODELS
from .manifest import make_fixture_path, MODELS
from ..models.path import FIXTURES as MODEL_FIXTURES

try:
    from google.cloud import storage
except ImportError:
    storage = None


CONSTRUE_BUCKET = "construe"
GOOGLE_CREDENTIALS = "GOOGLE_APPLICATION_CREDENTIALS"


def upload_models(fixtures=MODEL_FIXTURES, exclude=None, include=None, credentials=None):
    """
    Upload all models from the model fixtures directory.
    """
    exclude = exclude or []
    exclude = set([item.strip().lower() for item in exclude])

    include = include or []
    include = set([
        item.strip().lower() for item in include
    ])

    if include:
        for source in SOURCE_MODELS:
            if source not in include:
                exclude.add(source)

    version = get_version(short=True)
    client = connect_storage(credentials)

    for name in SOURCE_MODELS:
        if name in exclude:
            continue

        name += ".zip"
        dst = make_fixture_path(name, MODELS, version=version)
        src = os.path.join(fixtures, name)

        upload(dst, src, client)


def upload(name, path, client=None, bucket=CONSTRUE_BUCKET):
    """
    Upload data from source path to a bucket with destination name.
    """
    if client is None:
        client = connect_storage()

    if not os.path.exists(path) or not os.path.isfile(path):
        raise ValueError("no zip file exists at " + path)

    bucket = client.get_bucket(bucket)
    blob = bucket.blob(name)
    blob.upload_from_filename(path)

    return blob.public_url


def connect_storage(credentials=None):
    """
    Create a google cloud storage client and connect.
    """
    # Attempt to fetch credentials from environment
    credentials = credentials or os.environ.get(GOOGLE_CREDENTIALS, None)

    # Attempt to get credentials from the .secret folder
    credentials = credentials or find_service_account()

    if credentials is None:
        raise RuntimeError(
            "could not find service account credentials: "
            "set either $GOOGLE_APPLICATION_CREDENTIALS to the path "
            "or store the credentials in the .secret folder"
        )

    # Cannot connect without the storage library.
    if storage is None:
        raise ImportError(
            "the google.cloud.storage module is required, install using pip"
        )

    return storage.Client.from_service_account_json(credentials)


def find_service_account():
    secret = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..", "..", ".secret", "*.json"
    ))

    for path in glob.glob(secret):
        with open(path, "r") as f:
            data = json.load(f)
            if "universe_domain" in data and data["universe_domain"] == "googleapis.com":
                return path

    return None
