"""
Manages datasets used for inferencing
"""

from .loaders import * # noqa
from .download import download_data, load_manifest
from .path import get_data_home, cleanup_dataset

try:
    DATASETS = load_manifest()
except Exception:
    DATASETS = None
