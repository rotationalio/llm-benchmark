"""
This module handles downloading data from original data sources and converting them
into the source format expected by the construe library.
"""

import os
import json
import shutil
import zipfile
import soundfile as sf

from .path import FIXTURES
from .download import CHUNK

from datasets import Audio
from datasets import load_dataset
from urllib.request import urlopen


AEGIS_BASENAME = "aegis.zip"
AEGIS_ZIPNAME = "aegis.jsonl"
AEGIS_DATASET_URL = "nvidia/Aegis-AI-Content-Safety-Dataset-1.0"

LOWLIGHT_BASENAME = "lowlight.zip"
LOWLIGHT_DATASET_URL = (
    "https://drive.google.com/uc?id=1DdGIJ4PZPlF2ikl8mNM9V-PdVxVLbQi6"
)

DIALECTS_BASENAME = "dialects.zip"
DIALECTS_DATASET_URL = "ylacombe/english_dialects"


def download_source_datasets(out=FIXTURES, exclude=None):
    """
    Download all source datasets to the specified out directory,
    excluding any by name in the downloaders dictionary.
    """
    downloaders = {
        "aegis": download_aegis,
        "lowlight": download_lowlight,
        "dialect": download_dialects,
    }

    exclude = exclude or []
    exclude = set([
        item.strip().lower() for item in exclude
    ])

    # Always exclude lowlight for now
    exclude.add("lowlight")

    for name, download in downloaders.items():
        if name in exclude:
            continue
        download(out=out)


def download_aegis(out=FIXTURES):
    """
    Download the AEGIS AI safety dataset from hugging faced and save as
    a zip compressed json lines file.
    """
    path = os.path.join(out, AEGIS_BASENAME)
    ds = load_dataset(AEGIS_DATASET_URL, split=None)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as z:
        with z.open(AEGIS_ZIPNAME, "w") as d:
            for split in ds.values():
                for row in split:
                    d.write(json.dumps(row).encode("utf-8"))
                    d.write("\n".encode("utf-8"))


def download_lowlight(url=LOWLIGHT_DATASET_URL, out=FIXTURES):
    """
    Download the lowlight dataset from Google Drive.

    NOTE: this dataset needs to be downloaded manually from a browser
    because a javascript button click is necessary to start the download.
    """
    path = os.path.join(out, LOWLIGHT_BASENAME)

    response = urlopen(url)
    with open(path, "wb") as f:
        while True:
            chunk = response.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)

    # TODO: unzip the download and create only one folder with all
    # the high and low photos without separation.


def download_dialects(out=FIXTURES):
    configs = [
        "irish_male",
        "midlands_female",
        "midlands_male",
        "northern_female",
        "northern_male",
        "scottish_female",
        "scottish_male",
        "southern_female",
        "southern_male",
        "welsh_female",
        "welsh_male",
    ]

    dirname, _ = os.path.splitext(DIALECTS_BASENAME)
    dir = os.path.join(out, dirname)
    if not os.path.exists(dir):
        os.makedirs(dir)

    for config in configs:
        ds = load_dataset(DIALECTS_DATASET_URL, config)
        cdir = os.path.join(dir, config)
        if not os.path.exists(cdir):
            os.makedirs(cdir)

        for split in ds.values():
            split = split.cast_column("audio", Audio(sampling_rate=16000))
            for row in split:
                path = os.path.join(cdir, row["audio"]["path"])
                audio = row["audio"]["array"] * 32767
                audio = audio.astype('int16')
                sf.write(path, audio, 16000)

    shutil.make_archive(dir, "zip", dir)
