"""
Handle downloading datasets from our content URL
"""

import os
import zipfile

from tqdm import tqdm
from functools import partial
from urllib.request import urlopen

from .signature import sha256sum
from .manifest import load_manifest
from .path import get_data_home, cleanup_dataset
from .path import DIALECTS, LOWLIGHT, REDDIT, MOVIES, ESSAYS, AEGIS, NSFW


from construe.exceptions import DatasetsError


# Downlod chunk size
CHUNK = 524288


def download_data(url, signature, data_home=None, replace=False, extract=True):
    """
    Downloads the zipped data set specified at the given URL, saving it to
    the data directory specified by ``get_data_home``. This function verifies
    the download with the given signature and extracts the archive.
    """
    data_home = get_data_home(data_home)

    # Get the name of the file from the URL
    basename = os.path.basename(url)
    name, _ = os.path.splitext(basename)

    # Get the archive and data directory paths
    archive = os.path.join(data_home, basename)
    datadir = os.path.join(data_home, name)

    # If the archive exists cleanup or raise override exception
    if os.path.exists(archive):
        if not replace:
            raise DatasetsError(
                f"dataset already exists at {archive}, set replace=False to overwrite"
            )
        cleanup_dataset(name, data_home=data_home)

    # Create the output directory if it does not exist
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    # Fetch the response in a streaming fashion and write it to disk.
    response = urlopen(url)
    content_length = int(response.headers["Content-Length"])

    with open(archive, "wb") as f:
        pbar = tqdm(
            unit="B", total=content_length, desc=f"Downloading {basename}", leave=False
        )
        while True:
            chunk = response.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)
            pbar.update(len(chunk))

    # Compare the signature of the archive to the expected one
    if sha256sum(archive) != signature:
        raise ValueError("Download signature does not match hardcoded signature!")

    # If extract, extract the zipfile.
    if extract:
        zf = zipfile.ZipFile(archive)
        zf.extractall(path=datadir)


def _download_dataset(name, sample=True, data_home=True, replace=False, extract=True):
    if sample and not name.endswith("-sample"):
        name = name + "-sample"

    datasets = load_manifest()
    if name not in datasets:
        raise DatasetsError(f"no dataset named {name} exists")

    info = datasets[name]
    info.update({"data_home": data_home, "replace": replace, "extract": extract})
    download_data(**info)


download_dialects = partial(_download_dataset, DIALECTS)
download_lowlight = partial(_download_dataset, LOWLIGHT)
download_reddit = partial(_download_dataset, REDDIT)
download_movies = partial(_download_dataset, MOVIES)
download_essays = partial(_download_dataset, ESSAYS)
download_aegis = partial(_download_dataset, AEGIS)
download_nsfw = partial(_download_dataset, NSFW)


DOWNLOADERS = [
    download_dialects, download_lowlight, download_reddit,
    download_movies, download_essays, download_aegis, download_nsfw,
]


def download_all_datasets(sample=True, data_home=True, replace=True, extract=True):
    for f in DOWNLOADERS:
        f(sample=sample, data_home=data_home, replace=replace, extract=extract)
