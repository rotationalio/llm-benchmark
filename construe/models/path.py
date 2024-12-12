"""
Path handling for model downloads
"""

import os

from pathlib import Path


# Fixtures is where model data being prepared is stored
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
MANIFEST = os.path.join(os.path.dirname(__file__), "manifest.json")

# Models dir is the location of downloaded model files
MODELSDIR = Path.home() / ".construe" / "models"


def get_models_home(path=None):
    """
    Return the path of the Construe models directory. This folder is used by
    model loaders to avoid downloading model parameters several times.

    By default, this folder is colocated with the code in the install directory
    so that data shipped with the package can be easily located. Alternatively
    it can be set by the ``$CONSTRUE_DATA`` environment variable, or
    programmatically by giving a folder path. Note that the ``'~'`` symbol is
    expanded to the user home directory, and environment variables are also
    expanded when resolving the path.
    """
    if path is None:
        path = os.environ.get("CONSTRUE_MODELS", MODELSDIR)

    path = os.path.expanduser(path)
    path = os.path.expandvars(path)

    if not os.path.exists(path):
        os.makedirs(path)

    return path
