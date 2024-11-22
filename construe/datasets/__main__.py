"""
Admin utility for managing datasets.
"""

import click

from construe.version import get_version

from .source import download_source_datasets, SOURCE_DATASETS
from .manifest import generate_manifest
from .path import FIXTURES


CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

EPILOG = "This tool is intended for construe developer use and not construe users."


@click.group(context_settings=CONTEXT_SETTINGS, epilog=EPILOG)
@click.version_option(get_version(), message="%(prog)s v%(version)s")
def main():
    pass


@main.command(epilog=EPILOG)
@click.option(
    "-f",
    "--fixtures",
    type=str,
    default=FIXTURES,
    help="path to fixtures directory to generate manifest from",
)
@click.option(
    "-o",
    "--out",
    type=str,
    default=None,
    help="path to write the manifest to",
)
def manifest(fixtures=FIXTURES, out=None):
    """
    Generate a manifest file from local fixtures.
    """
    generate_manifest(fixtures, out)


@main.command(epilog=EPILOG)
@click.option(
    "-f",
    "--fixtures",
    type=str,
    default=FIXTURES,
    help="path to fixtures directory to download source datasets to",
)
@click.option(
    "-e",
    "--exclude",
    type=click.Choice(SOURCE_DATASETS, case_sensitive=False),
    default=None,
    multiple=True,
    help="specify datasets to exclude from source download",
)
def originals(fixtures=FIXTURES, exclude=None):
    """
    Download original datasets and store them in fixtures.
    """
    download_source_datasets(out=FIXTURES, exclude=exclude)


if __name__ == "__main__":
    main(
        obj={},
        prog_name="construe-datasets",
    )
