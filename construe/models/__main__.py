"""
Admin utility for managing models and converting them to tflite.
"""

import click

from construe.version import get_version

from .source import download_source_models, SOURCE_MODELS
from .convert import convert_source_models
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
    help="path to fixtures directory to download source models to",
)
@click.option(
    "-e",
    "--exclude",
    type=click.Choice(SOURCE_MODELS, case_sensitive=False),
    default=None,
    multiple=True,
    help="specify models to exclude from source download",
)
@click.option(
    "-i",
    "--include",
    type=click.Choice(SOURCE_MODELS, case_sensitive=False),
    default=None,
    multiple=True,
    help="specify models to explictly include in tflite conversions",
)
def originals(fixtures=FIXTURES, exclude=None, include=None):
    """
    Download original models and store them in fixtures.
    """
    download_source_models(out=FIXTURES, exclude=exclude, include=include)


@main.command(epilog=EPILOG)
@click.option(
    "-f",
    "--fixtures",
    type=str,
    default=FIXTURES,
    help="path to fixtures directory where source models have been downloaded",
)
@click.option(
    "-e",
    "--exclude",
    type=click.Choice(SOURCE_MODELS, case_sensitive=False),
    default=None,
    multiple=True,
    help="specify models to exclude from tflite conversion",
)
@click.option(
    "-i",
    "--include",
    type=click.Choice(SOURCE_MODELS, case_sensitive=False),
    default=None,
    multiple=True,
    help="specify models to explictly include in tflite conversions",
)
def convert(fixtures=FIXTURES, exclude=None, include=None):
    """
    Convert source models to the tflite format for use in benchmarks.
    """
    convert_source_models(out=FIXTURES, exclude=exclude, include=include)


if __name__ == "__main__":
    main(
        obj={},
        prog_name="python -m construe.models",
    )
