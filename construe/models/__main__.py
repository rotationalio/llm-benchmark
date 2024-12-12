"""
Admin utility for managing models and converting them to tflite.
"""

import click

from construe.version import get_version

from .source import download_source_models, SOURCE_MODELS
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
    help="path to fixtures directory to download source datasets to",
)
@click.option(
    "-e",
    "--exclude",
    type=click.Choice(SOURCE_MODELS, case_sensitive=False),
    default=None,
    multiple=True,
    help="specify models to exclude from source download",
)
def originals(fixtures=FIXTURES, exclude=None):
    """
    Download original models and store them in fixtures.
    """
    download_source_models(out=FIXTURES, exclude=exclude)


if __name__ == "__main__":
    main(
        obj={},
        prog_name="python -m construe.models",
    )
