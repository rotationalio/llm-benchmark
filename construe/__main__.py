"""
Primary entry point for construe CLI application
"""

import click

from .version import get_version


CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(get_version(), message="%(prog)s v%(version)s")
def main():
    pass


if __name__ == "__main__":
    main(
        prog_name="construe",
    )
