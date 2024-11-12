"""
Primary entry point for construe CLI application
"""

import click

from .version import get_version
from .basic import BasicBenchmark


CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(get_version(), message="%(prog)s v%(version)s")
def main():
    pass


@main.command()
@click.option(
    "-e",
    "--env",
    default=None,
    help="name of the experimental environment for comparison (default is hostname)",
)
@click.option(
    "-o",
    "--saveto",
    default=None,
    help="path to write the measurements pickle data to",
)
@click.option(
    "-t",
    "--num-threads",
    default=None,
    type=int,
    help="specify number of threads for benchmark (default to maximum)",
)
@click.option(
    "-F",
    "--fuzz/--no-fuzz",
    default=False,
    help="fuzz the tensor sizes of the inputs to the benchmark",
)
@click.option(
    "-S",
    "--seed",
    default=None,
    type=int,
    help="set the random seed for random generation",
)
def basic(**kwargs):
    benchmark = BasicBenchmark(**kwargs)
    benchmark.run()


if __name__ == "__main__":
    main(
        prog_name="construe",
    )
