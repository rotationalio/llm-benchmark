"""
Primary entry point for construe CLI application
"""

import click
import torch
import platform

from .version import get_version
from .exceptions import DeviceError


from .datasets.path import get_data_home
from .datasets.loaders import cleanup_all_datasets
from .datasets.loaders import load_all_datasets, load_essays, load_aegis, load_nsfw
from .datasets.loaders import load_dialects, load_lowlight, load_reddit, load_movies

from .basic import BasicBenchmark
from .moondream import MoonDreamBenchmark


CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

DATASETS = [
    "all", "dialects", "lowlight", "reddit", "movies", "essays", "aegis", "nsfw",
]


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(get_version(), message="%(prog)s v%(version)s")
@click.option(
    "-d",
    "--device",
    default=None,
    type=str,
    envvar=["CONSTRUE_DEVICE", "TORCH_DEVICE"],
    help="specify the pytorch device to run on e.g. cpu, mps or cuda",
)
@click.option(
    "-e",
    "--env",
    default=None,
    envvar=["CONSTRUE_ENV", "ENV"],
    help="name of the experimental environment for comparison (default is hostname)",
)
@click.option(
    "-D",
    "--datadir",
    default=None,
    envvar="CONSTRUE_DATA",
    help="specify the location to download datasets to",
)
@click.option(
    "-C",
    "--cleanup/--no-cleanup",
    default=True,
    help="cleanup all downloaded datasets after the benchmark is run",
)
@click.pass_context
def main(ctx, env=None, device=None, datadir=None, cleanup=True):
    """
    A utility for executing inferencing benchmarks.
    """
    if device is not None:
        try:
            torch.set_default_device(device)
        except RuntimeError as e:
            raise DeviceError(e)

        click.echo(f"using torch.device(\"{device}\")")

    if env is None:
        env = platform.node()

    ctx.ensure_object(dict)
    ctx.obj["device"] = device
    ctx.obj["env"] = env
    ctx.obj["data_home"] = get_data_home(datadir)
    ctx.obj["cleanup"] = cleanup


@main.command()
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
@click.pass_context
def basic(ctx, **kwargs):
    """
    Runs basic dot product performance benchmarks.
    """
    kwargs["env"] = ctx.obj["env"]
    benchmark = BasicBenchmark(**kwargs)
    benchmark.run()


@main.command()
@click.pass_context
def moondream(ctx, **kwargs):
    """
    Executes image-to-text inferencing benchmarks.
    """
    kwargs["env"] = ctx.obj["env"]
    benchmark = MoonDreamBenchmark(**kwargs)
    benchmark.run()


@main.command()
@click.option(
    "-C",
    "--clean",
    default=False,
    help="cleanup the downloaded data cache and exit",
)
@click.option(
    "-d",
    "--download",
    default=None,
    type=click.Choice(DATASETS, case_sensitive=False),
)
@click.option(
    "-S",
    "--sample/--no-sample",
    default=True,
    help="if downloading a dataset, download only a sample",
)
@click.option(
    "-v",
    "--verbose/--no-verbose",
    default=False,
    help="print verbose info, otherwise just print the datasets path",
)
@click.pass_context
def datasets(ctx, clean=False, download=None, sample=True, verbose=False):
    """
    Helper utility for managing the dataset cache for benchmarks.
    """
    if clean and download:
        raise click.ClickException("cannot specify both --clean and --download")

    data_home = ctx.obj["data_home"]
    if clean:
        cleanup_all_datasets(data_home)
        return

    if download is not None:
        loader = {
            "all": load_all_datasets,
            "dialects": load_dialects,
            "lowlight": load_lowlight,
            "reddit": load_reddit,
            "movies": load_movies,
            "essays": load_essays,
            "aegis": load_aegis,
            "nsfw": load_nsfw
        }[download]

        next(loader(sample=sample, data_home=data_home))
        return

    # Provide some info
    if verbose:
        print(f"downloaded datasets are stored in {data_home}")
    else:
        print(data_home)


if __name__ == "__main__":
    main(
        obj={},
        prog_name="construe",
    )
