# Getting Started

Construe is designed to be quickly installed on an Ubuntu environment with Internet connectivity using `pip` and the benchmarks run quickly downloading models and datasets from a cloud bucket and cleaning them up to minimize disk space utilization. At the conclusion of the benchmarks, an output JSON file with the results is saved to disk.

To quickly run the benchmarks:

```
$ pip install construe
$ construe -e "device-name" run
```

Where "device-name" is the name of the device or experiment that you're running the benchmarks on.

## Installation

This package is intended to be installed with `pip` and it will create a command line program `construe` on your `$PATH` to execute benchmarking comamnds:

```
$ pip install -U construe
$ which construe
$ construe --version
```

It is perfectly fine to install `construe` into a virtual environment (in fact it is recommended). If you get a `pip` error about not being able to resolve dependencies; please `pip uninstall` all of the specified dependencies and try installing `construe` again.

## Basic Usage

There are several top-level configurations that you can specify either as an environment variable or a command line option before the command. The environment variables are as follows:

- `$CONSTRUE_ENV` or `$ENV`: specify the name of the experimental environment for comparison purposes usually the name of the device (by default the hostname is used).
- `$CONSTRUE_DEVICE` or `$TORCH_DEVICE`: specify the name of the default accelerator to use with PyTorch e.g. cpu, mps, or cuda (this does not influence tflite).
- `$CONSTRUE_DATA`: the path to download datasets to for temporary storage during the benchmarks.
- `$CONSTRUE_MODELS`: the path to download models to for temporary storage during the benchmarks.

The command line utility help is as follows:

```
$ construe --help
Usage: construe [OPTIONS] COMMAND [ARGS]...

  A utility for executing inferencing benchmarks.

Options:
  --version                     Show the version and exit.
  -o, --out TEXT                specify the path to write the benchmark
                                results to
  -d, --device TEXT             specify the pytorch device to run on e.g. cpu,
                                mps or cuda
  -e, --env TEXT                name of the experimental environment for
                                comparison (default is hostname)
  -c, --count INTEGER           specify the number of times to run each
                                benchmark
  -l, --limit INTEGER           limit the number of instances to inference on
                                in each benchmark
  -D, --datadir TEXT            specify the location to download datasets to
  -M, --modeldir TEXT           specify the location to download models to
  -S, --sample / --no-sample    use sample dataset instead of full dataset for
                                benchmark
  -C, --cleanup / --no-cleanup  cleanup all downloaded datasets after the
                                benchmark is run
  -Q, --verbose / --quiet       specify the verbosity of the output and
                                progress bars
  -h, --help                    Show this message and exit.

Commands:
  basic      Runs basic dot product performance benchmarks.
  datasets   Helper utility for managing the dataset cache.
  gliner     Executes GLiNER named entity discovery inferencing benchmarks.
  lowlight   Executes lowlight image enhancement inferencing benchmarks.
  mobilenet  Executes image classification inferencing benchmarks.
  mobilevit  Executes object detection inferencing benchmarks.
  models     Helper utility for managing the models cache.
  moondream  Executes image-to-text inferencing benchmarks.
  nsfw       Executes NSFW image classification inferencing benchmarks.
  offensive  Executes offensive speech text classification inferencing...
  run        Executes all available benchmarks.
  whisper    Executes audio-to-text inferencing benchmarks.
```

## Run Benchmarks

You can either run all available benchamrks (excluding some, or specifying which benchmarks to include) or you can run an individual benchmark. To run all benchmarks:

```
$ construe -e "MacBook Pro 2022 M1" run
```

This will run all available benchmarks. The `-e` flag specifies the environment for comparison purposes and the results will be saved as a JSON file on the local disk.

When running each benchmark, the model and dataset for that benchmark is downloaded, the benchmark is executed, then the model and dataset are cleaned up. If you do not want the data to be cleaned up use the `-C` or `--no-cleanup` flag to cache the models and datasets between runs.

If you would like to limit the number of instances per run you can use the `-l` or `--limit` flag; this might speed up the benchmarks if you're just trying to get a simple sense of inferening on the device. You can also specify the `-c` or `--count` flag to run each benchmark multiple times on the same instances to get more detailed results.

To run an individual benchmark, run it by name; for example to run the `whisper` speech-to-text benchmark:

```
$ construe whisper
```

Alternatively if you want to exclude `whisper` (e.g. run all benchmarks but `whisper`), use the `-E` or `--exclude` flag as follows:

```
$ construe run -E whisper
```

## Data Storage

The benchmarks download both models and datasets (samples by default) from the cloud before executing the benchmark, then deletes the models and datasets after the benchmark is run to preserve device space. By default, data is stored in `$HOME/.construe`, however, construe allows you to configure where the models and data are stored so that you can specify a volume that has enough storage space.

To specify the location where the downloaded data and models are stored, use either the `-D` and `-M` flags with `construe` as follows:

```
$ construe -D /path/to/data -M /path/to/models run
```

Or set the `$CONSTRUE_DATA` and `$CONSTRUE_MODELS` environment variables with the specified paths.

You can determine where the model and datasets are stored byh using the following commands:

```
$ construe datasets
/home/user/.construe/data
```

```
$ construe models
/home/user/.construe/models
```

These commands also allow you to download the models and datasets before running the benchmarks. This is often preferable to ensure multiple runs of the benchmarks cache the data. Use the `--no-cleanup` flag with `construe` to ensure that manually downloaded models and datasets are not cleaned up after the benchmarks are run.

## Basic Benchmarks

The basic benchmarks implement dot product benchmarks from the [PyTorch documentation](https://pytorch.org/tutorials/recipes/recipes/benchmark.html). These benchmarks can be run using `construe basic`; for example by running:

```
$ construe -e "MacBook Pro 2022 M1" basic -o results-macbook.pickle
```

The `-e` flag specifies the environment for comparison purposes and the `-o` flag saves the measurements out to disk as a Pickle file that can be loaded for comparison to other environments later.

Command usage is as follows:

```
Usage: construe basic [OPTIONS]

Options:
  -e, --env TEXT             name of the experimental environment for
                             comparison (default is hostname)
  -o, --saveto TEXT          path to write the measurements pickle data to
  -t, --num-threads INTEGER  specify number of threads for benchmark (default
                             to maximum)
  -F, --fuzz / --no-fuzz     fuzz the tensor sizes of the inputs to the
                             benchmark
  -S, --seed INTEGER         set the random seed for random generation
  -h, --help                 Show this message and exit.
```