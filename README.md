# Construe: An LLM Benchmark Utility

**An LLM inferencing benchmark tool focusing on device-specific latency and memory usage.**

## Quick Start

This package is intended to be installed with `pip` and it will create a command line program `construe` on your `$PATH` to execute benchmarking comamnds:

```
$ pip install construe
$ which construe
$ construe --help
```

## Basic Benchmarks

The basic benchmarks implement dot product benchmarks from the [PyTorch documentation](https://pytorch.org/tutorials/recipes/recipes/benchmark.html). These benchmarks can be run using `construe basic`; for example by running:

```
$ construe basic -e "MacBook Pro 2022 M1" -o results-macbook.pickle
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

## Moondream Benchmarks

The [moondream](https://huggingface.co/vikhyatk/moondream2) package contains small image-to-text computer vision models that can be used in the first step of a [content moderation](https://www.cloudraft.io/blog/content-moderation-using-llamaindex-and-llm) workflow (e.g. image to text, moderate text). This benchmark executes the model for _encoding_ and _inferencing_ on a small number of images and reports the average time for both operations and the line-by-line memory usage of the model.

It can be run as follows:

```
$ construe moondream
```

Command usage is as follows:

```
Usage: construe moondream [OPTIONS]

Options:
  -h, --help  Show this message and exit.
```

## Releases

To release the construe library and deploy to PyPI run the following commands:

```
$ python -m build
$ twine upload dist/*
```