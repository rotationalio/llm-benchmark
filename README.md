# Construe: An LLM Benchmark Utility

**An LLM inferencing benchmark tool focusing on device-specific latency and memory usage.**

## Quick Start

This package is intended to be installed with `pip` and it will create a command line program `construe` on your `$PATH` to execute benchmarking comamnds:

```
$ pip install construe
$ which construe
$ construe --help
```

## Releases

To release the construe library and deploy to PyPI run the following commands:

```
$ python -m build
$ twine upload dist/*
```