# Construe: An LLM Benchmark Utility

Construe is an LLM benchmarking utility that focuses on _inferencing_ latency and memory usage on devices that support inferencing acceleration. It currently focuses on the embedded device space and uses `tflite` as the inferencing environment (though PyTorch support may be added in the future). The primary purpose of this library is to understand how different acceleration support such as [APUs](https://en.wikipedia.org/wiki/AMD_APU), [NPUs](https://en.wikipedia.org/wiki/AI_accelerator), [TPUs](https://en.wikipedia.org/wiki/Tensor_Processing_Unit), and [GPUs](https://en.wikipedia.org/wiki/Graphics_processing_unit) behave differently for AI in an edge or resource constrained environment.

Although this Python package can be used as a library for benchmarking purposes; the primary use is via the installation of a command line program, `construe`. This program runs multiple or individual benchmarks and saves the results to an output file. When running the benchmarks, for each benchmark, `construe` takes the following actions:

1. Download the model
2. Download the dataset (by default, a sample of the full dataset)
3. Execute benchmark
4. Cleanup the model and dataset
5. Repeat steps 1-4 for all remaining benchmarks
6. Save the output results to disk

Model and dataset downloads are from a hosted cloud bucket rather than from Hugging Face or from GitHub to ensure that the data can be geographically located in proximity to where the benchmarks are being run and to isolate specific versions and conversions of the models and datasets.

In order to conserve space on limited devices - the downloads are cleaned up before moving on to the next benchmark. Samples of the dataset are downloaded by default to also reduce space utilization.

The output results are saved in a JSON format for easy analysis and reporting.

## Table of Contents

```{toctree}
:maxdepth: 2

quickstart
benchmarks
developers
api/index
```