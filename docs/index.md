# Construe: An LLM Benchmark Utility

**An LLM inferencing benchmark tool focusing on device-specific latency and memory usage.**

## Quick Start

This package is intended to be installed with `pip` and it will create a command line program `construe` on your `$PATH` to execute benchmarking comamnds:

```
$ pip install construe
$ which construe
$ construe --help
```

There are several top-level configurations that you can specify either as an environment variable or a command line option before the command. The environment variables are as follows:

- `$CONSTRUE_ENV` or `$ENV`: specify the name of the experimental environment for comparison purposes.
- `$CONSTRUE_DEVICE` or `$TORCH_DEVICE`: specify the name of the default device to use with PyTorch e.g. cpu, mps, or cuda.

The command line utility help is as follows:

```
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
  models     Helper utility for managing the models cache.
  moondream  Executes image-to-text inferencing benchmarks.
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

## Model References

1. Image to Text: [Moondream (vikhyatk/moondream2)](https://huggingface.co/vikhyatk/moondream2)
2. Speech to Text: [Whisper (openai/whisper-tiny.en)](https://huggingface.co/docs/transformers/en/model_doc/whisper)
3. Image Classification: [MobileNet (google/mobilenet_v2_1.0_224)](https://huggingface.co/docs/transformers/en/model_doc/mobilenet_v2)
4. Object Detection: [MobileViT (apple/mobilevit-xx-small)](https://huggingface.co/docs/transformers/en/model_doc/mobilevit)
5. NSFW Image Classification [Fine-Tuned Vision Transformer (ViT) for NSFW Image Classification (Falconsai/nsfw_image_detection)](https://huggingface.co/Falconsai/nsfw_image_detection)
6. Image Enhancement [LoL MIRNet (keras-io/lowlight-enhance-mirnet)](https://huggingface.co/keras-io/lowlight-enhance-mirnet)
7. Text Classification: [Offensive Speech Detector (KoalaAI/OffensiveSpeechDetector)](https://huggingface.co/KoalaAI/OffensiveSpeechDetector)
8. Token Classification: [GLiNER (knowledgator/gliner-bi-small-v1.0)](https://huggingface.co/knowledgator/gliner-bi-small-v1.0)

## Dataset References

1. [AEGIS AI Content Safety v1.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0): Text data that is used to show examples of content safety (e.g. harmful text) described by Nvidia's content safety taxonomy.
2. [LoL (Low-Light) Dataset](https://paperswithcode.com/dataset/lol): Contains 500 low-light and normal-light image pairs for image enhancement.
3. [English Dialects](https://huggingface.co/datasets/ylacombe/english_dialects): Contains 31 hours of audo from 120 individuals speaking with different accents of the British Isles and is used for speech to text.
4. [Reddit Posts Comments](https://huggingface.co/datasets/ummagumm-a/reddit_posts_comments): A text dataset of comments on Reddit posts that can be used for NER and content moderation tasks on short form text.
5. [Student and LLM Essays](https://huggingface.co/datasets/knarasi1/student_and_llm_essays): A text dataset of essays written by students (and LLMs) that can be used for NER and content moderation tasks on longer form text.
6. [NSFW Detection](https://huggingface.co/datasets/zanderlewis/nsfw_detection_large): An image dataset that contains NSFW and SFW images used for content moderation.
7. [Movie Scenes](https://huggingface.co/datasets/unography/movie-scenes): An image dataset that contains stills from commercial movies and can be used for image classification and content-moderation tasks.

```{toctree}
:maxdepth: 3

api
```