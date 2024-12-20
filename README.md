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

Options:
  --version          Show the version and exit.
  -d, --device TEXT  specify the pytorch device to run on e.g. cpu, mps or
                     cuda
  -e, --env TEXT     name of the experimental environment for comparison
                     (default is hostname)
  -h, --help         Show this message and exit.

Commands:
  basic
  moondream
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


## Developer Information

If you are a construe developer there are several helper utilities built into the library that will allow you to manage datasets and models both locally and in the cloud. But first, there are additional dependencies that you must install.

In `requirements.txt` uncomment the section that says: `"# Packaging Dependencies"`, e.g. your requirements should now have a section that appears similar to:

```
# Packaging Dependencies
black==24.10.0
build==1.2.2.post1
datasets==3.1.0
flake8==7.1.1
google-cloud-storage==2.19.0
packaging==24.2
pip==24.3.1
setuptools==75.3.0
twine==5.1.1
wheel==0.45.0
```

**NOTE:** the README might not be up to date with all required dependencies, so make sure you use the latest `requirements.txt`.

Then install these dependencies and the test dependencies:

```
$ pip install -r requirements.txt
$ pip install -r tests/requirements.txt
```

### Tests and Linting

All tests are in the `tests` folder and are structured similarly to the `construe` module. All tests can be run with `pytest`:

```
$ pytest
```

We use `flake8` for linting as configured in `setup.cfg` -- note that the `.flake8` file is for IDEs only and is not used when running tests. If you want to use `black` to automatically format your files:

```
$ black path/to/file.py
```

### Dataset Management

The `python -m construe.datasets` utility provides some helper functionality for managing datasets including the following commands:

- **manifest**: Generate a manifest file from local fixtures.
- **originals**: Download original datasets and store them in fixtures.
- **sample**: Create a sample dataset from the original that is smaller.
- **upload**: Upload datasets to GCP for user downloads.

To regenerate the datasets you would run the `originals` command first to download the datasets from HuggingFace or elsewhere on the web, then run `sample` to create statistical samples on those datasets. Run `manifest` to generate the new manifest for the datasets and SHA256 signatures, then run `upload` to save them to our GCP bucket.

You must have valid GCP service account credentials to upload datasets.

### Models Management

The `python -m construe.models` utility provides helpers for managing models and converting them to the tflite format including the following commands:

- **convert**: Convert source models to the tflite format for use in embeded systems.
- **manifest**: Generate a manifest file from local fixtures.
- **originals**: Download original models and store them in fixtures.
- **upload**: Upload converted models to GCP for user downloads.

To regenerate the models you would run the `originals` command to download the models from HuggingFace, then run `convert` to transform them into the tflite format. Run `manifest` to generate the new manifest for the models and SHA256 signatures, then run `upload` to save them to our GCP bucket.

You must have valid GCP service account credentials to upload datasets.

### Releases

To release the construe library and deploy to PyPI run the following commands:

```
$ python -m build
$ twine upload dist/*
```