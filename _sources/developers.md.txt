# Development Guidelines

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

**NOTE:** the docs might not be up to date with all required dependencies, so make sure you use the latest `requirements.txt`.

Then install these dependencies and the test dependencies:

```
$ pip install -r requirements.txt
$ pip install -r tests/requirements.txt
```

## Tests and Linting

All tests are in the `tests` folder and are structured similarly to the `construe` module. All tests can be run with `pytest`:

```
$ pytest
```

We use `flake8` for linting as configured in `setup.cfg` -- note that the `.flake8` file is for IDEs only and is not used when running tests. If you want to use `black` to automatically format your files:

```
$ black path/to/file.py
```

## Dataset Management

The `python -m construe.datasets` utility provides some helper functionality for managing datasets including the following commands:

- **manifest**: Generate a manifest file from local fixtures.
- **originals**: Download original datasets and store them in fixtures.
- **sample**: Create a sample dataset from the original that is smaller.
- **upload**: Upload datasets to GCP for user downloads.

To regenerate the datasets you would run the `originals` command first to download the datasets from HuggingFace or elsewhere on the web, then run `sample` to create statistical samples on those datasets. Run `manifest` to generate the new manifest for the datasets and SHA256 signatures, then run `upload` to save them to our GCP bucket.

You must have valid GCP service account credentials to upload datasets.

## Models Management

The `python -m construe.models` utility provides helpers for managing models and converting them to the tflite format including the following commands:

- **convert**: Convert source models to the tflite format for use in embeded systems.
- **manifest**: Generate a manifest file from local fixtures.
- **originals**: Download original models and store them in fixtures.
- **upload**: Upload converted models to GCP for user downloads.

To regenerate the models you would run the `originals` command to download the models from HuggingFace, then run `convert` to transform them into the tflite format. Run `manifest` to generate the new manifest for the models and SHA256 signatures, then run `upload` to save them to our GCP bucket.

You must have valid GCP service account credentials to upload datasets.

## Releases

To release the construe library and deploy to PyPI run the following commands:

```
$ python -m build
$ twine upload dist/*
```