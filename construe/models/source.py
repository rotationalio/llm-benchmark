"""
This package downloads and instantiates the source models from HuggingFace.
"""

import os

from .path import FIXTURES

from transformers import TFWhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MOONDREAM_ID = "vikhyatk/moondream2"
MOONDREAM_REVISION = "2024-08-26"
MOONDREAM_DIR = "moondream"
MOONDREAM_SAVED_MODEL_DIR = "transformers_moondream_saved"
MOONDREAM_TFLITE = "moondream.tflite"

WHISPER_ID = "openai/whisper-tiny.en"
WHISPER_REVISION = "main"
WHISPER_DIR = "whisper"
WHISPER_SAVED_MODEL_DIR = "tf_whisper_saved"
WHISPER_TFLITE = "whisper.tflite"

SOURCE_MODELS = [MOONDREAM_DIR, WHISPER_DIR]


def download_source_models(out=FIXTURES, exclude=None):
    """
    Download all source models to the specified directory, excluding any by name.
    """
    downloaders = {
        MOONDREAM_DIR: download_moondream,
        WHISPER_DIR: download_whisper,
    }

    exclude = exclude or []
    exclude = set([
        item.strip().lower() for item in exclude
    ])

    # Always exclude moondream for now
    exclude.add("moondream")

    for name, download in downloaders.items():
        if name in exclude:
            continue
        download(out=out)


def download_moondream(
    out=FIXTURES, model_id=MOONDREAM_ID, revision=MOONDREAM_REVISION
):
    """
    Download the moondream model and save it to fixtures.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    path = os.path.join(out, MOONDREAM_DIR, MOONDREAM_SAVED_MODEL_DIR)
    model.save_pretrained_merged(path, tokenizer)
    return path


def download_whisper(out=FIXTURES, model_id=WHISPER_ID, revision=WHISPER_REVISION):
    """
    Download the whisper tiny english model and save it to fixtures.
    """
    model = TFWhisperModel.from_pretrained(model_id, revision=revision)
    path = os.path.join(out, WHISPER_DIR, WHISPER_SAVED_MODEL_DIR)
    model.save(path)
    return path
