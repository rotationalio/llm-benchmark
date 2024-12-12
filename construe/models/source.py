"""
This package downloads and instantiates the source models from HuggingFace.
"""

import os

from .path import FIXTURES

from transformers import MobileViTImageProcessor, MobileViTForImageClassification
from transformers import AutoModelForImageClassification, AutoImageProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import ViTImageProcessor
from transformers import TFWhisperModel

from gliner import GLiNER
from huggingface_hub import from_pretrained_keras


MOONDREAM_ID = "vikhyatk/moondream2"
MOONDREAM_REVISION = "2024-08-26"
MOONDREAM_DIR = "moondream"
MOONDREAM_SAVED_MODEL_DIR = "hf_moondream_saved"
MOONDREAM_TFLITE = "moondream.tflite"

WHISPER_ID = "openai/whisper-tiny.en"
WHISPER_REVISION = "main"
WHISPER_DIR = "whisper"
WHISPER_SAVED_MODEL_DIR = "tf_whisper_saved"
WHISPER_TFLITE = "whisper.tflite"

MOBILENET_ID = "google/mobilenet_v2_1.0_224"
MOBILENET_REVISION = "main"
MOBILENET_DIR = "mobilenet"
MOBILENET_SAVED_MODEL_DIR = "hf_mobilenet_saved"
MOBILENET_TFLITE = "mobilenet.tflite"

MOBILEVIT_ID = "apple/mobilevit-xx-small"
MOBILEVIT_REVISION = "main"
MOBILEVIT_DIR = "mobilevit"
MOBILEVIT_SAVED_MODEL_DIR = "hf_mobilevit_saved"
MOBILEVIT_TFLITE = "mobilevit.tflite"

NSFW_ID = "Falconsai/nsfw_image_detection"
NSFW_REVISION = "main"
NSFW_DIR = "nsfw"
NSFW_SAVED_MODEL_DIR = "hf_nsfw_saved"
NSFW_TFLITE = "nsfw.tflite"

LOWLIGHT_ID = "keras-io/lowlight-enhance-mirnet"
LOWLIGHT_DIR = "lowlight"
LOWLIGHT_SAVED_MODEL_DIR = "tf_lowlight_saved"
LOWLIGHT_TFLITE = "lowlight.tflite"

OFFENSIVE_ID = "KoalaAI/OffensiveSpeechDetector"
OFFENSIVE_REVISION = "main"
OFFENSIVE_DIR = "offensive"
OFFENSIVE_SAVED_MODEL_DIR = "hf_offensive_saved"
OFFENSIVE_TFLITE = "offensive.tflite"

GLINER_ID = "knowledgator/gliner-bi-small-v1.0"
GLINER_DIR = "gliner"
GLINER_SAVED_MODEL_DIR = "hf_gliner_saved"
GLINER_TFLITE = "gliner.tflite"

SOURCE_MODELS = [
    MOONDREAM_DIR,
    WHISPER_DIR,
    MOBILENET_DIR,
    MOBILEVIT_DIR,
    NSFW_DIR,
    LOWLIGHT_DIR,
    OFFENSIVE_DIR,
    GLINER_DIR,
]


def download_source_models(out=FIXTURES, exclude=None):
    """
    Download all source models to the specified directory, excluding any by name.
    """
    downloaders = {
        MOONDREAM_DIR: download_moondream,
        WHISPER_DIR: download_whisper,
        MOBILENET_DIR: download_mobilenet,
        MOBILEVIT_DIR: download_mobilevit,
        NSFW_DIR: download_nsfw,
        LOWLIGHT_DIR: download_lowlight,
        OFFENSIVE_DIR: download_offensive,
        GLINER_DIR: download_gliner,
    }

    exclude = exclude or []
    exclude = set([
        item.strip().lower() for item in exclude
    ])

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

    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    return path


def download_whisper(out=FIXTURES, model_id=WHISPER_ID, revision=WHISPER_REVISION):
    """
    Download the whisper tiny english model and save it to fixtures.
    """
    model = TFWhisperModel.from_pretrained(model_id, revision=revision)
    path = os.path.join(out, WHISPER_DIR, WHISPER_SAVED_MODEL_DIR)
    model.save(path)
    return path


def download_mobilenet(
    out=FIXTURES, model_id=MOBILENET_ID, revision=MOBILENET_REVISION
):
    """
    Download the MobileNet model and save it to fixtures.
    """
    preprocessor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModelForImageClassification.from_pretrained(
        model_id, revision=revision, trust_remote_code=True
    )

    path = os.path.join(out, MOBILENET_DIR, MOBILENET_SAVED_MODEL_DIR)

    model.save_pretrained(path)
    preprocessor.save_pretrained(path)

    return path


def download_mobilevit(
    out=FIXTURES, model_id=MOBILEVIT_ID, revision=MOBILEVIT_REVISION
):
    """
    Download the MobileNet model and save it to fixtures.
    """
    preprocessor = MobileViTImageProcessor.from_pretrained(model_id, revision=revision)
    model = MobileViTForImageClassification.from_pretrained(model_id, revision=revision)

    path = os.path.join(out, MOBILEVIT_DIR, MOBILEVIT_SAVED_MODEL_DIR)

    model.save_pretrained(path)
    preprocessor.save_pretrained(path)

    return path


def download_nsfw(out=FIXTURES, model_id=NSFW_ID, revision=NSFW_REVISION):
    """
    Download the NSFW image classification model and save it to fixtures.
    """
    preprocessor = ViTImageProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModelForImageClassification.from_pretrained(model_id, revision=revision)

    path = os.path.join(out, NSFW_DIR, NSFW_SAVED_MODEL_DIR)

    model.save_pretrained(path)
    preprocessor.save_pretrained(path)

    return path


def download_lowlight(out=FIXTURES, model_id=LOWLIGHT_ID):
    """
    Download the lowlight Keras model and safe it to fixtures.
    """
    model = from_pretrained_keras(model_id)
    path = os.path.join(out, LOWLIGHT_DIR, LOWLIGHT_SAVED_MODEL_DIR)
    model.save(path)
    return path


def download_offensive(
    out=FIXTURES, model_id=OFFENSIVE_ID, revision=OFFENSIVE_REVISION
):
    """
    Download the offensive speech detector model and save it to fixtures.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    path = os.path.join(out, OFFENSIVE_DIR, OFFENSIVE_SAVED_MODEL_DIR)

    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    return path


def download_gliner(out=FIXTURES, model_id=GLINER_ID):
    """
    Download the gliner model and save it to fixtures.
    """
    model = GLiNER.from_pretrained(model_id)
    path = os.path.join(out, GLINER_DIR, GLINER_SAVED_MODEL_DIR)
    model.save_pretrained(path)
    return path
