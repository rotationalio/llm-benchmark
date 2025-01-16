"""
Whisper benchmark runner
"""

import soundfile as sf

from .exceptions import DatasetsError
from .exceptions import InferenceError
from .benchmark import Benchmark, limit_generator
from .models import load_whisper, cleanup_whisper
from .datasets import load_dialects, cleanup_dialects, DATASETS


class Whisper(Benchmark):

    @staticmethod
    def total(**kwargs):
        # Return the number of dialects from the manifest
        use_sample = kwargs.pop("use_sample", True)
        name = "dialects-sample" if use_sample else "dialects"
        if name not in DATASETS:
            raise DatasetsError("dialects dataset not found in manifest")
        return DATASETS[name]["instances"]

    @property
    def description(self):
        return (
            "utilizes the whisper-tiny english model to "
            "transcribe audio from various UK dialects"
        )

    def before(self):
        model, processor = load_whisper(model_home=self.model_home)
        self.model = model
        self.processor = processor
        self.generate = self.model.get_signature_runner()

    def after(self, cleanup=True):
        if cleanup:
            cleanup_whisper(model_home=self.model_home)
            cleanup_dialects(data_home=self.data_home, sample=self.use_sample)

    def instances(self, limit=None):
        dataset = load_dialects(data_home=self.data_home, sample=self.use_sample)
        return limit_generator(dataset, limit)

    def preprocess(self, instance):
        # Instance is a path to a a sound file on disk.
        try:
            audio, samplerate = sf.read(instance)
        except Exception as e:
            raise InferenceError("could not extract audio from file") from e

        audio = audio.astype('float64')
        audio = audio / 32767.0

        return self.processor(audio, sampling_rate=samplerate, return_tensors="tf")

    def inference(self, instance):
        audio = instance.input_features
        sequences = self.generate(input_features=audio)["sequences"]
        return self.processor.batch_decode(sequences, skip_special_tokens=True)
