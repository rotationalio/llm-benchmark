"""
Whisper benchmark runner
"""

import soundfile as sf

from .benchmark import Benchmark
from .exceptions import InferenceError
from .models import load_whisper, cleanup_whisper
from .datasets import load_dialects, cleanup_dialects


class Whisper(Benchmark):

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

    def instances(self):
        return load_dialects(data_home=self.data_home, sample=self.use_sample)

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
