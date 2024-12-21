"""
Whisper benchmark runner
"""

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
        # TODO: the instance is a path to an .mp3 file on disk.
        audio = instance.get("audio", {}).get("array", None)
        if not audio:
            raise InferenceError("could not extract audio from instance")
        return self.processor(audio, return_tensors="tf")

    def inference(self, instance):
        audio = instance.input_features
        sequences = self.generate(input_features=audio)["sequences"]
        return self.processor.batch_decode(sequences, skip_special_tokens=True)
