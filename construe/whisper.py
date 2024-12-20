"""
Whisper benchmark runner
"""

from typing import Dict

from .datasets import load_dialects
from .exceptions import InferenceError
from .models import load_whisper, cleanup_whisper


class Whisper(object):

    def __init__(self, **kwargs):
        self.model_home = None
        self.data_home = None
        for key, val in kwargs.items():
            setattr(self, key, val)

    def before(self):
        model, processor = load_whisper(model_home=self.model_home)
        self.model = model
        self.processor = processor
        self.generate = self.model.get_signature_runner()

    def run(self):
        dataset = load_dialects(data_home=self.data_home)
        for instance in dataset:
            # TODO: time how long it takes to perform the inference
            # TODO: measure memory consumption during inferencing
            # TODO: make this function part of the base object
            transcript = self.inference(instance)
            print(transcript)

    def after(self):
        cleanup_whisper(model_home=self.model_home)

    def preprocess(self, instance: Dict):
        audio = instance.get("audio", {}).get("array", None)
        if not audio:
            raise InferenceError("could not extract audio from instance")

        return self.processor(audio, return_tensors="tf")

    def inference(self, instance: Dict):
        audio = self.preprocess(instance)
        sequences = self.generate(input_features=audio.input_featrures)["sequences"]
        return self.processor.batch_decode(sequences, skip_special_tokens=True)
