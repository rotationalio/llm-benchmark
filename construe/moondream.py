"""
Moondream is a computer vision model (image to text) that is optimized for use
on embedded devices and serves as an example model in content moderation use
cases where the image is captioned and then the caption is moderated.
"""

import time
import tqdm
import numpy as np

from PIL import Image
from memory_profiler import profile
from construe.datasets import load_content_moderation
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "vikhyatk/moondream2"
REVISION = "2024-08-26"


class MoonDreamBenchmark(object):

    def __init__(self):
        self.moondream = MoonDreamProfiler()
        self.dataset = list(load_content_moderation())

    @profile
    def run(self):
        results = []
        for path in tqdm.tqdm(self.dataset):
            encoded, encode_time = self.moondream.encode_image(path)
            inference_time = self.moondream.inference(encoded)
            results.append((encode_time, inference_time))

        encode_mean = np.array([result[0] for result in results]).mean()
        inference_mean = np.array([result[1] for result in results]).mean()

        print(f"Encode Average: {encode_mean:0.2f}")
        print(f"Inference Average: {inference_mean:0.2f}")


class MoonDreamProfiler(object):

    def __init__(self):
        self.load_model()

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=True, revision=REVISION,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)

    def encode_image(self, path):
        image = Image.open(path)
        start = time.perf_counter()
        encoded = self.model.encode_image(image)
        delta = time.perf_counter() - start
        return encoded, delta

    def inference(self, image):
        start = time.perf_counter()
        self.model.answer_question(
            image, "Describe this image in detail with transparency.", self.tokenizer
        )
        return time.perf_counter() - start
