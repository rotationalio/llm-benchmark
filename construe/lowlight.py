"""
LowLight image enhancement benchmark runner
"""

import os
import numpy as np
import tensorflow as tf

from .datasets import DATASETS
from .exceptions import DatasetsError
from .benchmark import Benchmark, limit_generator
from .models import load_lowlight as load_lowlight_model
from .models import cleanup_lowlight as cleanup_lowlight_model
from .datasets import load_lowlight as load_lowlight_dataset
from .datasets import cleanup_lowlight as cleanup_lowlight_dataset


class LowLight(Benchmark):

    @staticmethod
    def total(**kwargs):
        ## Return number of lowlight images from the manifest
        use_sample = kwargs.pop("use_sample", True)
        name = "lowlight-sample" if use_sample else "lowlight"
        if name not in DATASETS:
            raise DatasetsError("lowlight dataset not found in manifest")
        return DATASETS[name]["classes"]["low"]

    @property
    def description(self):
        return (
            "lowlight enhances image quality using a convolutional "
            "model that understands how to enrich low-light images"
        )

    def before(self):
        # Load and setup the interpreter for the lowlight dataset
        model = load_lowlight_model(model_home=self.model_home)
        model.resize_tensor_input(0, [1, 400, 600, 3])
        model.allocate_tensors()

        self.model = model
        self.input_details = model.get_input_details()
        self.output_details = model.get_output_details()

    def after(self, cleanup=True):
        if cleanup:
            cleanup_lowlight_model(model_home=self.model_home)
            cleanup_lowlight_dataset(data_home=self.data_home, sample=self.use_sample)

    def instances(self, limit=None):
        dataset = load_lowlight_dataset(
            data_home=self.data_home, sample=self.use_sample,
        )

        def filter_instances(dataset):
            for instance in dataset:
                if os.path.dirname(instance).endswith("low"):
                    yield instance

        return limit_generator(filter_instances(dataset), limit)

    def preprocess(self, instance):
        image = tf.io.read_file(instance)
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.cast(image, dtype=tf.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        return image

    def inference(self, instance):
        self.model.set_tensor(self.input_details[0]["index"], instance)
        self.model.invoke()
        self.model.get_tensor(self.output_details[0]["index"])
