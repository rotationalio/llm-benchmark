"""
LowLight image enhancement benchmark runner
"""

from .benchmark import Benchmark, limit_generator
from .models import load_lowlight as load_lowlight_model
from .models import cleanup_lowlight as cleanup_lowlight_model
from .datasets import load_lowlight as load_lowlight_dataset
from .datasets import cleanup_lowlight as cleanup_lowlight_dataset


class LowLight(Benchmark):

    @staticmethod
    def total(**kwargs):
        ## TODO: load this number from the manifest instead of counting
        data_home = kwargs.pop("data_home", None)
        use_sample = kwargs.pop("use_sample", True)
        return sum(
            1 for _ in load_lowlight_dataset(data_home=data_home, sample=use_sample)
        )

    @property
    def description(self):
        return (
            "lowlight enhances image quality using a convolutional "
            "model that understands how to enrich low-light images"
        )

    def before(self):
        model, processor = load_lowlight_model(model_home=self.model_home)
        self.model = model
        self.processor = processor

    def after(self, cleanup=True):
        if cleanup:
            cleanup_lowlight_model(model_home=self.model_home)
            cleanup_lowlight_dataset(data_home=self.data_home, sample=self.use_sample)

    def instances(self, limit=None):
        dataset = load_lowlight_dataset(
            data_home=self.data_home, sample=self.use_sample
        )
        return limit_generator(dataset, limit)

    def preprocess(self, instance):
        raise NotImplementedError("LowLight preprocess not implemented")

    def inference(self, instance):
        raise NotImplementedError("LowLight inference not implemented")
