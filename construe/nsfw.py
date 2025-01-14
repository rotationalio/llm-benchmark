"""
NSFW Image Classification benchmark runner
"""

from .benchmark import Benchmark, limit_generator
from .models import load_nsfw as load_nsfw_model
from .models import cleanup_nsfw as cleanup_nsfw_model
from .datasets import load_nsfw as load_nsfw_dataset
from .datasets import cleanup_nsfw as cleanup_nsfw_dataset


class NSFW(Benchmark):

    @staticmethod
    def total(**kwargs):
        ## TODO: load this number from the manifest instead of counting
        data_home = kwargs.pop("data_home", None)
        use_sample = kwargs.pop("use_sample", True)
        return sum(1 for _ in load_nsfw_dataset(data_home=data_home, sample=use_sample))

    @property
    def description(self):
        return (
            "uses a fine-tuned model to classify images as "
            "safe or not safe for work (nsfw)"
        )

    def before(self):
        model, processor = load_nsfw_model(model_home=self.model_home)
        self.model = model
        self.processor = processor

    def after(self, cleanup=True):
        if cleanup:
            cleanup_nsfw_model(model_home=self.model_home)
            cleanup_nsfw_dataset(data_home=self.data_home, sample=self.use_sample)

    def instances(self, limit=None):
        dataset = load_nsfw_dataset(data_home=self.data_home, sample=self.use_sample)
        return limit_generator(dataset, limit)

    def preprocess(self, instance):
        raise NotImplementedError("NSFW preprocess not implemented")

    def inference(self, instance):
        raise NotImplementedError("NSFW inference not implemented")
