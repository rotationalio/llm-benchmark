"""
MobileNet benchmark runner
"""

from .benchmark import Benchmark, limit_generator
from .models import load_mobilevit, cleanup_mobilevit
from .datasets import load_movies, cleanup_movies


class MobileNet(Benchmark):

    @staticmethod
    def total(**kwargs):
        ## TODO: load this number from the manifest instead of counting
        data_home = kwargs.pop("data_home", None)
        use_sample = kwargs.pop("use_sample", True)
        return sum(1 for _ in load_movies(data_home=data_home, sample=use_sample))

    @property
    def description(self):
        return (
            "uses the MobileNet v2 model to classify "
            "objects in scenes from movie stills"
        )

    def before(self):
        model, processor = load_mobilevit(model_home=self.model_home)
        self.model = model
        self.processor = processor

    def after(self, cleanup=True):
        if cleanup:
            cleanup_mobilevit(model_home=self.model_home)
            cleanup_movies(data_home=self.data_home, sample=self.use_sample)

    def instances(self, limit=None):
        dataset = load_movies(data_home=self.data_home, sample=self.use_sample)
        return limit_generator(dataset, limit)

    def preprocess(self, instance):
        raise NotImplementedError("MobileNet preprocess not implemented")

    def inference(self, instance):
        raise NotImplementedError("MobileNet inference not implemented")
