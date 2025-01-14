"""
Offensive speech benchmark runner
"""

from .benchmark import Benchmark, limit_generator
from .models import load_offensive, cleanup_offensive
from .datasets import load_aegis, cleanup_aegis


class Offensive(Benchmark):

    @staticmethod
    def total(**kwargs):
        ## TODO: load this number from the manifest instead of counting
        data_home = kwargs.pop("data_home", None)
        use_sample = kwargs.pop("use_sample", True)
        return sum(1 for _ in load_aegis(data_home=data_home, sample=use_sample))

    @property
    def description(self):
        return (
            "applies the offensive speech detection model "
            "to the aegis content safety dataset"
        )

    def before(self):
        model, processor = load_offensive(model_home=self.model_home)
        self.model = model
        self.processor = processor

    def after(self, cleanup=True):
        if cleanup:
            cleanup_offensive(model_home=self.model_home)
            cleanup_aegis(data_home=self.data_home, sample=self.use_sample)

    def instances(self, limit=None):
        dataset = load_aegis(data_home=self.data_home, sample=self.use_sample)
        return limit_generator(dataset, limit)

    def preprocess(self, instance):
        raise NotImplementedError("Offensive preprocess not implemented")

    def inference(self, instance):
        raise NotImplementedError("Offensive inference not implemented")
