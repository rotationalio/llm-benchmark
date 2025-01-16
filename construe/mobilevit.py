"""
MobileViT benchmark runner
"""

from .exceptions import DatasetsError
from .benchmark import Benchmark, limit_generator
from .models import load_mobilevit, cleanup_mobilevit
from .datasets import load_movies, cleanup_movies, DATASETS


class MobileViT(Benchmark):

    @staticmethod
    def total(**kwargs):
        ## Return the number of movie stills from the manifest
        use_sample = kwargs.pop("use_sample", True)
        name = "movies-sample" if use_sample else "movies"
        if name not in DATASETS:
            raise DatasetsError("movies dataset not found in manifest")
        return DATASETS[name]["instances"]

    @property
    def description(self):
        return (
            "uses the MobileViT model to identify "
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
        raise NotImplementedError("MobileViT preprocess not implemented")

    def inference(self, instance):
        raise NotImplementedError("MobileViT inference not implemented")
