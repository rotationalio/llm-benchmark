"""
GLiNER named entity discovery benchmark runner
"""

from .benchmark import Benchmark, limit_generator
from .models import load_gliner, cleanup_gliner
from .datasets import load_essays, cleanup_essays


class GLiNER(Benchmark):

    @staticmethod
    def total(**kwargs):
        ## TODO: load this number from the manifest instead of counting
        data_home = kwargs.pop("data_home", None)
        use_sample = kwargs.pop("use_sample", True)
        return sum(1 for _ in load_essays(data_home=data_home, sample=use_sample))

    @property
    def description(self):
        return (
            "applies the GLiNER model to identify and classify "
            "named entities in long-form essays"
        )

    def before(self):
        model, processor = load_gliner(model_home=self.model_home)
        self.model = model
        self.processor = processor

    def after(self, cleanup=True):
        if cleanup:
            cleanup_gliner(model_home=self.model_home)
            cleanup_essays(data_home=self.data_home, sample=self.use_sample)

    def instances(self, limit=None):
        dataset = load_essays(data_home=self.data_home, sample=self.use_sample)
        return limit_generator(dataset, limit)

    def preprocess(self, instance):
        raise NotImplementedError("GLiNER preprocess not implemented")

    def inference(self, instance):
        raise NotImplementedError("GLiNER inference not implemented")
