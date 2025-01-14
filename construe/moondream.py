"""
Moondream is a computer vision model (image to text) that is optimized for use
on embedded devices and serves as an example model in content moderation use
cases where the image is captioned and then the caption is moderated.
"""

from .benchmark import Benchmark, limit_generator
from .datasets import load_nsfw, cleanup_nsfw
from .models import load_moondream, cleanup_moondream


class MoonDream(Benchmark):

    @staticmethod
    def total(**kwargs):
        # TODO: load this number from the manifest instead of counting
        data_home = kwargs.pop("data_home", None)
        use_sample = kwargs.pop("use_sample", True)
        return sum(1 for _ in load_nsfw(data_home=data_home, sample=use_sample))

    @property
    def description(self):
        return (
            "performs content moderation by captioning an image and "
            "then using the offensive speech model to moderate the caption"
        )

    def before(self):
        model, processor = load_moondream(model_home=self.model_home)
        self.model = model
        self.processor = processor
        self.generate = self.model.get_signature_runner()

    def after(self, cleanup=True):
        if cleanup:
            cleanup_moondream(model_home=self.model_home)
            cleanup_nsfw(data_home=self.data_home, sample=self.use_sample)

    def instances(self, limit=None):
        dataset = load_nsfw(data_home=self.data_home, sample=self.use_sample)
        return limit_generator(dataset, limit)

    def preprocess(self, instance):
        raise NotImplementedError("MoonDream preprocess not implemented")

    def inference(self, instance):
        raise NotImplementedError("MoonDream inference not implemented")
