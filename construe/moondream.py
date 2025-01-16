"""
Moondream is a computer vision model (image to text) that is optimized for use
on embedded devices and serves as an example model in content moderation use
cases where the image is captioned and then the caption is moderated.
"""

from .exceptions import DatasetsError
from .benchmark import Benchmark, limit_generator
from .models import load_moondream, cleanup_moondream
from .datasets import load_nsfw, cleanup_nsfw, DATASETS


class MoonDream(Benchmark):

    @staticmethod
    def total(**kwargs):
        # Return the number of nsfw images from the manifest
        use_sample = kwargs.pop("use_sample", True)
        name = "nsfw-sample" if use_sample else "nsfw"
        if name not in DATASETS:
            raise DatasetsError("nsfw dataset not found in manifest")
        return DATASETS[name]["instances"]

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
