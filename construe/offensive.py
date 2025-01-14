"""
Offensive speech benchmark runner
"""

from .benchmark import Benchmark


class Offensive(Benchmark):

    @staticmethod
    def total(**kwargs):
        ## TODO: load this number from the manifest instead of counting
        return 0

    @property
    def description(self):
        return (
            " "
            ""
        )

    def before(self):
        pass

    def after(self, cleanup=True):
        pass

    def instances(self, limit=None):
        pass

    def preprocess(self, instance):
        pass

    def inference(self, instance):
        pass
