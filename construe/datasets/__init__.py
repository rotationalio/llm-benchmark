"""
Manages datasets used for inferencing
"""

import glob


def load_content_moderation():
    for path in glob.glob("datasets/content-moderation/**/*"):
        yield path
