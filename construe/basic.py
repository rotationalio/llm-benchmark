"""
Benchmarks basic dot product torch operators.
"""

import torch


def batched_dot_mul_sum(a, b):
    """
    Computes batched dot by multiplying and summing
    """
    return a.mul(b).sum(-1)


def batched_dot_bmm(a, b):
    """
    Computes batched dot by reducing to bmm
    """
    a = a.reshape(-1, 1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a, b).flatten(-3)
