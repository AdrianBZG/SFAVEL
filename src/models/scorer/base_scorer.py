"""
Module defining a base scorer. Children classes implement specific scoring functions.
"""


class BaseScorer:
    def __init__(self):
        super().__init__()

    def score(self, z_claims, z_triplets):
        raise NotImplementedError("A child class should be implemented for scoring")
