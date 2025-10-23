from dataclasses import dataclasses
from typing import Iterable
from .inference import Inference

@dataclasses
class Trainer:
    method: Inference
    dataloader: Iterable # Dataloader

    def fit(self, n: int):
        """Fit the inference method for `n` iterations"""
        for i in range(n):
            batch = self.dataloader.next()
            method.step(batch)