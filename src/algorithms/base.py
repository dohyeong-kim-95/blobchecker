"""Abstract base class for all blobchecker algorithms."""

from abc import ABC, abstractmethod
import numpy as np


class BaseAlgorithm(ABC):
    """
    Every algorithm must implement these three methods.

    The evaluator calls them in a fixed loop:
        row, col = algorithm.next_query()
        algorithm.update(row, col, oracle.query(row, col))
        predicted = algorithm.predict()
    """

    def __init__(self, H: int, W: int, budget: int):
        self.H = H
        self.W = W
        self.budget = budget

    @abstractmethod
    def next_query(self) -> tuple[int, int]:
        """Return (row, col) of the next coordinate to query."""

    @abstractmethod
    def update(self, row: int, col: int, labels: np.ndarray) -> None:
        """Incorporate oracle labels (shape (8,), uint8) at (row, col)."""

    @abstractmethod
    def predict(self) -> np.ndarray:
        """Return predicted_blob_mask, shape (8, H, W), dtype uint8."""
