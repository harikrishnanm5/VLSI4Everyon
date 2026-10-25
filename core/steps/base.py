"""Base interface for semiconductor process steps."""

from abc import ABC, abstractmethod


class ProcessStep(ABC):
    """Abstract base class representing a single semiconductor fabrication step."""

    @abstractmethod
    def apply(self, state):
        """Apply the process step to the current wafer state."""
        pass
