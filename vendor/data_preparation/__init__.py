"""Data preparation module for tabular regression tasks."""

__version__ = "1.0.0"

from data_preparation.pipeline import DataPreparationPipeline
from data_preparation.config import PipelineConfig

__all__ = ["DataPreparationPipeline", "PipelineConfig"]
