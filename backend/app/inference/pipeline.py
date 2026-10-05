"""Compatibility import for older code paths.

The real implementation lives in app.pipeline so there is only one inference pipeline.
"""
from app.pipeline import FoodPipeline

pipeline = FoodPipeline()

__all__ = ["FoodPipeline", "pipeline"]
