"""
This is a boilerplate pipeline 'training_pipeline'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, pipeline,node
from .nodes import data_cleaning


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([node(
        func=data_cleaning,
        inputs="raw_data",
        outputs=None
    )])
