from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs=["preprocessed_data"],
            outputs="split_output",
            name="split_data_node",
        ),
        node(
            func=train_model,
            inputs="split_output",
            outputs="model",
            name="train_model_node",
        ),
    ])
