from dataclasses import dataclass
from typing import Optional


@dataclass
class dataset_reference:
    "A single dataset"

    # The full rucio dataset name
    name: str


@dataclass
class conversation_context:
    "Context used to pass between tools"

    # The dataset we will pull data from
    ds: dataset_reference

    # What we want to plot that is in the dataset.
    what_to_plot: str


def dataclass_type_from_string(type_str: Optional[str]):
    "Convert a string to a dataclass type"
    if type_str is None:
        return None
    if type_str == "dataset_reference":
        return dataset_reference
    elif type_str == "conversation_context":
        return conversation_context
    else:
        raise ValueError(f"Unknown type: {type_str}")
