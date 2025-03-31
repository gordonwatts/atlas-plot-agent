from dataclasses import dataclass


@dataclass
class dataset_reference:
    "A single dataset"

    # The full rucio (for now) dataset name
    name: str


@dataclass
class conversation_context:
    "Context used to pass between tools"

    ds: dataset_reference
