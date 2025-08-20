from typing import Dict, List, Optional

from pydantic import BaseModel
from query_config import load_yaml_file


class ModelInfo(BaseModel):
    model_name: str
    input_cost_per_million: float
    output_cost_per_million: float
    endpoint: Optional[str] = None  # e.g., OpenAI API endpoint or local server URL


def load_models(models_path: str = "models.yaml") -> Dict[str, ModelInfo]:
    """
    Load models and their costs from a YAML file, returning a dict of model_name to ModelInfo.
    """
    data = load_yaml_file(models_path)
    raw_models = data["models"]
    return {name: ModelInfo(**info) for name, info in raw_models.items()}


def process_model_request(
    requested_models: Optional[str],
    all_models: Dict[str, ModelInfo],
    default_model_name: str,
) -> List[str]:
    """
    Processes the requested model names and returns a validated list of model names.
    Built to be used with a command line option.

    Args:
        requested_models (Optional[str]): Comma-separated string of requested model names, or None.
        all_models (Dict[str, ModelInfo]): Dictionary of available models.
        default_model_name (str): The default model name to use if none are requested.

    Returns:
        List[str]: List of validated model names.

    Raises:
        ValueError: If any requested model name is not found in the available models.
    """
    if requested_models:
        model_names = [m.strip() for m in requested_models.split(",") if m.strip()]
        if "all" in model_names:
            model_names = list(all_models.keys())
    else:
        model_names = [default_model_name]

    # Validate model names
    invalid_model_names = [m for m in model_names if m not in all_models]
    if invalid_model_names:
        raise ValueError(
            f"Error: model(s) not found in models.yaml: {', '.join(invalid_model_names)}"
        )

    return model_names
