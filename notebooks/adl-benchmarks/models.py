import logging
import os
import sys
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import openai
from disk_cache import diskcache_decorator
from dotenv import dotenv_values, find_dotenv
from pydantic import BaseModel
from query_config import load_yaml_file

from atlas_plot_agent.usage_info import UsageInfo, get_usage_info


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


@diskcache_decorator(".openai_response_cache")
def _get_openai_response(prompt: str, model_name: str, endpoint: Optional[str] = None):
    import time

    if endpoint:
        client = openai.OpenAI(base_url=endpoint)
    else:
        client = openai.OpenAI()
    start_time = time.time()
    response = client.chat.completions.create(
        model=model_name, messages=[{"role": "user", "content": prompt}]
    )
    elapsed = time.time() - start_time
    assert response.choices[0].message.content is not None, "No content in response"
    # Return both response and timing for caching
    return {"response": response, "elapsed": elapsed}


def run_llm(prompt: str, model_info, ignore_cache=False) -> Tuple[UsageInfo, str]:
    # Set API key based on endpoint hostname, using <node-name>_API_KEY
    endpoint_host = None
    if model_info.endpoint:
        endpoint_host = urlparse(model_info.endpoint).hostname
    if not endpoint_host:
        endpoint_host = "api.openai.com"
    env_var = f"{endpoint_host.replace('.', '_')}_API_KEY"
    env_path = find_dotenv()
    env_vars = dotenv_values(env_path)
    api_key = env_vars.get(env_var)
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        openai.api_key = api_key
    else:
        logging.warning(f"API key not found for {env_var}")
        if "OPENAI_API_KEY" in env_vars:
            del os.environ["OPENAI_API_KEY"]

    # Do the query
    llm_result = _get_openai_response(
        prompt,
        model_info.model_name,
        model_info.endpoint,
        ignore_cache=ignore_cache,  # type: ignore
    )
    response = llm_result["response"]
    elapsed = llm_result["elapsed"]
    message = None
    if response and response.choices and response.choices[0].message:
        message = response.choices[0].message.content

    print("\n")
    if message:
        cleaned_message = message.strip()
        sys.stdout.flush()
        sys.stdout.buffer.write((cleaned_message + "\n").encode("utf-8"))
        sys.stdout.flush()
    else:
        print("No response content returned.")

    usage_info = get_usage_info(response, model_info, elapsed)

    return usage_info, str(message)
