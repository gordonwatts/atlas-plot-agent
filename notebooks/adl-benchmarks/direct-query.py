import functools
import logging
import os
from typing import Optional
import sys

import fsspec
import openai
import typer
import yaml
from diskcache import Cache
from dotenv import load_dotenv
from pydantic import BaseModel


def load_yaml_file(filename: str) -> dict:
    """
    Load a YAML file from local or script directory and return its contents as a dict.
    """
    local_path = os.path.abspath(filename)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, filename)

    if os.path.exists(local_path):
        logging.info(f"Loaded {filename} from local directory: {local_path}")
        path_to_load = local_path
    elif os.path.exists(script_path):
        logging.info(f"Loaded {filename} from script directory: {script_path}")
        path_to_load = script_path
    else:
        raise FileNotFoundError(
            f"File not found in local or script directory: {filename}"
        )

    with open(path_to_load, "r") as f:
        return yaml.safe_load(f)


class ModelInfo(BaseModel):
    model_name: str
    input_cost_per_million: float
    output_cost_per_million: float
    endpoint: Optional[str] = None  # e.g., OpenAI API endpoint or local server URL


def load_models(models_path: str = "models.yaml") -> dict:
    """
    Load models and their costs from a YAML file, returning a dict of model_name to ModelInfo.
    """
    data = load_yaml_file(models_path)
    raw_models = data["models"]
    return {name: ModelInfo(**info) for name, info in raw_models.items()}


class DirectQueryConfig(BaseModel):
    """
    Configuration model for direct-query CLI.
    Contains a list of hint files, a prompt string, and a model name.
    """

    hint_files: list[str]
    prompt: str
    model_name: str = "gpt-4-1106-preview"


def load_config(config_path: str = "direct-query-config.yaml") -> DirectQueryConfig:
    """
    Load configuration from a YAML file and return a DirectQueryConfig instance.
    Sets default model_name to gpt-4-1106-preview if not present.
    """
    data = load_yaml_file(config_path)
    return DirectQueryConfig(**data)


# Generic diskcache-backed decorator
def diskcache_decorator(cache: Cache):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ignore_cache = kwargs.pop("ignore_cache", False)
            key = (func.__name__, args, tuple(sorted(kwargs.items())))
            if not ignore_cache and key in cache:
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            return result

        return wrapper

    return decorator


file_cache = Cache(".hint_file_cache")


@diskcache_decorator(file_cache)
def load_file_content(path: str) -> str:
    """
    Load file content using built-in open for local files, with disk-backed cache.
    For remote filesystems, replace with fsspec.open or fsspec.open_files.
    """
    with fsspec.open(path, "r") as f:
        content = f.read()  # type: ignore
    return content


def load_hint_files(hint_files: list[str]) -> list[str]:
    """
    Load all hint files into a list of strings, using cache for speed.
    """
    return [load_file_content(hint_file) for hint_file in hint_files]


response_cache = Cache(".openai_response_cache")


@diskcache_decorator(response_cache)
def get_openai_response(prompt: str, model_name: str, endpoint: Optional[str] = None):
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


app = typer.Typer(
    help=(
        "use default configuration to ask the api a question and "
        "generate code in response"
    )
)


def run_model(question: str, prompt: str, model_info, ignore_cache=False):
    """
    Run the model, print heading and result, and return info for the table.
    """
    # Get cached response and timing
    result = get_openai_response(
        prompt,
        model_info.model_name,
        model_info.endpoint,
        ignore_cache=ignore_cache,
    )
    response = result["response"]
    elapsed = result["elapsed"]
    message = None
    if response and response.choices and response.choices[0].message:
        message = response.choices[0].message.content

    print("\n")
    print(f"## Model: {model_info.model_name}\n")
    if message:
        cleaned_message = (
            message.replace(">>start-reply<<", "").replace(">>end-reply<<", "").strip()
        )
        sys.stdout.flush()
        sys.stdout.buffer.write((cleaned_message + "\n").encode("utf-8"))
        sys.stdout.flush()
    else:
        print("No response content returned from OpenAI.")

    usage = getattr(response, "usage", None)
    if usage:
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)
        cost = (prompt_tokens / 1_000_000) * model_info.input_cost_per_million + (
            completion_tokens / 1_000_000
        ) * model_info.output_cost_per_million
        return {
            "model": model_info.model_name,
            "elapsed": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
        }
    else:
        return {
            "model": model_info.model_name,
            "elapsed": elapsed,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cost": None,
        }


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask the API"),
    models: str = typer.Option(
        None,
        help="Comma-separated list of model names to run (default: pulled from config). "
        "Use `all` to run all known models.",
    ),
    ignore_cache: bool = typer.Option(
        False, "--ignore-cache", help="Ignore disk cache for model queries."
    ),
):
    """
    Command to ask a question using the default configuration.
    Runs the question against one or more models, prints results, and prints a summary table.
    """
    # Load environment variables from .env
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logging.warning("OPENAI_API_KEY not found in environment.")
        return
    openai.api_key = openai_api_key

    # Load configuration
    config = load_config()
    hint_contents = load_hint_files(config.hint_files)

    # Load models
    all_models = load_models()
    if models:
        model_names = [m.strip() for m in models.split(",") if m.strip()]
        if "all" in model_names:
            model_names = list(all_models.keys())
    else:
        model_names = [config.model_name]

    # Validate model names
    valid_model_names = [m for m in model_names if m in all_models]
    invalid_model_names = [m for m in model_names if m not in all_models]
    if invalid_model_names:
        print(
            f"Error: model(s) not found in models.yaml: {', '.join(invalid_model_names)}"
        )
        return

    # Build the prompt
    prompt = config.prompt.format(question=question, hints="\n".join(hint_contents))
    logging.info(f"Built prompt: {prompt}")

    print(f"# {question}\n")
    table_rows = []
    for model_name in valid_model_names:
        model_info = all_models[model_name]
        row = run_model(question, prompt, model_info, ignore_cache=ignore_cache)
        table_rows.append(row)

    # Print markdown table
    print("## Summary")
    print(
        "| Model | Time (s) | Prompt Tokens | Completion Tokens | Total Tokens | "
        "Estimated Cost ($) |"
    )
    print(
        "|-------|----------|--------------|------------------|--------------|"
        "--------------------|"
    )
    for row in table_rows:
        model = row["model"]
        elapsed = f"{row['elapsed']:.2f}"
        prompt_tokens = (
            row["prompt_tokens"] if row["prompt_tokens"] is not None else "-"
        )
        completion_tokens = (
            row["completion_tokens"] if row["completion_tokens"] is not None else "-"
        )
        total_tokens = row["total_tokens"] if row["total_tokens"] is not None else "-"
        cost = f"{row['cost']:.4f}" if row["cost"] is not None else "-"
        print(
            f"| {model} | {elapsed} | {prompt_tokens} | {completion_tokens} | {total_tokens}"
            f" | {cost} |"
        )


if __name__ == "__main__":
    app()
