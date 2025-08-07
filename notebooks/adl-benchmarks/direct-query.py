import logging
import os
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
    input_cost_per_million: float
    output_cost_per_million: float


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
import functools


def diskcache_decorator(cache: Cache):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (func.__name__, args, tuple(sorted(kwargs.items())))
            if key in cache:
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
def get_openai_response(prompt: str, model_name: str):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model_name, messages=[{"role": "user", "content": prompt}]
    )
    assert response.choices[0].message.content is not None, "No content in response"
    return response


app = typer.Typer(
    help=(
        "use default configuration to ask the api a question and "
        "generate code in response"
    )
)


@app.command()
def ask(question: str = typer.Argument(..., help="The question to ask the API")):
    """
    Command to ask a question using the default configuration.
    Loads config, loads .env for OpenAI API key, builds prompt, queries OpenAI, and prints result.
    Uses cache for prompt responses.
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

    # Load models and validate model_name
    models = load_models()
    if config.model_name not in models:
        print(f"Error: model_name '{config.model_name}' not found in models.yaml.")
        return
    model_info = models[config.model_name]
    input_cost = model_info.input_cost_per_million
    output_cost = model_info.output_cost_per_million

    # Build the prompt
    prompt = config.prompt.format(question=question, hints="\n".join(hint_contents))
    logging.info(f"Built prompt: {prompt}")

    # Disk-backed cache for OpenAI responses
    response = get_openai_response(prompt, config.model_name)
    message = None
    if response and response.choices and response.choices[0].message:  # type: ignore
        message = response.choices[0].message.content  # type: ignore
    if message:
        print(f"OpenAI response:\n{message.strip()}")
    else:
        print("No response content returned from OpenAI.")

    # Print token usage and cost
    usage = getattr(response, "usage", None)
    if usage:
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)
        print(
            f"Token usage: prompt={prompt_tokens}, "
            f"completion={completion_tokens}, total={total_tokens}"
        )
        cost = (prompt_tokens / 1_000_000) * input_cost + (
            completion_tokens / 1_000_000
        ) * output_cost
        print(f"Estimated cost: ${cost:.4f}")
    else:
        print("Token usage information not available.")


if __name__ == "__main__":
    app()
