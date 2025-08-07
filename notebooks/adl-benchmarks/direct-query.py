import logging
import os

import fsspec
import openai
import typer
import yaml
from cachetools import TTLCache, cached
from dotenv import load_dotenv
from pydantic import BaseModel


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
    Checks local directory first, then script directory, and logs which file is loaded.
    Sets default model_name to gpt-4-1106-preview if not present.
    """
    local_path = os.path.abspath(config_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, config_path)

    if os.path.exists(local_path):
        logging.info(f"Loaded config from local directory: {local_path}")
        path_to_load = local_path
    elif os.path.exists(script_path):
        logging.info(f"Loaded config from script directory: {script_path}")
        path_to_load = script_path
    else:
        raise FileNotFoundError(
            f"Config file not found in local or script directory: {config_path}"
        )

    with open(path_to_load, "r") as f:
        data = yaml.safe_load(f)
    if "model_name" not in data:
        data["model_name"] = "gpt-4-1106-preview"
    return DirectQueryConfig(**data)


# Disk-backed cache for file contents (TTLCache for demonstration, can be replaced with diskcache)


file_cache = TTLCache(maxsize=128, ttl=3600)


@cached(file_cache)
def load_file_content(path: str) -> str:
    """
    Load file content using built-in open for local files.
    For remote filesystems, replace with fsspec.open or fsspec.open_files.
    """
    with fsspec.open(path, "r") as f:
        return f.read()  # type: ignore


def load_hint_files(hint_files: list[str]) -> list[str]:
    """
    Load all hint files into a list of strings, using cache for speed.
    """
    return [load_file_content(hint_file) for hint_file in hint_files]


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

    # Build the prompt
    prompt = config.prompt.format(question=question, hints="\n".join(hint_contents))
    logging.info(f"Built prompt: {prompt}")

    # Cache for OpenAI responses (prompt -> response)
    response_cache = TTLCache(maxsize=128, ttl=3600)

    @cached(response_cache)
    def get_openai_response(prompt: str) -> str:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=config.model_name, messages=[{"role": "user", "content": prompt}]
        )
        assert response.choices[0].message.content is not None, "No content in response"
        return response.choices[0].message.content.strip()

    result = get_openai_response(prompt)
    print(f"OpenAI response:\n{result}")


if __name__ == "__main__":
    app()
