import typer


from pydantic import BaseModel
import yaml
import os
import logging

import fsspec
from cachetools import cached
from cachetools import TTLCache


class DirectQueryConfig(BaseModel):
    """
    Configuration model for direct-query CLI.
    Contains a list of hint files and a prompt string.
    """

    hint_files: list[str]
    prompt: str


def load_config(config_path: str = "direct-query-config.yaml") -> DirectQueryConfig:
    """
    Load configuration from a YAML file and return a DirectQueryConfig instance.
    Checks local directory first, then script directory, and logs which file is loaded.
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
    Loads config and prints the question and config for demonstration.
    """
    config = load_config()
    hint_contents = load_hint_files(config.hint_files)

    # Build the prompt
    prompt = config.prompt.format(question=question, hints="\n".join(hint_contents))
    print(f"Built prompt: {prompt}")


if __name__ == "__main__":
    app()
