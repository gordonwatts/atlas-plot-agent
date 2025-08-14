import functools
import hashlib
import logging
import os
import sys
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import fsspec
import openai
import typer
import yaml
from diskcache import Cache
from dotenv import dotenv_values, find_dotenv
from pydantic import BaseModel

from atlas_plot_agent.run_in_docker import (
    DockerRunResult,
    check_code_policies,
    run_python_in_docker,
)
from atlas_plot_agent.usage_info import UsageInfo, get_usage_info, sum_usage_infos

if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")  # type: ignore
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore


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


def load_models(models_path: str = "models.yaml") -> Dict[str, ModelInfo]:
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
    modify_prompt: str
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


# Diskcache for docker results
docker_cache = Cache(".docker_run_cache")


@diskcache_decorator(docker_cache)
def cached_run_python_in_docker(code: str, ignore_cache=False):
    "Caching version"
    return run_python_in_docker(code)


def extract_code_from_response(response) -> Optional[str]:
    """
    Extract Python code from an OpenAI response object.
    Looks for code blocks in the message content and returns the first Python block
    found.
    """
    if not response or not hasattr(response, "choices") or not response.choices:
        return None
    message = response.choices[0].message.content if response.choices[0].message else ""
    if not message:
        return None
    import re

    # Find all Python code blocks
    code_blocks = re.findall(r"```python(.*?)```", message, re.DOTALL | re.IGNORECASE)
    if code_blocks:
        return code_blocks[0].strip()
    # Fallback: any code block
    code_blocks = re.findall(r"```(.*?)```", message, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    return None


app = typer.Typer(
    help=(
        "use default configuration to ask the api a question and "
        "generate code in response"
    )
)


def run_model(
    prompt: str, model_info, png_prefix: str, ignore_cache=False
) -> Tuple[UsageInfo, bool, Optional[DockerRunResult], Optional[str]]:
    """
    Run the model, print heading and result, and return info for the table.
    Runs the code once and returns:
        - UsageInfo for the run
        - True/False if the run succeeded
        - DockerRunResult
        - The code
    """
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
    llm_result = get_openai_response(
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
    print(f"## Model: {model_info.model_name}\n")
    if message:
        cleaned_message = (
            message.replace(">>start-reply<<", "").replace(">>end-reply<<", "").strip()
        )
        sys.stdout.flush()
        sys.stdout.buffer.write((cleaned_message + "\n").encode("utf-8"))
        sys.stdout.flush()
    else:
        print("No response content returned.")

    usage_info = get_usage_info(response, model_info, elapsed)

    # Run the code.
    print("### Running\n")
    code = extract_code_from_response(response)
    run_result = False
    result: Optional[DockerRunResult] = None
    if code is not None:
        r = check_code_policies(code)
        if r is True:
            # Run code in Docker and capture output and files, using cache
            # If we get timeouts, keep trying...
            # TODO: We should be using a retry library, not this!
            max_retries = 3
            attempt = 0
            result = None
            while attempt < max_retries:
                # For first attempt, use original ignore_cache; for retries,
                # force ignore_cache=True
                use_ignore_cache = ignore_cache if attempt == 0 else True
                result = cached_run_python_in_docker(
                    code, ignore_cache=use_ignore_cache
                )
                # If no ConnectTimeout, break
                has_timeout = "httpcore.ConnectTimeout" in str(result.stderr)
                if not has_timeout:
                    break
                attempt += 1
                logging.warning(
                    "Retrying cached_run_python_in_docker due to httpcore.ConnectTimeout "
                    f"(attempt {attempt+1}/{max_retries})"
                )
        else:
            assert isinstance(r, DockerRunResult)
            result = r

        assert isinstance(result, DockerRunResult)
        print(f"*Output:*\n```\n{result.stdout}\n```")
        print(f"*Error:*\n```\n{result.stderr}\n```")

        # Did we run without an error?
        run_result = result.exit_code == 0

        # Save PNG files locally, prefixed with model name
        for f_name, data in result.png_files:
            # Sanitize model_name for filesystem
            safe_model_name = model_info.model_name.replace("/", "_")
            local_name = f"{png_prefix}_{safe_model_name}_{f_name}"
            with open(local_name, "wb") as dst:
                dst.write(data)
            print(f"![{local_name}]({local_name})")  # Markdown image include

    else:
        print("No code found to run.")

    return usage_info, run_result, result, code


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
    n_iter: int = typer.Option(
        1, "--n-iter", "-n", min=1, help="Number of iterations to run (must be >= 1)."
    ),
):
    """
    Command to ask a question using the default configuration.
    Runs the question against one or more models, prints results, and prints a summary table.
    """

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

    if n_iter < 1:
        logging.error(
            f"Error: command line option `n_iter` must be >= 1 (got {n_iter})"
        )
        return

    print(f"# {question}\n")
    table_rows = []
    question_hash = hashlib.sha1(question.encode("utf-8")).hexdigest()[:8]
    code = None
    errors = None
    for model_name in valid_model_names:
        run_info = []
        for iter in range(n_iter):
            # Build the prompt
            base_prompt = config.prompt if iter == 0 else config.modify_prompt
            prompt = base_prompt.format(
                question=question,
                hints="\n".join(hint_contents),
                error=errors,
                old_code=code,
            )
            logging.info(f"Built prompt for iteration {iter}: {prompt}")

            model_info = all_models[model_name]
            row = run_model(
                prompt, model_info, question_hash, ignore_cache=ignore_cache
            )
            run_info.append(row[0:2])

            # If things worked, then we don't need to go again!
            if row[1]:
                break

            # Build up prompt info for next time.
            if row[2] is None or row[3] is None:
                break
            code = row[3]
            errors = "\n".join([row[2].stdout, row[2].stderr])

        total_usage = sum_usage_infos([u for u, _ in run_info])
        attempt_results = [r for _, r in run_info]
        table_rows.append([total_usage, attempt_results])

    # Print markdown table
    print("## Summary\n")
    # Determine max number of python run attempts
    max_attempts = max(len(attempts) for _, attempts in table_rows) if table_rows else 0
    # Build header
    base_header = (
        "| Model(s) | Time (s) | Prompt Tokens | Completion Tokens | Total Tokens "
        "| Estimated Cost ($) |"
    )
    attempt_headers = "".join([" Python Run |"] * max_attempts) if max_attempts else ""
    print(base_header + attempt_headers)
    print(
        "|-------|----------|--------------|------------------|--------------"
        "|--------------------|"
        + "".join(["--------------|" for _ in range(max_attempts)])
    )
    for row in table_rows:
        usage_info, run_result = row
        model = usage_info.model
        elapsed = f"{usage_info.elapsed:.2f}"
        prompt_tokens = (
            usage_info.prompt_tokens if usage_info.prompt_tokens is not None else "-"
        )
        completion_tokens = (
            usage_info.completion_tokens
            if usage_info.completion_tokens is not None
            else "-"
        )
        total_tokens = (
            usage_info.total_tokens if usage_info.total_tokens is not None else "-"
        )
        cost = f"${usage_info.cost:.3f}" if usage_info.cost is not None else "-"
        run_cell = "".join(" Success |" if r else " Fail |" for r in run_result)
        print(
            f"| {model} | {elapsed} | {prompt_tokens} | {completion_tokens} | {total_tokens} "
            f"| {cost} |" + run_cell
        )


if __name__ == "__main__":
    app()
