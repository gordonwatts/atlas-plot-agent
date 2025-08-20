import hashlib
import logging
import os
import sys
from typing import Optional, Tuple
from urllib.parse import urlparse

import openai
import typer
import yaml
from disk_cache import diskcache_decorator
from dotenv import dotenv_values, find_dotenv
from hint_files import load_hint_files
from query_config import load_config
from models import load_models, process_model_request

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


@diskcache_decorator(".openai_response_cache")
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


@diskcache_decorator(".docker_run_cache")
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
    message = ensure_closing_triple_backtick(message)

    if not message:
        return None
    import re

    # Find all Python code blocks
    code_blocks = re.findall(r"```python(.*?)```", message, re.DOTALL | re.IGNORECASE)
    if code_blocks:
        return code_blocks[-1].strip()
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
        # Ensure closing triple backtick if opening exists but not closed
        message = ensure_closing_triple_backtick(message)

    print("\n")
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
    print("#### Code Execution\n")
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
        run_result = result.exit_code == 0 and len(result.png_files) > 0

        # Save PNG files locally, prefixed with model name
        if run_result:
            print("</details>\n")
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
    error_info: bool = typer.Option(
        False,
        "--write-error-info",
        help="Writes a small `fail` file for each error that can be analyzed later",
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
    valid_model_names = process_model_request(models, all_models, config.model_name)

    # Check number of requested iterations is good
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
        print(f"\n## Model {all_models[model_name].model_name}")
        run_info = []
        for iter in range(n_iter):
            print(
                f"<details><summary>Run {iter+1} Details</summary>\n\n### Run {iter+1}"
            )
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
            print("</details>")

            # It is also possible there was a catastrophic failure, and we shouldn't
            # even try again.
            if row[2] is None or row[3] is None:
                break

            # If we are writing out the error info, we should do that here
            code = row[3]
            if error_info:
                e_info = {
                    "code": code,
                    "question": question,
                    "stdout": row[2].stdout,
                    "stderr": row[2].stderr,
                    "model": model_name,
                    "iteration": iter + 1,
                }
                output_file = f"z_fail_{question_hash}_{model_name}_{iter+1}.yaml"
                with open(output_file, "w") as f:
                    yaml.dump(e_info, f)

            # Build up prompt info for next time.
            errors = "\n".join([row[2].stdout, row[2].stderr])

        total_usage = sum_usage_infos([u for u, _ in run_info])
        attempt_results = [r for _, r in run_info]
        table_rows.append([total_usage, attempt_results])

    # CSV section
    print("\n## CSV\n")
    # CSV header
    # Determine max number of python run attempts
    csv_header = [
        "Model",
        "Time",
        "PromptTokens",
        "CompletionTokens",
        "TotalTokens",
        "EstimatedCost",
        "Attempts",
        "Result",
    ]
    print(",".join(csv_header))
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
        cost = f"{usage_info.cost:.3f}" if usage_info.cost is not None else "-"
        attempts = len(run_result)
        result = "Success" if any(run_result) else "Fail"
        csv_row = [
            str(model),
            str(elapsed),
            str(prompt_tokens),
            str(completion_tokens),
            str(total_tokens),
            str(cost),
            str(attempts),
            result,
        ]
        print(",".join(csv_row))

    # Markdown summary section
    print("\n## Summary\n")
    # Build header
    print(
        "| Model(s) | Time (s) | Prompt Tokens | Completion Tokens | Total Tokens | "
        "Estimated Cost ($) | Attempts | Result |"
    )
    print(
        "|-------|----------|--------------|------------------|--------------|"
        "--------------------|----------|--------|"
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
        attempts = len(run_result)
        result = "Success" if any(run_result) else "Fail"
        print(
            f"| {model} | {elapsed} | {prompt_tokens} | {completion_tokens} | "
            f"{total_tokens} | {cost} | {attempts} | {result} |"
        )


def ensure_closing_triple_backtick(message: str) -> str:
    """
    Ensure that if a message contains an opening triple backtick, it also has a closing one.
    If the number of triple backticks is odd, append a closing triple backtick.
    """
    if "```" in message:
        backtick_count = message.count("```")
        if backtick_count % 2 != 0:
            message = message + "\n```"
    return message


if __name__ == "__main__":
    app()
