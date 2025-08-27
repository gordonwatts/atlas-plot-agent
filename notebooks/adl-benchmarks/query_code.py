import logging
from io import TextIOWrapper
from typing import Callable, Dict, Optional, Tuple

from disk_cache import diskcache_decorator
from models import ModelInfo, extract_code_from_response, run_llm
from utils import IndentedDetailsBlock

from atlas_plot_agent.run_in_docker import DockerRunResult, run_python_in_docker
from atlas_plot_agent.usage_info import UsageInfo


@diskcache_decorator(".docker_run_cache")
def cached_run_python_in_docker(code: str, ignore_cache=False):
    "Caching version"
    return run_python_in_docker(code)


def run_code_in_docker(code: str, ignore_cache: bool = False) -> DockerRunResult:

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
        result = cached_run_python_in_docker(code, ignore_cache=use_ignore_cache)
        # If no ConnectTimeout, break
        has_timeout = "httpcore.ConnectTimeout" in str(result.stderr)
        if not has_timeout:
            break
        attempt += 1
        logging.warning(
            "Retrying cached_run_python_in_docker due to httpcore.ConnectTimeout "
            f"(attempt {attempt+1}/{max_retries})"
        )

    assert result is not None
    return result


def code_it_up(
    fh_out: TextIOWrapper,
    model: ModelInfo,
    prompt_write_code: str,
    prompt_fix_code: str,
    max_iter: int,
    called_code: str,
    prompt_args: Dict[str, str],
    ignore_code_cache: bool = False,
    ignore_llm_cache: bool = False,
    llm_usage_callback: Optional[Callable[[str, UsageInfo], None]] = None,
    docker_usage_callback: Optional[Callable[[str, DockerRunResult], None]] = None,
) -> Tuple[DockerRunResult, str]:

    # Build code with initial prompt
    base_prompt = prompt_write_code

    error_text = ""
    output_text = ""
    generated_code = ""

    # Run iteration times, attempting to fix code if it doesn't work.
    for n_iter in range(max_iter):
        # Fill in prompt arguments
        extra_args = {
            "errors": error_text,
            "output": output_text,
            "code": generated_code,
        }
        prompt = base_prompt.format(**prompt_args, **extra_args)
        logging.debug(f"Built prompt to generate code: {prompt}")

        # Run against model
        with IndentedDetailsBlock(fh_out, f"Run {n_iter+1}"):
            logging.debug(f"Running against model {model.model_name}")
            usage_info, message = run_llm(
                prompt,
                model,
                fh_out,
                ignore_cache=ignore_llm_cache,
            )
            if llm_usage_callback is not None:
                llm_usage_callback(f"Run {n_iter+1}", usage_info)

            # Now run the code to fetch the data
            code = extract_code_from_response(message)
            assert code is not None, "Can't work with null code for now"

            code_to_run = code + "\n" + called_code + '\nprint("**Success**")\n'

            result = run_code_in_docker(code_to_run, ignore_cache=ignore_code_cache)
            if docker_usage_callback is not None:
                docker_usage_callback(f"Run {n_iter+1}", result)

            fh_out.write(f"### stdout:\n\n```text\n{result.stdout}\n```\n\n")
            fh_out.write(f"### stderr:\n\n```text\n{result.stderr}\n```\n\n")

            # To test for success, look for "**Success**" in the output.
            good_run = "**Success**" in result.stdout

        if good_run:
            break

        base_prompt = prompt_fix_code
        error_text = result.stderr
        output_text = result.stdout
        generated_code = code

    assert isinstance(result, DockerRunResult)
    assert isinstance(code, str)
    return result, code
