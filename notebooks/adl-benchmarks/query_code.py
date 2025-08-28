import logging
from io import TextIOWrapper
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple

from disk_cache import diskcache_decorator
from models import ModelInfo, extract_code_from_response, run_llm
from utils import IndentedDetailsBlock

from atlas_plot_agent.run_in_docker import (
    DockerRunResult,
    check_code_policies,
    run_python_in_docker,
    Policy,
)
from atlas_plot_agent.usage_info import UsageInfo


class CodeExtractablePolicy(Policy):

    def check(self, code: str) -> Optional[str]:
        try:
            extract_code_from_response(code)
            return None
        except Exception as e:
            return f"Extracting code from response failed: {str(e)}"
        return True


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
        logging.warning(
            "Retrying cached_run_python_in_docker due to httpcore.ConnectTimeout "
            f"(attempt {attempt+1}/{max_retries})"
        )
        attempt += 1

    assert result is not None
    return result


def llm_execute_loop(
    fh_out: TextIOWrapper,
    prompt_generator: Iterable[Tuple[str, List[Policy]]],
    max_iter: int,
    prompt_args: Dict[str, str],
    llm_dispatch: Callable[[str, int], Optional[str]],
    filter_code: Callable[[str], str],
    execute_code: Callable[[str, int], Tuple[bool, Dict[str, str]]],
    policy_check: Callable[[str, List[Policy]], Tuple[bool, Dict[str, str]]],
):
    # Step 1: The LLM
    # Step 2: Check policy
    # Step 3: Execute final step

    prompt_iter = iter(prompt_generator)
    base_prompt, policies = next(prompt_iter)

    # Other prompt parameters we want to track
    prompt_args_extra = {"code": ""}

    for i_iter in range(max_iter):
        with IndentedDetailsBlock(fh_out, f"Run {i_iter+1}"):

            # Call the LLM
            prompt = base_prompt.format(
                **prompt_args,
                **prompt_args_extra,
            )
            logging.debug(f"Calling LLM with (iteration {i_iter+1}): {prompt}")
            response = llm_dispatch(prompt, i_iter)
            good_run = response is not None and len(response) > 0
            if not good_run:
                logging.info("LLM call returned no output")
                continue

            assert response is not None
            prompt_args_extra["code"] = response

            # Apply policy. If the policy is violated, we need to feed that back
            # into the next iteration.
            good_run, updates = policy_check(response, policies)
            if not good_run:
                logging.info("Policy check failed")
                prompt_args_extra.update(updates)

            # Just get the code we really want to get here
            code = filter_code(response)
            prompt_args_extra["code"] = code

            # Execute any final step
            if good_run:
                good_run, updates = execute_code(code, i_iter)
                prompt_args_extra.update(updates)

            if good_run:
                break

            base_prompt, policies = next(prompt_iter, (base_prompt, policies))

    return prompt_args_extra["code"]


def check_policy(
    output: TextIOWrapper, message: str, policies: List[Policy]
) -> Tuple[bool, Dict[str, str]]:
    r = check_code_policies(message, policies)

    if isinstance(r, DockerRunResult):
        output.write(f"Policy failure: {r.stderr}\n")
        return False, {"errors": r.stderr, "output": r.stdout}

    return True, {}


def code_it_up(
    fh_out: TextIOWrapper,
    model: ModelInfo,
    prompt_write_code: str,
    prompt_fix_code: str,
    code_policies: List[Policy],
    max_iter: int,
    called_code: str,
    prompt_args: Dict[str, str],
    ignore_code_cache: bool = False,
    ignore_llm_cache: bool = False,
    llm_usage_callback: Optional[Callable[[str, UsageInfo], None]] = None,
    docker_usage_callback: Optional[Callable[[str, DockerRunResult], None]] = None,
) -> Tuple[Optional[DockerRunResult], str]:

    def prompt_and_policy() -> Generator[tuple[str, List[Policy]], Any, None]:
        yield prompt_write_code, code_policies
        yield prompt_fix_code, code_policies

    def llm_dispatcher(prompt: str, n_iter: int) -> str:
        logging.debug(f"Running against model {model.model_name}")
        usage_info, message = run_llm(
            prompt,
            model,
            fh_out,
            ignore_cache=ignore_llm_cache,
        )
        if llm_usage_callback is not None:
            llm_usage_callback(f"Run {n_iter+1}", usage_info)
        return message

    final_result: Optional[DockerRunResult] = None

    def extract_code(message: str) -> str:
        code = extract_code_from_response(message)
        assert code is not None, "Internal error - should always return code"
        return code

    def execute_code_in_docker(code: str, n_iter: int) -> Tuple[bool, Dict[str, str]]:
        # Extract the code from the data.
        code_to_run = code + "\n" + called_code + '\nprint("**Success**")\n'

        result = run_code_in_docker(code_to_run, ignore_cache=ignore_code_cache)
        if docker_usage_callback is not None:
            docker_usage_callback(f"Run {n_iter+1}", result)

        fh_out.write(f"### stdout:\n\n```text\n{result.stdout}\n```\n\n")
        fh_out.write(f"### stderr:\n\n```text\n{result.stderr}\n```\n\n")

        # To test for success, look for "**Success**" in the output.
        nonlocal final_result
        final_result = result
        return "**Success**" in result.stdout, {
            "errors": result.stderr,
            "output": result.stdout,
        }

    code = llm_execute_loop(
        fh_out,
        prompt_and_policy(),
        max_iter,
        prompt_args,
        llm_dispatcher,
        extract_code,
        execute_code_in_docker,
        lambda msg, pols: check_policy(fh_out, msg, pols),
    )

    return final_result, code


def run_llm_loop_simple(
    fh_out: TextIOWrapper,
    prompt: str,
    prompt_args: Dict[str, str],
    n_iter: int,
    model: ModelInfo,
    ignore_llm_cache: bool,
    llm_usage_callback: Optional[Callable[[str, UsageInfo], None]],
) -> str:
    def prompt_and_policy():
        yield prompt, []

    def llm_dispatcher(prompt: str, n_iter: int) -> str:
        logging.debug(f"Running against model {model.model_name}")
        usage_info, message = run_llm(
            prompt,
            model,
            fh_out,
            ignore_cache=ignore_llm_cache,
        )
        if llm_usage_callback is not None:
            llm_usage_callback(f"Run {n_iter+1}", usage_info)
        return message

    message = llm_execute_loop(
        fh_out,
        prompt_and_policy(),
        n_iter,
        prompt_args,
        llm_dispatcher,
        lambda s: s,
        lambda c, i: (True, {}),
        lambda msg, pols: check_policy(fh_out, msg, pols),
    )

    return message
