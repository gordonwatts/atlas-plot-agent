import logging
from disk_cache import diskcache_decorator

from atlas_plot_agent.run_in_docker import (
    DockerRunResult,
    run_python_in_docker,
)


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
