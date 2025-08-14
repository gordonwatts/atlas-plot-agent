import os
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import yaml


class Policy(ABC):
    """
    Abstract base class for code policies.
    """

    @abstractmethod
    def check(self, python_code: str) -> str | None:
        """
        Returns a string describing the violation, or None if no violation.
        """
        pass


class PltSavefigPolicy(Policy):
    """
    Policy that checks for a plt.savefig call in the source code.
    """

    def check(self, python_code: str) -> str | None:
        code_no_comments_no_strings = remove_comments_and_strings(python_code)
        if "plt.savefig" not in code_no_comments_no_strings:
            return (
                "plt.savefig not found in source code - "
                "save your plot to a file using plt.savefig()."
            )
        return None


class NFilesPolicy(Policy):
    """
    Policy that checks for 'NFiles=1' outside of comments and strings.
    """

    def check(self, python_code: str) -> str | None:
        code_no_comments_no_strings = remove_comments_and_strings(python_code)
        if "NFiles=1" not in code_no_comments_no_strings:
            return (
                "NFiles=1 not found in source code - it must be present in the ServiceX "
                "`Sample` definition to assure a quick test run."
            )
        return None


# Global list of policies
POLICIES: list[Any] = [NFilesPolicy(), PltSavefigPolicy()]


def copy_servicex_yaml_if_exists(target_dir: str):
    """
    Copies servicex.yaml from the user's home directory to target_dir if it exists.
    """

    home_servicex = os.path.expanduser("~/servicex.yaml")
    target_servicex = os.path.join(target_dir, "servicex.yaml")
    if os.path.exists(home_servicex):
        with open(home_servicex, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        # If loaded is not a dict, make it a dict
        if not isinstance(loaded, dict):
            loaded = {}  # treat as empty config
        loaded["cache_path"] = "/cache"
        with open(target_servicex, "w", encoding="utf-8") as f:
            yaml.safe_dump(loaded, f)


@dataclass
class DockerRunResult:
    """
    Contains the output and results of running a python script in docker.
    png_files: list of (filename, file bytes)
    """

    stdout: str
    stderr: str
    elapsed: float
    png_files: List[Tuple[str, bytes]]
    exit_code: int


def run_python_in_docker(python_code: str) -> DockerRunResult:
    """
    Runs the given python_code in a Docker container, captures stdout/stderr, elapsed time,
    and PNG outputs.

    Returns a DockerRunResult dataclass.
    """
    temp_dir = tempfile.mkdtemp()
    script_path = os.path.join(temp_dir, "script.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(python_code)

    # Copy servicex.yaml from home directory if it exists
    copy_servicex_yaml_if_exists(temp_dir)

    # Run the docker container
    docker_image = "atlasplotagent:latest"
    container_dir = "/app"
    # Mount a docker volume at /cache
    cache_volume = "atlasplotagent_servicex_cache"
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{temp_dir}:{container_dir}",
        "-v",
        f"{cache_volume}:/cache",
        docker_image,
        "bash",
        "-i",  # so it executes the `.bashrc` file which defines the venv
        "-c",
        f"python {container_dir}/script.py",
    ]

    start = time.time()
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    stdout, stderr = proc.communicate()
    elapsed = time.time() - start
    exit_code = proc.returncode

    # Find PNG files in temp_dir and load them into memory
    png_files = []
    for p in Path(temp_dir).glob("*.png"):
        with open(p, "rb") as f:
            png_files.append((p.name, f.read()))

    # Clean up temp_dir if desired (optional)
    # shutil.rmtree(temp_dir)

    return DockerRunResult(
        stdout=stdout,
        stderr=stderr,
        elapsed=elapsed,
        png_files=png_files,
        exit_code=exit_code,
    )


def remove_comments_and_strings(python_code: str) -> str:
    """
    Utility function to remove comments and strings from Python code.

    Args:
        python_code: The Python source code as a string

    Returns:
        The code with comments and strings removed
    """
    import re

    # Remove comments and empty lines
    code_lines = []
    for line in python_code.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        # Remove trailing inline comments
        if "#" in line:
            line = line.split("#", 1)[0]
        code_lines.append(line)
    code_no_comments = "\n".join(code_lines)

    # Remove strings
    def remove_strings(s):
        s = re.sub(r"'''[\s\S]*?'''", "", s)
        s = re.sub(r'"""[\s\S]*?"""', "", s)
        s = re.sub(r"'(?:\\.|[^'])*'", "", s)
        s = re.sub(r'"(?:\\.|[^"])*"', "", s)
        return s

    return remove_strings(code_no_comments)


def check_code_policies(python_code: str) -> bool | DockerRunResult:
    """
    Check all policies in the global POLICIES list.
    Returns True if all pass, otherwise returns a DockerRunResult
    with a markdown list of violations.
    """
    violations = []
    for policy in POLICIES:
        violation = policy.check(python_code)
        if violation:
            violations.append(violation)
    if violations:
        markdown = "\n".join([f"- {v}" for v in violations])
        return DockerRunResult(
            stdout="",
            stderr=f"Policy violations found:\n{markdown}",
            elapsed=0,
            png_files=[],
            exit_code=1,
        )
    return True
