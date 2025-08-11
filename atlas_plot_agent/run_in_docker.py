import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import yaml


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

    # Find PNG files in temp_dir and load them into memory
    png_files = []
    for p in Path(temp_dir).glob("*.png"):
        with open(p, "rb") as f:
            png_files.append((p.name, f.read()))

    # Clean up temp_dir if desired (optional)
    # shutil.rmtree(temp_dir)

    return DockerRunResult(
        stdout=stdout, stderr=stderr, elapsed=elapsed, png_files=png_files
    )
