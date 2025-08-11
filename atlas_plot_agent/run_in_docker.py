from dataclasses import dataclass
from typing import List, Tuple
import tempfile
import shutil
import subprocess
import time
import os
from pathlib import Path


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
    home_servicex = os.path.expanduser("~/servicex.yaml")
    if os.path.exists(home_servicex):
        shutil.copy(home_servicex, os.path.join(temp_dir, "servicex.yaml"))

    # Run the docker container
    docker_image = "atlasplotagent:latest"
    container_dir = "/app"
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{temp_dir}:{container_dir}",
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
