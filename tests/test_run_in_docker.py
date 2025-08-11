from atlas_plot_agent.run_in_docker import run_python_in_docker, DockerRunResult


def test_run_python_in_docker_success():
    code = """
print("Hello from docker!")
"""
    result = run_python_in_docker(code)
    assert isinstance(result, DockerRunResult)
    assert "Hello from docker!" in result.stdout
    assert result.stderr == "" or "Traceback" not in result.stderr
    assert result.elapsed > 0


def test_run_python_in_docker_failure():
    code = """
raise Exception('This should fail')
"""
    result = run_python_in_docker(code)
    assert isinstance(result, DockerRunResult)
    assert result.stdout == "" or "Hello" not in result.stdout
    assert "Exception" in result.stderr or "Traceback" in result.stderr
    assert result.elapsed > 0


def test_run_python_in_docker_awkward():
    code = """
import awkward as ak
"""
    result = run_python_in_docker(code)
    assert isinstance(result, DockerRunResult)
    assert result.stderr == "" or "Traceback" not in result.stderr
    assert result.elapsed > 0


def test_run_python_in_docker_png_creation():
    code = """
import matplotlib.pyplot as plt
plt.plot([1,2,3],[4,5,6])
plt.savefig('output.png')
"""
    result = run_python_in_docker(code)
    assert isinstance(result, DockerRunResult)
    pngs = [f for f, data in result.png_files if f == "output.png"]
    assert len(pngs) == 1
    # Check that the file bytes start with PNG header
    png_bytes = [data for fname, data in result.png_files if fname == "output.png"][0]
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_run_python_in_docker_servicex_yaml_present():
    code = """
import os
assert os.path.exists('servicex.yaml'), 'servicex.yaml not found in working directory'
"""
    result = run_python_in_docker(code)
    assert isinstance(result, DockerRunResult)

    assert result.stderr == "" or "Traceback" not in result.stderr
    assert result.elapsed > 0


def test_run_python_in_docker_cache_persistence():
    # First run: create a file in /cache
    code_create = """
with open('/cache/testfile.txt', 'w') as f:
    f.write('persistent data')
"""
    result_create = run_python_in_docker(code_create)
    assert isinstance(result_create, DockerRunResult)
    assert result_create.stderr == "" or "Traceback" not in result_create.stderr

    # Second run: check that the file exists and contents are correct, then remove it
    code_check_and_remove = """
import os
with open('/cache/testfile.txt', 'r') as f:
    content = f.read()
print('CACHE_CONTENT:', content)
os.remove('/cache/testfile.txt')
"""
    result_check = run_python_in_docker(code_check_and_remove)
    assert isinstance(result_check, DockerRunResult)
    assert "CACHE_CONTENT: persistent data" in result_check.stdout
