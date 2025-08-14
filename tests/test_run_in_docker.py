import os

import yaml

from atlas_plot_agent.run_in_docker import (
    DockerRunResult,
    check_code_policies,
    copy_servicex_yaml_if_exists,
    run_python_in_docker,
)


def test_check_code_policies_plt_savefig_present():
    code = """
import matplotlib.pyplot as plt
plt.plot([1,2,3],[4,5,6])
NFiles=1
plt.savefig('output.png')
"""
    result = check_code_policies(code)
    assert result is True


def test_check_code_policies_plt_savefig_missing():

    code = """
import matplotlib.pyplot as plt
plt.plot([1,2,3],[4,5,6])
# plt.savefig('output.png')
"""
    result = check_code_policies(code)
    assert isinstance(result, DockerRunResult)
    assert "plt.savefig not found" in result.stderr


def test_copy_servicex_yaml_adds_cache_path(tmp_path, monkeypatch):
    servicex_path = tmp_path / "home1" / "servicex.yaml"
    servicex_path.parent.mkdir()
    no_cache_yaml = {"some_key": "some_value"}
    servicex_path.write_text(yaml.safe_dump(no_cache_yaml))
    monkeypatch.setattr(
        os.path,
        "expanduser",
        lambda p: str(servicex_path) if p == "~/servicex.yaml" else p,
    )
    monkeypatch.setattr(os.path, "exists", lambda p: str(p) == str(servicex_path))
    monkeypatch.setattr("shutil.copy", lambda src, dst: None)
    target_dir = tmp_path / "target1"
    target_dir.mkdir()
    copy_servicex_yaml_if_exists(str(target_dir))
    copied_path = target_dir / "servicex.yaml"
    with open(copied_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["cache_path"] == "/cache"
    assert data["some_key"] == "some_value"


def test_copy_servicex_yaml_overwrites_cache_path(tmp_path, monkeypatch):
    servicex_path = tmp_path / "home2" / "servicex.yaml"
    servicex_path.parent.mkdir()
    with_cache_yaml = {"cache_path": "/old_cache", "other": 123}
    servicex_path.write_text(yaml.safe_dump(with_cache_yaml))
    monkeypatch.setattr(
        os.path,
        "expanduser",
        lambda p: str(servicex_path) if p == "~/servicex.yaml" else p,
    )
    monkeypatch.setattr(os.path, "exists", lambda p: str(p) == str(servicex_path))
    monkeypatch.setattr("shutil.copy", lambda src, dst: None)
    target_dir = tmp_path / "target2"
    target_dir.mkdir()
    copy_servicex_yaml_if_exists(str(target_dir))
    copied_path = target_dir / "servicex.yaml"
    with open(copied_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["cache_path"] == "/cache"
    assert data["other"] == 123


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


def test_check_code_policies_pass():
    code = """
NFiles=1
plt.savefig('output.png')
print('ok')
"""
    result = check_code_policies(code)
    assert result is True


def test_check_code_policies_missing_nfiles():
    code = """
print('no NFiles')
"""
    result = check_code_policies(code)
    assert isinstance(result, DockerRunResult)
    assert "Policy violation" in result.stderr


def test_check_code_policies_comment_handling():
    code = """
# NFiles=1 in comment
print('no NFiles')
"""
    result = check_code_policies(code)
    assert isinstance(result, DockerRunResult)
    assert "Policy violation" in result.stderr


def test_check_code_policies_comment_trailing():
    code = """
i = 1 # NFiles=1 in comment
print('no NFiles')
"""
    result = check_code_policies(code)
    assert isinstance(result, DockerRunResult)
    assert "Policy violation" in result.stderr


def test_check_code_policies_comment_string():
    code = """
print('no NFiles=1 in code')
"""
    result = check_code_policies(code)
    assert isinstance(result, DockerRunResult)
    assert "Policy violation" in result.stderr
