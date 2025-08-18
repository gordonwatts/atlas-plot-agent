import typer
import yaml
from typing import List

from query_config import load_config
from hint_files import load_hint_files

app = typer.Typer()

prompt = """
Your task is to analyze the error in some generated python code.

You will be given the original particle physics question that an LLM generated the code to answer,
a set of hint files that guided the LLM in generating the code, the code that was generated,
and stdout and stderr from running that code.

Your task is to:
1. Understand which phase the error occurred in: servicex, awkward, hist, or vector.
1. Write a one line description of what the error was
1. Determine if this was a policy error (e.g. there will be a note in the
stderr output to that effect)
1. And determine if the code does not follow instructions in the hint files (the alternative
is not using servicex or award correctly)
1. If relevant, text to add to the hint files to help avoid this error next time the original
LLM runs.

** Question that generated the code **
{question}

** Hints **
{hints}

** Previous Code **
```python
{old_code}
```

** stdout **

```text
{stdout}
```

** stderr **
```text
{stderr}
```

** Final Instruction **
Please write your reply in this form (yaml). If you find more than one error,
add more than one entry:
>>start-reply<<
- phase: "<phase>"
  error_description: "<error_description>"
  policy_error: <True/False>
  hint_violation: <True/False>
  hint_text_suggestion: <suggested-text>
>>end-reply<<
"""


def analysis(
    question: str, code: str, stderr: str, stdout: str, hint_text: str
) -> List[dict]:
    full_prompt = prompt.format(
        hints=hint_text,
        old_code=code,
        stderr=stderr,
        stdout=stdout,
        question=question,
    )
    return [{}]


@app.command()
def analyze(files: List[str]):
    """Analyze a list of failure YAML files."""

    # Load hint files from the standard direct-query config
    config = load_config()
    hint_files = load_hint_files(config.hint_files)
    hint_text = "\n".join(hint_files)

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        code = data.get("code", "")
        stderr = data.get("stderr", "")
        stdout = data.get("stdout", "")
        question = data.get("question", "")
        result = analysis(question, code, stderr, stdout, hint_text)
        typer.echo(f"Analysis for {file_path}: {result}")


if __name__ == "__main__":
    app()
