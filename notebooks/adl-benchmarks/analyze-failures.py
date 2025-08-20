import os
from typing import List

import openai
import typer
import yaml
from disk_cache import diskcache_decorator
from dotenv import dotenv_values, find_dotenv
from hint_files import load_hint_files
from query_config import load_config
from tqdm import tqdm

app = typer.Typer()

prompt = """
Your task is to analyze the error in some generated python code.

You will be given the original particle physics question that an LLM generated the code to answer,
a set of hint files that guided the LLM in generating the code, the code that was generated,
and stdout and stderr from running that code.

There is a special kind of error - if the LLM was trying to do something that was not needed to
answer the question (e.g. it made the question more complex), or it didn't understand the question.
In that case, there was an *interpretation* error.

Your task is to:
1. Understand what the code was trying to do when it encountered the error and how it is related
to the original question (e.g. was it explicitly asked for or a necessary step)
1. Understand which phase the error occurred in: interpretation, servicex, awkward, hist, or
vector.
1. Write a one line description of what the error was.
1. Determine if this was a policy error (e.g. there will be a note in the
stderr output to that effect)
1. And determine if the code does not follow instructions in the hint files (the alternative
is not using servicex or award correctly)
1. If relevant, a line of text to add to the hint files to help avoid this error next time the
original LLM runs.

Repeat as necessary above if you spot more than one error (e.g. an interpretation error and
a code error itself.)

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
- goal: "<what-part-of-question-was-it-trying-to-answer>"
  phase: "<phase>"
  error_description: "<error_description>"
  policy_error: <True/False>
  hint_violation: <True/False>
  hint_text_suggestion: <concise-suggested-text>
>>end-reply<<
"""


@diskcache_decorator(".openai_analysis_responses")
def get_openai_response(prompt: str, model_name: str):

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model_name, messages=[{"role": "user", "content": prompt}]
    )
    assert response.choices[0].message.content is not None, "No content in response"
    # Return both response and timing for caching
    return response


def analysis(
    question: str, code: str, stderr: str, stdout: str, hint_text: str
) -> List[dict]:
    """
    Analyze the failure of generated Python code using an LLM.

    Given the original question, generated code, standard error and output, and hint text,
    this function formats a prompt for an LLM to analyze the error, determine its phase,
    provide a description, and suggest improvements to hint files. The response is parsed
    from YAML into a list of dictionaries describing each error found.

    Args:
        question (str): The original question that led to code generation.
        code (str): The generated Python code.
        stderr (str): The standard error output from running the code.
        stdout (str): The standard output from running the code.
        hint_text (str): The concatenated text from hint files.

    Returns:
        List[dict]: A list of dictionaries, each describing an error found in the analysis.
    """
    full_prompt = prompt.format(
        hints=hint_text,
        old_code=code,
        stderr=stderr,
        stdout=stdout,
        question=question,
    )

    # Get the LLM response and turn it into a dictionary
    response = get_openai_response(full_prompt, "gpt-5")
    message = response.choices[0].message.content
    assert message is not None
    cleaned_message = (
        message.replace(">>start-reply<<", "").replace(">>end-reply<<", "").strip()
    )
    info = yaml.safe_load(cleaned_message)

    return info


@app.command()
def analyze(files: List[str]):
    """Analyze a list of failure YAML files."""

    # Load hint files from the standard direct-query config
    config = load_config()
    hint_files = load_hint_files(config.hint_files)
    hint_text = "\n".join(hint_files)

    # Load API key for openAI
    env_path = find_dotenv()
    env_vars = dotenv_values(env_path)
    api_key = env_vars.get("api_openai_com_API_KEY")
    assert api_key is not None, "No openai key found!"
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

    # Loop through each of the files and accumulate the results
    error_catalog = []
    for file_path in tqdm(files, desc="Analyzing files"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        code = data.get("code", "")
        stderr = data.get("stderr", "")
        stdout = data.get("stdout", "")
        question = data.get("question", "")
        result = analysis(question, code, stderr, stdout, hint_text)

        for d in result:
            d["iteration"] = data.get("iteration", 1)
            d["model"] = data.get("model", "")
            d["question"] = data.get("question", "")

        error_catalog.extend(result)

    # Store the error analysis
    with open("error_catalog.yaml", "w", encoding="utf-8") as f:
        yaml.dump(error_catalog, f)


if __name__ == "__main__":
    app()
