import hashlib
import logging
from enum import Enum
from pathlib import Path
import re
from typing import List

import typer
from hint_files import load_hint_files
from models import (
    load_models,
    process_model_request,
    run_llm,
    extract_by_phase,
)
from query_config import load_plan_config
from questions import extract_questions
from query_code import code_it_up
from utils import IndentedDetailsBlock


# Enum for allowed cache types
class CacheType(str, Enum):
    llm_plan = "llm_plan"
    llm_code = "llm_code"
    code = "code"
    hints = "hints"


def extract_struct_line(stdout: str) -> str:
    match = re.search(r"^\d+\s*\*\s*\{.*\}$", stdout, re.MULTILINE)
    assert match, f"Failed to find structured line in {stdout}"
    return match.group(0)


app = typer.Typer()


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask the API"),
    models: str = typer.Option(
        None,
        help="Comma-separated list of model names to run (default: pulled from config). "
        "Use `all` to run all known models.",
    ),
    output: Path = typer.Argument(
        Path("output.md"), help="Output filename, defaults to `output.md`"
    ),
    ignore_cache: List[CacheType] = typer.Option(
        [],
        "--ignore-cache",
        help="Cache types to ignore (llm_plan, llm_code, code, hints). Use multiple times.",
        case_sensitive=True,
    ),
    n_iter: int = typer.Option(
        1,
        "--n-iter",
        "-n",
        min=1,
        help="Number of iterations to run (must be >= 1).",
    ),
    verbose: int = typer.Option(
        0,
        "-v",
        count=True,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    ),
):

    # Set logging level
    if verbose == 0:
        logging.basicConfig(level=logging.WARNING)
    elif verbose == 1:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG)

    # See if this is a reference to a particular question.
    if str.isdigit(question):
        questions = extract_questions()
        n_question = int(question)
        if n_question > len(questions):
            raise ValueError(
                f"Requested question number {n_question} is out of range "
                f"(there are only {len(questions)})"
            )
        question = questions[n_question - 1]

    # Get the config loaded
    config = load_plan_config()
    plan_hint_contents = load_hint_files(
        config.hint_files["plan"], CacheType.hints in ignore_cache
    )

    # Load models
    all_models = load_models()
    valid_model_names = process_model_request(models, all_models, config.model_name)

    with output.open("wt", encoding="utf-8") as fh_out:

        # stdout is a markdown file
        fh_out.write(f"# {question}\n")
        question_hash = hashlib.sha1(question.encode("utf-8")).hexdigest()[:8]

        # Loop over each model
        for model_name in valid_model_names:
            fh_out.write(f"\n## Model {all_models[model_name].model_name}\n")

            # Build prompt
            fh_out.write("\n### Problem Analysis & Breakdown\n")
            with IndentedDetailsBlock(fh_out, "Solution Outline"):
                base_prompt = config.prompts["preplan"]
                prompt = base_prompt.format(
                    question=question,
                    hints="\n".join(plan_hint_contents),
                )
                logging.debug(f"Built prompt for planning: {prompt}")

                # Run against model
                logging.debug(
                    f"Running against model {all_models[model_name].model_name}"
                )
                usage_info, message = run_llm(
                    prompt,
                    all_models[model_name],
                    fh_out,
                    ignore_cache=CacheType.llm_plan in ignore_cache,
                )
                solution_outline = message
                print(usage_info)

            # Next, do the same for the phase plan
            with IndentedDetailsBlock(fh_out, "Solution Code Phases"):
                base_prompt = config.prompts["phase_plan"]
                prompt = base_prompt.format(
                    question=question,
                    hints="\n".join(plan_hint_contents),
                    solution_outline=solution_outline,
                )
                logging.debug(f"Built prompt code phases: {prompt}")

                # Run against model
                logging.debug(
                    f"Running against model {all_models[model_name].model_name}"
                )
                usage_info, message = run_llm(
                    prompt,
                    all_models[model_name],
                    fh_out,
                    ignore_cache=CacheType.llm_plan in ignore_cache,
                )
                print(usage_info)

            # Split the code into sections
            code_sections = extract_by_phase(message)

            fh_out.write("\n### Code\n")

            # Build the code for servicex
            hint_phase_code_sx = load_hint_files(
                config.hint_files["phase_code_sx"], CacheType.hints in ignore_cache
            )

            called_code = """
r = load_data_from_sx()
print(r.type)
        """

            with IndentedDetailsBlock(fh_out, "ServiceX Code"):
                sx_code_result, sx_code = code_it_up(
                    fh_out,
                    all_models[model_name],
                    config.prompts["phase_code_sx"],
                    config.prompts["phase_code_sx_fix"],
                    4,
                    called_code,
                    prompt_args={
                        "question": question,
                        "hints": "\n".join(hint_phase_code_sx),
                        "sx_code": code_sections["ServiceX"],
                    },
                    ignore_llm_cache=CacheType.llm_plan in ignore_cache,
                    ignore_code_cache=CacheType.llm_code in ignore_cache,
                )

            # Build the code for awkward
            hint_phase_code_awkward = load_hint_files(
                config.hint_files["phase_code_awkward"], CacheType.hints in ignore_cache
            )
            data_format = extract_struct_line(sx_code_result.stdout)

            called_code = f"""
{sx_code}
data = load_data_from_sx()
r = generate_histogram_data(data)
print(r.type)
        """

            with IndentedDetailsBlock(fh_out, "Awkward Code"):
                _, awk_code = code_it_up(
                    fh_out,
                    all_models[model_name],
                    config.prompts["phase_code_awkward"],
                    config.prompts["phase_code_awkward_fix"],
                    4,
                    called_code,
                    prompt_args={
                        "question": question,
                        "hints": "\n".join(hint_phase_code_awkward),
                        "awkward_code": code_sections["Awkward"],
                        "data_format": data_format,
                    },
                    ignore_llm_cache=CacheType.llm_plan in ignore_cache,
                    ignore_code_cache=CacheType.llm_code in ignore_cache,
                )

            # Build the code for histogram
            hint_phase_code_hist = load_hint_files(
                config.hint_files["phase_code_hist"], CacheType.hints in ignore_cache
            )

            called_code = f"""
{sx_code}
{awk_code}
data = load_data_from_sx()
r = generate_histogram_data(data)
plot_hist(r)
        """

            with IndentedDetailsBlock(fh_out, "Hist Code"):
                hist_result, _ = code_it_up(
                    fh_out,
                    all_models[model_name],
                    config.prompts["phase_code_hist"],
                    config.prompts["phase_code_hist_fix"],
                    1,
                    called_code,
                    prompt_args={
                        "question": question,
                        "hints": "\n".join(hint_phase_code_hist),
                        "hist_code": code_sections["Histogram"],
                    },
                    ignore_llm_cache=CacheType.llm_plan in ignore_cache,
                    ignore_code_cache=CacheType.llm_code in ignore_cache,
                )

            # If there are png files, then save them!
            for f_name, data in hist_result.png_files:
                # Sanitize model_name for filesystem
                safe_model_name = all_models[model_name].model_name.replace("/", "_")
                local_name = f"{question_hash}_{safe_model_name}_{f_name}"
                with open(local_name, "wb") as dst:
                    dst.write(data)
                fh_out.write(f"![{local_name}]({local_name})")


if __name__ == "__main__":
    app()
