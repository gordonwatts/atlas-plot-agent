import logging
from enum import Enum
from pathlib import Path
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


# Enum for allowed cache types
class CacheType(str, Enum):
    llm_plan = "llm_plan"
    llm_code = "llm_code"
    code = "code"
    hints = "hints"


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
        # question_hash = hashlib.sha1(question.encode("utf-8")).hexdigest()[:8]

        # Loop over each model
        for model_name in valid_model_names:
            fh_out.write(f"\n## Model {all_models[model_name].model_name}\n")

            # Build prompt
            fh_out.write("\n### Solution Outline\n")
            base_prompt = config.prompts["preplan"]
            prompt = base_prompt.format(
                question=question,
                hints="\n".join(plan_hint_contents),
            )
            logging.debug(f"Built prompt for planning: {prompt}")

            # Run against model
            logging.debug(f"Running against model {all_models[model_name].model_name}")
            usage_info, message = run_llm(
                prompt,
                all_models[model_name],
                fh_out,
                ignore_cache=CacheType.llm_plan in ignore_cache,
            )
            solution_outline = message
            print(usage_info)

            # Next, do the same for the phase plan
            fh_out.write("\n### Solution Code Phases\n")
            base_prompt = config.prompts["phase_plan"]
            prompt = base_prompt.format(
                question=question,
                hints="\n".join(plan_hint_contents),
                solution_outline=solution_outline,
            )
            logging.debug(f"Built prompt code phases: {prompt}")

            # Run against model
            logging.debug(f"Running against model {all_models[model_name].model_name}")
            usage_info, message = run_llm(
                prompt,
                all_models[model_name],
                fh_out,
                ignore_cache=CacheType.llm_plan in ignore_cache,
            )
            print(usage_info)

            # Split the code into sections
            code_sections = extract_by_phase(message)

            # Build the code for servicex
            hint_phase_code_sx = load_hint_files(
                config.hint_files["phase_code_sx"], CacheType.hints in ignore_cache
            )

            code_it_up(
                fh_out,
                all_models[model_name],
                config.prompts["phase_code_sx"],
                config.prompts["phase_code_sx_fix"],
                4,
                prompt_args={
                    "question": question,
                    "hints": "\n".join(hint_phase_code_sx),
                    "sx_code": code_sections["ServiceX"],
                },
                ignore_llm_cache=CacheType.llm_plan in ignore_cache,
                ignore_code_cache=CacheType.llm_code in ignore_cache,
            )


if __name__ == "__main__":
    app()
