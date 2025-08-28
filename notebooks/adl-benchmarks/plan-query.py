import hashlib
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import typer
from hint_files import load_hint_files
from models import (
    load_models,
    process_model_request,
    run_llm,
    extract_by_phase,
    UsageInfo,
)
from query_config import load_plan_config
from questions import extract_questions
from query_code import (
    code_it_up,
    DockerRunResult,
    CodeExtractablePolicy,
    llm_execute_loop,
    check_policy,
    Policy,
    run_llm_loop_simple,
)
from utils import IndentedDetailsBlock
from atlas_plot_agent.usage_info import print_md_table_for_phased_usage
from atlas_plot_agent.run_in_docker import (
    print_md_table_for_phased_usage_docker,
    NFilesPolicy,
    PltSavefigPolicy,
)


# Enum for allowed cache types
class CacheType(str, Enum):
    llm_plan = "llm_plan"
    llm_code = "llm_code"
    code = "code"
    hints = "hints"


def extract_struct_line(
    stdout: str, search_string: str = "ServiceX Data Type Structure"
) -> str:
    lines = stdout.splitlines()
    struct_lines = [ln for ln in lines if search_string in ln]
    assert (
        len(struct_lines) == 1
    ), f"Failed to find awkward array structured line in '{stdout}'"
    return struct_lines[0]


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
        model_usage: Dict[str, Tuple[UsageInfo, float, bool]] = {}
        good_run = True
        for model_name in valid_model_names:
            fh_out.write(f"\n## Model {all_models[model_name].model_name}\n")

            llm_usage: List[Tuple[str, UsageInfo]] = []
            code_run_usage: List[Tuple[str, DockerRunResult]] = []

            # Build prompt
            fh_out.write("\n### Problem Analysis & Breakdown\n")
            with IndentedDetailsBlock(fh_out, "Solution Outline"):

                solution_outline = run_llm_loop_simple(
                    fh_out,
                    config.prompts["preplan"],
                    {"question": question, "hints": "\n".join(plan_hint_contents)},
                    n_iter,
                    all_models[model_name],
                    CacheType.llm_plan in ignore_cache,
                    lambda n, u: llm_usage.append((f"Solution Outline {n}", u)),
                )

            good_run = len(solution_outline) > 0

            # Next, do the same for the phase plan
            if good_run:
                with IndentedDetailsBlock(fh_out, "Solution Code Phases"):

                    class CodePhasePolicy(Policy):
                        def check(self, m: str) -> Optional[str]:
                            code_sections = extract_by_phase(m)
                            good_run = all(
                                p in code_sections.keys()
                                for p in ["ServiceX", "Awkward", "Histogram"]
                            )
                            if not good_run:
                                return (
                                    "You must have a `ServiceX`, `Awkward`, and `Histogram` "
                                    "section as in required format instructions."
                                )
                            return None

                    def prompt_and_policy() -> (
                        Generator[tuple[str, List[Policy]], Any, None]
                    ):
                        yield config.prompts["phase_plan"], [CodePhasePolicy()]
                        yield config.prompts["phase_plan_fix"], [CodePhasePolicy()]

                    def llm_dispatcher(prompt: str, n_iter: int):
                        usage_info, message = run_llm(
                            prompt,
                            all_models[model_name],
                            fh_out,
                            ignore_cache=CacheType.llm_plan in ignore_cache,
                        )
                        llm_usage.append(("Code Phases", usage_info))
                        return message

                    hints = {
                        "question": question,
                        "hints": "\n".join(plan_hint_contents),
                        "solution_outline": solution_outline,
                    }

                    def execute_code_null(
                        code: str, n_iter: int
                    ) -> Tuple[bool, Dict[str, str]]:
                        return True, {}

                    message, good_run = llm_execute_loop(
                        fh_out,
                        prompt_and_policy(),
                        n_iter,
                        hints,
                        llm_dispatcher,
                        lambda s: s,
                        execute_code_null,
                        lambda msg, pols: check_policy(fh_out, msg, pols),
                    )

                # Split the code into sections
                if not good_run:
                    fh_out.write("\n**Failed Phase Generation**\n")

            if good_run:
                fh_out.write("\n### Code\n")

                # Build the code for servicex
                hint_phase_code_sx = load_hint_files(
                    config.hint_files["phase_code_sx"], CacheType.hints in ignore_cache
                )

                called_code = """
r = load_data_from_sx()
print("ServiceX Data Type Structure: " + str(r.type))
            """

                code_sections = extract_by_phase(message)

                with IndentedDetailsBlock(fh_out, "ServiceX Code"):
                    sx_code_result, sx_code, good_run = code_it_up(
                        fh_out,
                        all_models[model_name],
                        config.prompts["phase_code_sx"],
                        config.prompts["phase_code_sx_fix"],
                        [NFilesPolicy(), CodeExtractablePolicy()],
                        n_iter,
                        called_code,
                        prompt_args={
                            "question": question,
                            "hints": "\n".join(hint_phase_code_sx),
                            "sx_code": code_sections["ServiceX"],
                        },
                        ignore_llm_cache=CacheType.llm_code in ignore_cache,
                        ignore_code_cache=CacheType.code in ignore_cache,
                        llm_usage_callback=lambda n, u: llm_usage.append(
                            (f"ServiceX Code {n}", u)
                        ),
                        docker_usage_callback=lambda n, u: code_run_usage.append(
                            (f"ServiceX Code {n}", u)
                        ),
                    )

                if not good_run:
                    fh_out.write("\n**Failed ServiceX Code Generation**\n")

            # Build the code for awkward
            if good_run:
                hint_phase_code_awkward = load_hint_files(
                    config.hint_files["phase_code_awkward"],
                    CacheType.hints in ignore_cache,
                )
                assert sx_code_result is not None
                data_format = extract_struct_line(sx_code_result.stdout)

                called_code = f"""
{sx_code}
data = load_data_from_sx()
r = generate_histogram_data(data)
print ("Histogram Data: " + str(r.keys()))
        """

                with IndentedDetailsBlock(fh_out, "Awkward Code"):
                    awk_code_result, awk_code, good_run = code_it_up(
                        fh_out,
                        all_models[model_name],
                        config.prompts["phase_code_awkward"],
                        config.prompts["phase_code_awkward_fix"],
                        [CodeExtractablePolicy()],
                        n_iter,
                        called_code,
                        prompt_args={
                            "question": question,
                            "hints": "\n".join(hint_phase_code_awkward),
                            "awkward_code": code_sections["Awkward"],
                            "data_format": data_format,
                        },
                        ignore_llm_cache=CacheType.llm_code in ignore_cache,
                        ignore_code_cache=CacheType.code in ignore_cache,
                        llm_usage_callback=lambda n, u: llm_usage.append(
                            (f"Awkward Code {n}", u)
                        ),
                        docker_usage_callback=lambda n, u: code_run_usage.append(
                            (f"Awkward Code {n}", u)
                        ),
                    )

                if not good_run:
                    fh_out.write("\n**Failed Awkward Code Generation**\n")

            if good_run:
                assert awk_code_result is not None
                histogram_dict_names = extract_struct_line(
                    awk_code_result.stdout, "Histogram Data: "
                )

                # Build the code for histogram
                hint_phase_code_hist = load_hint_files(
                    config.hint_files["phase_code_hist"],
                    CacheType.hints in ignore_cache,
                )

                called_code = f"""
{sx_code}
{awk_code}
data = load_data_from_sx()
r = generate_histogram_data(data)
plot_hist(r)
        """

                with IndentedDetailsBlock(fh_out, "Hist Code"):
                    hist_result, _, good_run = code_it_up(
                        fh_out,
                        all_models[model_name],
                        config.prompts["phase_code_hist"],
                        config.prompts["phase_code_hist_fix"],
                        [PltSavefigPolicy(), CodeExtractablePolicy()],
                        n_iter,
                        called_code,
                        prompt_args={
                            "question": question,
                            "hints": "\n".join(hint_phase_code_hist),
                            "hist_code": code_sections["Histogram"],
                            "histogram_dict_names": histogram_dict_names,
                        },
                        ignore_llm_cache=CacheType.llm_code in ignore_cache,
                        ignore_code_cache=CacheType.code in ignore_cache,
                        llm_usage_callback=lambda n, u: llm_usage.append(
                            (f"Histogram Code {n}", u)
                        ),
                        docker_usage_callback=lambda n, u: code_run_usage.append(
                            (f"Histogram Code {n}", u)
                        ),
                    )

                if not good_run:
                    reason = "Crash" if hist_result is None else "No PNG files found"
                    fh_out.write(f"\n**Failed Histogram Code Generation ({reason})**\n")

            # Print out usage info for this in a markdown table.
            fh_out.write("\n\n### Usage\n\n")
            with IndentedDetailsBlock(fh_out, "LLM Usage"):
                total_llm_usage = print_md_table_for_phased_usage(fh_out, llm_usage)
            with IndentedDetailsBlock(fh_out, "Docker Usage"):
                total_seconds = print_md_table_for_phased_usage_docker(
                    fh_out, code_run_usage
                )

            model_usage[model_name] = (total_llm_usage, total_seconds, good_run)

            # If there are png files, then save them!
            if good_run:
                fh_out.write("\n\n### Plots\n\n")
                assert hist_result is not None
                output_directory = output.parent / "img"
                output_directory.mkdir(exist_ok=True)
                for f_name, data in hist_result.png_files:
                    # Sanitize model_name for filesystem
                    safe_model_name = all_models[model_name].model_name.replace(
                        "/", "_"
                    )
                    local_name = f"{question_hash}_plan_{safe_model_name}_{f_name}"
                    with (output_directory / local_name).open("wb") as dst:
                        dst.write(data)
                    fh_out.write(f"![{local_name}](img/{local_name})")

        if model_usage:
            fh_out.write("\n\n## Model Usage\n\n")
            fh_out.write(
                "| Model | Success? | LLM Time (secs) | Docker Time (secs) "
                "| Total Time (secs) | LLM Cost (USD) |\n"
            )
            fh_out.write("|---" * 6 + "|\n")
            for model_name, (u_llm, u_docker, u_success) in model_usage.items():
                fh_out.write(
                    f"| {model_name} "
                    f"| {"Success" if u_success else "Fail"} "
                    f"| {u_llm.elapsed:.2f} "
                    f"| {u_docker:.2f} "
                    f"| {u_llm.elapsed + u_docker:.2f} "
                    f"| ${u_llm.cost:.3f} |\n"
                )


if __name__ == "__main__":
    app()
