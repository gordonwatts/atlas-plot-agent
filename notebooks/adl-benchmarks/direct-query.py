import hashlib
import logging
from pathlib import Path
from typing import List, Tuple

import typer
from hint_files import load_hint_files
from models import load_models, process_model_request
from query_code import CodeExtractablePolicy, code_it_up
from query_config import load_config
from questions import extract_questions
from utils import IndentedDetailsBlock

from atlas_plot_agent.run_in_docker import (
    DockerRunResult,
    NFilesPolicy,
    PltSavefigPolicy,
    print_md_table_for_phased_usage_docker,
)
from atlas_plot_agent.usage_info import (
    UsageInfo,
    print_md_table_for_phased_usage,
    sum_usage_infos,
)

app = typer.Typer(
    help=(
        "use default configuration to ask the api a question and "
        "generate code in response"
    )
)


@app.command()
def ask(
    question: str = typer.Argument(
        ...,
        help="The question to ask the API, or an index into the README questions (1 based index)",
    ),
    output: Path = typer.Argument(..., help="Output file for markdown"),
    models: str = typer.Option(
        None,
        help="Comma-separated list of model names to run (default: pulled from config). "
        "Use `all` to run all known models.",
    ),
    ignore_cache: bool = typer.Option(
        False, "--ignore-cache", help="Ignore disk cache for model queries."
    ),
    error_info: bool = typer.Option(
        False,
        "--write-error-info",
        help="Writes a small `fail` file for each error that can be analyzed later",
    ),
    n_iter: int = typer.Option(
        1, "--n-iter", "-n", min=1, help="Number of iterations to run (must be >= 1)."
    ),
    docker_image: str = typer.Option(
        None,
        "--docker-image",
        help="Override the docker image name (default: use value from config)",
    ),
):
    """
    Command to ask a question using the default configuration.
    Runs the question against one or more models, prints results, and prints a summary table.
    """

    # Load configuration
    config = load_config()
    hint_contents = load_hint_files(config.hint_files)

    # Load models
    all_models = load_models()
    valid_model_names = process_model_request(models, all_models, config.model_name)

    # Check number of requested iterations is good
    if n_iter < 1:
        logging.error(
            f"Error: command line option `n_iter` must be >= 1 (got {n_iter})"
        )
        return

    # If the question is an integer, use it to fetch the proper question.
    if str.isnumeric(question):
        questions = extract_questions()
        if int(question) >= len(questions):
            raise ValueError(
                f"Invalid question index: {question}. Max is {len(questions)}."
            )
        question = questions[int(question) - 1]

    # Process everything!
    with output.open("wt", encoding="utf-8") as fh_out:
        fh_out.write(f"# {question}\n\n")
        question_hash = hashlib.sha1(question.encode("utf-8")).hexdigest()[:8]

        table_rows = []

        for model_name in valid_model_names:
            fh_out.write(f"## Model {all_models[model_name].model_name}\n\n")

            llm_usage: List[Tuple[str, UsageInfo]] = []
            code_usage: List[Tuple[str, DockerRunResult]] = []

            result, code, good_run = code_it_up(
                fh_out,
                all_models[model_name],
                config.prompt,
                config.modify_prompt,
                [NFilesPolicy(), PltSavefigPolicy(), CodeExtractablePolicy()],
                n_iter,
                "",
                {"question": question, "hints": "\n".join(hint_contents)},
                docker_image if docker_image is not None else config.docker_image,
                ignore_cache,
                ignore_cache,
                lambda s, usg: llm_usage.append((s, usg)),
                lambda s, doc_usg: code_usage.append((s, doc_usg)),
            )

            if not good_run:
                fh_out.write("\n**Failed**\n\n")

            fh_out.write("\n\n")

            # Write out the png files
            if good_run and result is not None:
                output_directory = output.parent / "img"
                output_directory.mkdir(exist_ok=True)
                for f_name, data in result.png_files:
                    # Sanitize model_name for filesystem
                    safe_model_name = model_name.replace("/", "_")
                    local_name = f"{question_hash}_{safe_model_name}_{f_name}"
                    with (output_directory / local_name).open("wb") as dst:
                        dst.write(data)
                    fh_out.write(f"![{local_name}](img/{local_name})\n")

            # # If we are writing out the error info, we should do that here
            # if error_info and result is not None:
            #     e_info = {
            #         "code": code,
            #         "question": question,
            #         "stdout": result.stdout,
            #         "stderr": result.stderr,
            #         "model": model_name,
            #     }
            #     output_file = f"z_fail_{question_hash}_{model_name}.yaml"
            #     with open(output_file, "w") as f:
            #         yaml.dump(e_info, f)

            # Write out summary tables with details of what we "did".
            with IndentedDetailsBlock(fh_out, "Usage"):
                print_md_table_for_phased_usage(fh_out, llm_usage)
                print_md_table_for_phased_usage_docker(fh_out, code_usage)

            fh_out.write("\n\n")

            total_llm_usage = sum_usage_infos([l for _, l in llm_usage])
            table_rows.append(
                {
                    "model": model_name,
                    "llm_time": total_llm_usage.elapsed,
                    "prompt_tokens": total_llm_usage.prompt_tokens,
                    "completion_tokens": total_llm_usage.completion_tokens,
                    "total_tokens": total_llm_usage.total_tokens,
                    "cost": total_llm_usage.cost,
                    "attempts": len(llm_usage),
                    "code_time": sum([c.elapsed for _, c in code_usage]),
                    "result": good_run,
                }
            )

        # Write out final totals CSV and tabular data
        fh_out.write("## CSV\n\n")
        # Write CSV header
        csv_header = [
            "Model",
            "Time",
            "Prompt Tokens",
            "Completion Tokens",
            "Total Tokens",
            "Estimated Cost",
            "Attempts",
            "Code Time",
            "Result",
        ]
        fh_out.write(",".join([s.replace(" ", "") for s in csv_header]) + "\n")

        # Write each row
        for row in table_rows:
            csv_row = [
                str(row["model"]),
                f"{row['llm_time']:.2f}",
                str(row["prompt_tokens"]) if row["prompt_tokens"] is not None else "-",
                (
                    str(row["completion_tokens"])
                    if row["completion_tokens"] is not None
                    else "-"
                ),
                str(row["total_tokens"]) if row["total_tokens"] is not None else "-",
                f"{row['cost']:.3f}" if row["cost"] is not None else "-",
                str(row["attempts"]),
                f"{row['code_time']:.2f}",
                "Success" if row["result"] else "Failure",
            ]
            fh_out.write(",".join(csv_row) + "\n")

        fh_out.write("## Summary\n")
        # Write markdown table header
        fh_out.write("| " + " | ".join(csv_header) + " |\n")
        fh_out.write("|" + "|".join(["-" * len(h) for h in csv_header]) + "|\n")
        # Write each row as markdown table
        for row in table_rows:
            fh_out.write(
                f"| {row['model']} "
                f"| {row['llm_time']:.2f} "
                f"| {row['prompt_tokens']} "
                f"| {row['completion_tokens']} "
                f"| {row['total_tokens']} "
                f"| ${row['cost']:.3f} "
                f"| {row['attempts']} "
                f"| {row['code_time']:.2f} "
                f"| {'Success' if row['result'] else 'Fail'} |\n"
            )


if __name__ == "__main__":
    app()


if __name__ == "__main__":
    app()
