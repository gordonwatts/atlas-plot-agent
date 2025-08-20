import hashlib
import logging
import typer
from query_config import load_plan_config
from hint_files import load_hint_files
from models import load_models, process_model_request

app = typer.Typer()


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask the API"),
    models: str = typer.Option(
        None,
        help="Comma-separated list of model names to run (default: pulled from config). "
        "Use `all` to run all known models.",
    ),
    ignore_cache: bool = typer.Option(
        False, "--ignore-cache", help="Ignore disk cache for model queries."
    ),
    n_iter: int = typer.Option(
        1, "--n-iter", "-n", min=1, help="Number of iterations to run (must be >= 1)."
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

    # Get the config loaded
    config = load_plan_config()
    plan_hint_contents = load_hint_files(config.hint_files["plan"])

    # Load models
    all_models = load_models()
    valid_model_names = process_model_request(models, all_models, config.model_name)

    # stdout is a markdown file
    print(f"# {question}\n")
    question_hash = hashlib.sha1(question.encode("utf-8")).hexdigest()[:8]

    # Loop over each model
    for model_name in valid_model_names:
        print(f"\n## Model {all_models[model_name].model_name}")

        base_prompt = config.plan_prompt
        prompt = base_prompt.format(
            question=question,
            hints="\n".join(plan_hint_contents),
        )
        logging.debug(f"Built prompt for planning: {prompt}")


if __name__ == "__main__":
    app()
