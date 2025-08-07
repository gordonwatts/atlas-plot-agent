import typer


from pydantic import BaseModel
import yaml


class DirectQueryConfig(BaseModel):
    """
    Configuration model for direct-query CLI.
    Contains a list of hint files and a prompt string.
    """

    hint_files: list[str]
    prompt: str


def load_config(config_path: str = "direct-query-config.yaml") -> DirectQueryConfig:
    """
    Load configuration from a YAML file and return a DirectQueryConfig instance.
    """
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return DirectQueryConfig(**data)


app = typer.Typer(
    help=(
        "use default configuration to ask the api a question and "
        "generate code in response"
    )
)


@app.command()
def ask(question: str = typer.Argument(..., help="The question to ask the API")):
    """
    Command to ask a question using the default configuration.
    Loads config and prints the question and config for demonstration.
    """
    config = load_config()
    print(f"Loaded config: {config}")
    print(f"Question: {question}")


if __name__ == "__main__":
    app()
