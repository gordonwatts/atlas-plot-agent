import typer

app = typer.Typer()


@app.command()
def run_agent(task: str):
    """Run everything.

    Args:
        task (str): What we need to do
    """
    # result = openai_agents.process(task)  # Adjust based on actual API
    typer.echo("Result: fork it")


if __name__ == "__main__":
    app()
