import subprocess
import sys
from pathlib import Path
from typing import Optional
import typer
from questions import extract_questions

app = typer.Typer()


@app.command()
def main(
    models: str = typer.Option("all", help="Models to use"),
    n_iter: int = typer.Option(3, "-n", help="Number of iterations"),
    question: Optional[str] = typer.Option(
        None, help="Question number (int) or text (str)"
    ),
):
    """Run plan-query.py for all or a specific question."""
    questions = extract_questions()
    script_path = Path(__file__).parent / "plan-query.py"

    if question is not None:
        if isinstance(question, str) and question.isdigit():
            idx = int(question) - 1
            if idx < 0 or idx >= len(questions):
                typer.echo(f"Question number {question} is out of range.")
                raise typer.Exit(1)
            qs = [(idx + 1, questions[idx])]
        else:
            qs = [(0, question)]
    else:
        qs = list(enumerate(questions, 1))

    for i, q in qs:
        output_file = (
            Path.cwd() / f"plan-question-{i:02d}.md"
            if i
            else Path.cwd() / "plan-question-custom.md"
        )
        typer.echo(
            f"Running question {i:02d}..." if i else "Running custom question..."
        )
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                q,
                str(output_file),
                "--models",
                models,
                "-n",
                str(n_iter),
            ],
            text=True,
        )
        typer.echo(f"Saved output to {output_file}")


if __name__ == "__main__":
    app()
