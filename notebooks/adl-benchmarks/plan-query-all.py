import subprocess
import sys
from pathlib import Path
from questions import extract_questions


def main():
    questions = extract_questions()
    script_path = Path(__file__).parent / "plan-query.py"
    for i, q in enumerate(questions, 1):
        output_file = Path.cwd() / f"plan-question-{i:02d}.md"
        print(f"Running question {i:02d}...")

        # Run direct-query.py with the question and --models all
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                q,
                str(output_file),
                "--models",
                "all",
                "-n",
                "3",
            ],
            text=True,
        )
        print(f"Saved output to {output_file}")


if __name__ == "__main__":
    main()
