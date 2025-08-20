import subprocess
import sys
from pathlib import Path
from questions import extract_questions


def main():
    questions = extract_questions()
    script_path = Path(__file__).parent / "direct-query.py"
    for i, q in enumerate(questions, 1):
        output_file = Path.cwd() / f"direct-question-{i:02d}.md"
        print(f"Running question {i:02d}...")

        # Run direct-query.py with the question and --models all
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                q,
                "--models",
                "all",
                "-n",
                "3",
                "--write-error-info",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        # Save output to file
        with open(output_file, "w", encoding="utf-8", errors="replace") as f:
            if result.stdout is not None:
                f.write(result.stdout)
        print(f"Saved output to {output_file}")


if __name__ == "__main__":
    main()
