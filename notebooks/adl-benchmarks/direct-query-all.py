import re
import subprocess
import sys
from pathlib import Path


def find_readme():
    # Always use README.md in the same directory as this script
    readme_path = Path(__file__).parent / "README.md"
    if not readme_path.exists():
        raise FileNotFoundError(f"README.md not found in {readme_path.parent}")
    return readme_path


def extract_questions(readme_path):
    with open(readme_path, encoding="utf-8") as f:
        content = f.read()
    # Find the Questions section
    match = re.search(r"## Questions(.*?)(?:##|$)", content, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError("Questions section not found in README.md")
    questions_section = match.group(1)
    # Find numbered questions (lines starting with 1. ...)
    questions = re.findall(r"^1\. (.+)", questions_section, re.MULTILINE)
    return questions


def main():
    readme_path = find_readme()
    questions = extract_questions(readme_path)
    script_path = Path(__file__).parent / "direct-query.py"
    for i, q in enumerate(questions, 1):
        output_file = Path.cwd() / f"direct-question-{i:02d}.md"
        print(f"Running question {i:02d}...")
        # Run direct-query.py with the question and --models all

        result = subprocess.run(
            [sys.executable, str(script_path), q, "--models", "all"],
            capture_output=True,
            text=True,
        )
        # Save output to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result.stdout)
        print(f"Saved output to {output_file}")


if __name__ == "__main__":
    main()
