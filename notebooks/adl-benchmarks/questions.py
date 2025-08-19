from pathlib import Path
import re
from typing import List, Optional


def find_readme():
    readme_path = Path(__file__).parent / "README.md"
    if not readme_path.exists():
        raise FileNotFoundError(f"README.md not found in {readme_path.parent}")
    return readme_path


def extract_questions(readme_path: Optional[Path] = None) -> List[str]:
    """
    Extracts numbered questions from the 'Questions' section of a README.md file.

    Args:
        readme_path (Optional[Path]): Path to the README.md file. If None, attempts to locate
                                      README.md in the current directory.

    Returns:
        List[str]: A list of questions found in the 'Questions' section.

    Raises:
        FileNotFoundError: If README.md cannot be found.
        ValueError: If the 'Questions' section is not present in README.md.
    """
    if readme_path is None:
        readme_path = find_readme()

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
