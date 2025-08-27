class IndentedDetailsBlock:
    """
    Context manager to emit a <details> block with a summary and an
    indented <div> for Markdown/HTML output.
    Usage:
            with IndentedDetailsBlock(fh, "Summary text") as f:
                    f.write("Indented content\n")
    """

    def __init__(self, file_handle, summary):
        self.file_handle = file_handle
        self.summary = summary

    def __enter__(self):
        self.file_handle.write(
            f'<details><summary>{self.summary}</summary>\n<div style="margin-left: 1em;">\n\n'
        )
        return self.file_handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file_handle.write("\n</div></details>\n")
