from disk_cache import diskcache_decorator
import fsspec


@diskcache_decorator(".hint_file_cache")
def load_file_content(path: str) -> str:
    """
    Load file content using built-in open for local files, with disk-backed cache.
    For remote filesystems, replace with fsspec.open or fsspec.open_files.
    """
    with fsspec.open(path, "r") as f:
        content = f.read()  # type: ignore
    return content


def load_hint_files(hint_files: list[str]) -> list[str]:
    """
    Load all hint files into a list of strings, using cache for speed.
    """
    return [load_file_content(hint_file) for hint_file in hint_files]
