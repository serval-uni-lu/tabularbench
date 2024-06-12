from pathlib import Path


def parent_exists(path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path
