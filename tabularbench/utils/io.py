def parent_exists(path: str) -> str:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path
