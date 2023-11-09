import os


def get_base_path(path: str):
    base = os.getcwd()
    return os.path.join(
        base,
        path
    )
