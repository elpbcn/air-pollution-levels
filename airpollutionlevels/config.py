from pathlib import Path

def get_project_root():
    """
    Returns the project root directory assuming this script is within the project structure.
    """
    # Get the current file's directory (this would be the config.py file location)
    current_dir = Path(__file__).resolve().parent

    # Traverse up the directory tree until you find a known root marker (e.g., the directory that contains the Makefile)
    while not (current_dir / 'Makefile').exists():
        if current_dir == current_dir.parent:
            raise FileNotFoundError("Project root with a Makefile not found.")
        current_dir = current_dir.parent

    return current_dir

def resolve_path(relative_path):
    """
    Resolves a relative path to an absolute path based on the project root.
    """
    return get_project_root() / relative_path
