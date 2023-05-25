from importlib import metadata
from platform import python_version

__version__ = "0.5.4b"


def show_versions() -> None:
    """Method to print the version of dependencies."""

    msg = (
        f"Python version: {python_version()}\n"
        f"NumPy version: {metadata.version('numpy')}\n"
        f"Xarray version: {metadata.version('xarray')}\n"
        f"Matplotlib version: {metadata.version('matplotlib')}\n"
        f"Flopy version: {metadata.version('flopy')}\n"
    )

    msg += f"\nnlmod version: {__version__}"

    return print(msg)
