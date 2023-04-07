import nlmod
from pathlib import Path

data_path = Path(__file__) / "data"


def test_read_meteobase() -> None:
    _ = nlmod.read.meteobase.read_meteobase(data_path / "Meteobase_ASCII_test.zip")
