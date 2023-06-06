from pathlib import Path

import nlmod

data_path = Path(__file__).parent / "data"


def test_read_meteobase() -> None:
    _ = nlmod.read.meteobase.read_meteobase(data_path / "Meteobase_ASCII_test.zip")
