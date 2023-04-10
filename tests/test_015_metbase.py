import os

import nlmod

metbase_path = os.path.join("data", "Meteobase_ASCII_test.zip")


def test_read_meteobase() -> None:
    _ = nlmod.read.meteobase.read_meteobase(metbase_path)
