# %%
import os
from glob import glob

import nbformat
import nlmod
from nbconvert.preprocessors import (
    ClearOutputPreprocessor,
    ClearMetadataPreprocessor,
)

logger = nlmod.util.get_color_logger("INFO")

# %% get a list of all notebooks
pathnames = [
    "data_sources",
    "examples",
    "utilities",
    "workflows",
    "advanced_stress_packages",
]
for pathname in pathnames:
    notebook_list = glob(os.path.join(pathname, "*.ipynb"))

    # %% clear output and metadata of all notebooks
    clear_output = ClearOutputPreprocessor()
    clear_metadata = ClearMetadataPreprocessor(clear_cell_metadata=False)

    for notebook in notebook_list:
        logger.info(f"Clearing output of notebook {notebook}")
        with open(notebook, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # run nbconvert preprocessors to clear outputs and metadata
        clear_output.preprocess(nb, {})
        clear_metadata.preprocess(nb, {})

        with open(notebook, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
