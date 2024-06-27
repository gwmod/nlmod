# %%
import os
import time
from glob import glob

import nbformat
import numpy as np
from nbconvert.preprocessors import ExecutePreprocessor

import nlmod

logger = nlmod.util.get_color_logger("INFO")

# %% get a list of all notebooks
notebook_list = glob("*.ipynb")

# %% run all notebooks
ep = ExecutePreprocessor()
elapsed_time = {}
exceptions = {}
for notebook in notebook_list:
    # set nb to None, to make sure we do not overwrite a notebook with data from another
    nb = None
    # get the start time
    logger.info(f"Running {notebook}")
    start_time = time.time()
    try:
        with open(notebook) as notebook_file:
            nb = nbformat.read(notebook_file, as_version=4)
            ep.preprocess(nb)
        elapsed_time[notebook] = time.time() - start_time
        seconds = int(np.round(elapsed_time[notebook]))
        logger.info(f"Running {notebook} succeeded in {seconds} seconds")
    except Exception as exception:
        logger.error(f"Running notebook failed: {notebook}")
        elapsed_time[notebook] = np.nan
        exceptions[notebook] = exception

    # save results in notebook
    with open(notebook, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


# %% clear output and metadata of all notebooks
for notebook in notebook_list:
    logger.info(f"Clearing output of notebook {notebook}")
    os.system(
        f"jupyter nbconvert --clear-output --inplace --ClearMetadataPreprocessor.enabled=True {notebook}"
    )

# %%
