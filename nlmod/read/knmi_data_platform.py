# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:53:17 2023

@author: Ruben
"""

import os
import requests
from typing import Optional, List
import logging
from tqdm import tqdm
import xarray as xr


logger = logging.getLogger(__name__)

api_key = "eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6IjI4ZWZlOTZkNDk2ZjQ3ZmE5YjMzNWY5NDU3NWQyMzViIiwiaCI6Im11cm11cjEyOCJ9"  # knmi
# base_url = "https://api.dataplatform.knmi.nl/dataset-content/v1/datasets"
base_url = "https://api.dataplatform.knmi.nl/open-data"


def get_list_of_files(
    dataset_name: str,
    dataset_version: str,
    max_keys: int = 500,
    start_after_filename: Optional[str] = None,
) -> List[str]:
    # Make sure to send the API key with every HTTP request
    files = []
    is_trucated = True
    while is_trucated:
        url = f"{base_url}/datasets/{dataset_name}/versions/{dataset_version}/files"
        r = requests.get(url, headers={"Authorization": api_key})
        params = {"maxKeys": f"{max_keys}"}
        if start_after_filename is not None:
            params["startAfterFilename"] = start_after_filename
        r = requests.get(url, params=params, headers={"Authorization": api_key})
        json = r.json()
        files.extend([x["filename"] for x in json["files"]])
        is_trucated = json["isTruncated"]
        start_after_filename = files[-1]
        logger.debug(f"Listed files untill {start_after_filename}")
    return files


def download_file(
    dataset_name: str,
    dataset_version: str,
    filename: str,
    dirname: str = ".",
    read: bool = True,
    hour: Optional[int] = None,
) -> None:
    url = f"{base_url}/datasets/{dataset_name}/versions/{dataset_version}/files/{filename}/url"
    r = requests.get(url, headers={"Authorization": api_key})
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    fname = os.path.join(dirname, filename)
    with requests.get(r.json()["temporaryDownloadUrl"], stream=True) as r:
        r.raise_for_status()
        with open(fname, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    logger.info(f"Successfully downloaded dataset file to {filename}")
    if read:
        if fname.endswith(".nc"):
            return xr.open_dataset(fname)
        elif fname.endswith(".zip"):
            return read_dataset_from_zip(fname, hour=hour)
        else:
            logger.warning("Unknow file type: {filename}")


def read_dataset_from_zip(fname: str, hour: Optional[int] = None) -> xr.Dataset():
    from zipfile import ZipFile

    with ZipFile(fname) as zipf:
        data = []
        for file in tqdm(zipf.namelist()):
            if hour is not None:
                if not file.endswith(f"{hour:02d}00.nc"):
                    continue
            else:
                if not file.endswith(".nc"):
                    continue
            fo = zipf.open(file)
            data.append(xr.open_dataset(fo))
        ds = xr.concat(data, dim="time").sortby("time")
    return ds
