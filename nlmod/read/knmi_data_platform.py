import logging
import os
from typing import List, Optional
from zipfile import ZipFile

import requests
import xarray as xr
from pandas import Timestamp, read_html
from tqdm import tqdm

logger = logging.getLogger(__name__)

# base_url = "https://api.dataplatform.knmi.nl/dataset-content/v1/datasets"
base_url = "https://api.dataplatform.knmi.nl/open-data"


def get_anonymous_api_key() -> str:
    try:
        url = "https://developer.dataplatform.knmi.nl/get-started"
        tables = read_html(url)  # get all tables from url
        for table in tables:
            for coln in table.columns:
                if "KEY" in coln.upper():  # look for columns with key
                    api_key_str = table.iloc[0].loc[
                        coln
                    ]  # get entry with key (first row)
                    api_key = max(
                        api_key_str.split(), key=len
                    )  # get key base on str length
                    logger.info(f"Retrieved anonymous API Key from {url}")
                    return api_key
    except Exception as exc:
        if Timestamp.today() < Timestamp("2023-07-01"):
            logger.info("Retrieved anonymous API Key from memory")
            api_key = (
                "eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6IjI4ZWZl"
                "OTZkNDk2ZjQ3ZmE5YjMzNWY5NDU3NWQyMzViIiwiaCI6Im11cm11cjEyOCJ9"
            )
            return api_key
        else:
            logger.error(
                f"Could not retrieve anonymous API Key from {url}, please"
                " create your own at https://api.dataplatform.knmi.nl/"
            )
            raise exc


def get_list_of_files(
    dataset_name: str,
    dataset_version: str,
    api_key: Optional[str] = None,
    max_keys: int = 500,
    start_after_filename: Optional[str] = None,
) -> List[str]:
    if api_key is None:
        api_key = get_anonymous_api_key()
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
    api_key: Optional[str] = None,
    read: bool = True,
    hour: Optional[int] = None,
) -> None:
    if api_key is None:
        api_key = get_anonymous_api_key()
    url = (
        f"{base_url}/datasets/{dataset_name}/versions/"
        f"{dataset_version}/files/{filename}/url"
    )
    r = requests.get(url, headers={"Authorization": api_key})
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    logger.info("Download {filename} to {dirname}")
    fname = os.path.join(dirname, filename)
    data = r.json()
    if "temporaryDownloadUrl" not in data:
        raise (Exception(f"{filename} not found"))
    with requests.get(data["temporaryDownloadUrl"], stream=True) as r:
        r.raise_for_status()
        with open(fname, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    if read:
        if fname.endswith(".nc"):
            return xr.open_dataset(fname)
        elif fname.endswith(".zip"):
            return read_dataset_from_zip(fname, hour=hour)
        else:
            logger.warning("Unknow file type: {filename}")


def download_files(
    dataset_name: str,
    dataset_version: str,
    filenames: list,
    read: bool = True,
    **kwargs: dict,
) -> xr.Dataset:
    data = []
    for filename in tqdm(filenames):
        data.append(
            download_file(
                dataset_name=dataset_name,
                dataset_version=dataset_version,
                filename=filename,
                read=read,
                **kwargs,
            )
        )

    if read:
        return xr.concat(data, dim="time")


def read_dataset_from_zip(fname: str, hour: Optional[int] = None) -> xr.Dataset:
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
