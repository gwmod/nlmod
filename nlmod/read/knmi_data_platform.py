import logging
import os
import re
import tarfile
from io import FileIO
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, Union
from zipfile import ZipFile

import requests
import xarray as xr
from numpy import arange, array, ndarray
from pandas import Timedelta, Timestamp
from tqdm import tqdm

logger = logging.getLogger(__name__)

# base_url = "https://api.dataplatform.knmi.nl/dataset-content/v1/datasets"
base_url = "https://api.dataplatform.knmi.nl/open-data/v1"


class KNMIDataPlatformError(Exception):
    """Custom exception for KNMI Data Platform errors."""


class MultipleDatasetsFound(Exception):
    """Custom exception for multiple datasets found in a file."""


def get_anonymous_api_key() -> Union[str, None]:
    """Get anonymous API Key from KNMI data platform."""
    try:
        url = "https://developer.dataplatform.knmi.nl/open-data-api#token"
        webpage = requests.get(url, timeout=120)  # get webpage
        api_key = (
            webpage.text.split("</code></pre>")[0].split("<pre><code>")[-1].strip()
        )  # obtain apikey from codeblock on webpage
        if len(api_key) != 120:
            msg = (
                f"Could not obtain API Key from {url}, trying API "
                f"Key from memory. Found API Key = {api_key}"
            )
            logger.error(msg)
            raise ValueError(msg)
        logger.info(f"Retrieved anonymous API Key from {url}")
        return api_key
    except Exception as exc:
        api_key_memory_date = "2025-07-01"
        if Timestamp.today() < Timestamp(api_key_memory_date):
            logger.info(
                "Retrieved anonymous API Key (available till"
                f" {api_key_memory_date}) from memory"
            )
            api_key = (
                "eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6ImE1OGI5N"
                "GZmMDY5NDRhZDNhZjFkMDBmNDBmNTQyNjBkIiwiaCI6Im11cm11cjEyOCJ9"
            )
            return api_key
        else:
            logger.error(
                f"Could not retrieve anonymous API Key from {url}, please"
                " create your own at https://developer.dataplatform.knmi.nl/"
            )
            raise exc


def get_list_of_files(
    dataset_name: str,
    dataset_version: str,
    api_key: Optional[str] = None,
    max_keys: int = 500,
    start_after_filename: Optional[str] = None,
    timeout: int = 120,
) -> List[str]:
    """Download list of files from KNMI data platform."""
    if api_key is None:
        api_key = get_anonymous_api_key()
    files = []
    is_trucated = True
    while is_trucated:
        url = f"{base_url}/datasets/{dataset_name}/versions/{dataset_version}/files"
        # r = requests.get(url, headers={"Authorization": api_key}, timeout=timeout)
        params = {"maxKeys": f"{max_keys}"}
        if start_after_filename is not None:
            params["startAfterFilename"] = start_after_filename
        logger.debug(f"Request to {url=} with {params=}")
        r = requests.get(
            url, params=params, headers={"Authorization": api_key}, timeout=timeout
        )
        rjson = r.json()
        if "error" in rjson:
            raise KNMIDataPlatformError(f"Error in response: {rjson['error']}")
        files.extend([x["filename"] for x in rjson["files"]])
        is_trucated = rjson["isTruncated"]
        start_after_filename = files[-1]
        logger.debug(f"Listed files untill {start_after_filename}")
    return files


def download_file(
    dataset_name: str,
    dataset_version: str,
    fname: str,
    dirname: str = ".",
    api_key: Optional[str] = None,
    timeout: int = 120,
) -> None:
    """Download file from KNMI data platform."""
    if api_key is None:
        api_key = get_anonymous_api_key()
    url = (
        f"{base_url}/datasets/{dataset_name}/versions/"
        f"{dataset_version}/files/{fname}/url"
    )
    r = requests.get(url, headers={"Authorization": api_key}, timeout=timeout)
    rjson = r.json()
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    logger.info(f"Download {fname} to {dirname}")
    fname = os.path.join(dirname, fname)
    if "temporaryDownloadUrl" not in rjson:
        raise KNMIDataPlatformError(f"{fname} not found")
    if "error" in rjson:
        raise KNMIDataPlatformError(f"Error in response: {rjson['error']}")
    with requests.get(rjson["temporaryDownloadUrl"], stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(fname, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def download_files(
    dataset_name: str,
    dataset_version: str,
    fnames: List[str],
    dirname: str = ".",
    api_key: Optional[str] = None,
    timeout: int = 120,
) -> None:
    """Download multiple files from KNMI data platform."""
    for fname in tqdm(fnames):
        download_file(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            fname=fname,
            dirname=dirname,
            api_key=api_key,
            timeout=timeout,
        )


def read_nc(fo: Union[str, FileIO], **kwargs: dict) -> xr.Dataset:
    """Read netcdf (.nc) file to xarray Dataset."""
    # could help to provide argument: engine="h5netcdf"
    return xr.open_dataset(fo, **kwargs)


def get_timestamp_from_fname(fname: str) -> Union[Timestamp, None]:
    """Get the Timestamp from a filename (with assumptions about the formatting)."""
    datestr = re.search("(_[0-9]{12})", fname)  # assumes YYYYMMDDHHMM
    if datestr is not None:
        match = datestr.group(0).replace("_", "")
        year = int(match[0:4])
        month = int(match[4:6])
        day = int(match[6:8])
        hour = int(match[8:10])
        minute = int(match[8:10])
        if hour == 24:
            dtime = Timestamp(
                year=year, month=month, day=day, hour=0, minute=minute
            ) + Timedelta(days=1)
        else:
            dtime = Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)
        return dtime
    else:
        raise FileNotFoundError(
            "Could not find filename with timestamp formatted as YYYYMMDDHHMM"
        )


def add_h5_meta(meta: Dict[str, Any], h5obj: Any, orig_ky: str = "") -> Dict[str, Any]:
    """Read metadata from hdf5 (.h5) file and add to existing metadata dictionary."""

    def cleanup(val: Any) -> Any:
        if isinstance(val, (ndarray, list)):
            if len(val) == 1:
                val = val[0]

        if isinstance(val, (bytes, bytearray)):
            val = str(val, encoding="utf-8")

        return val

    if hasattr(h5obj, "attrs"):
        attrs = h5obj.attrs
        submeta = {f"{orig_ky}/{ky}": cleanup(val) for ky, val in attrs.items()}
        meta.update(submeta)

    return meta


def read_h5_contents(h5fo: FileIO) -> Tuple[ndarray, Dict[str, Any]]:
    """Read contents from a hdf5 (.h5) file."""
    from h5py import Dataset as h5Dataset

    data = None
    meta = {}
    for ky in h5fo:
        group = h5fo[ky]
        meta = add_h5_meta(meta, group, f"{ky}")
        for gky in group:
            member = group[gky]
            meta = add_h5_meta(meta, member, f"{ky}/{gky}")
            if isinstance(member, h5Dataset):
                if data is None:
                    data = member[:]
                else:
                    raise MultipleDatasetsFound("h5 contains multiple datasets")
    return data, meta


def read_h5(fo: Union[str, FileIO]) -> xr.Dataset:
    """Read hdf5 (.h5) file to xarray Dataset."""
    from h5py import File as h5File

    with h5File(fo) as h5fo:
        data, meta = read_h5_contents(h5fo)

    cols = meta["geographic/geo_number_columns"]
    dx = meta["geographic/geo_pixel_size_x"]
    rows = meta["geographic/geo_number_rows"]
    dy = meta["geographic/geo_pixel_size_y"]
    x = arange(0 + dx / 2, cols + dx / 2, dx)
    y = arange(rows + dy / 2, 0 + dy / 2, dy)
    t = Timestamp(meta["overview/product_datetime_start"])

    ds = xr.Dataset(
        data_vars={"data": (["y", "x"], array(data, dtype=float))},
        coords={"x": x, "y": y, "time": t},
        attrs=meta,
    )
    return ds


def read_grib(
    fo: Union[str, FileIO], filter_by_keys=None, **kwargs: dict
) -> xr.Dataset:
    """Read GRIB file to xarray Dataset."""
    if kwargs is None:
        kwargs = {}

    if filter_by_keys is not None:
        if "backend_kwargs" not in kwargs:
            kwargs["backend_kwargs"] = {}
        kwargs["backend_kwargs"]["filter_by_keys"] = filter_by_keys
        if "errors" not in kwargs["backend_kwargs"]:
            kwargs["backend_kwargs"]["errors"] = "ignore"

    return xr.open_dataset(fo, engine="cfgrib", **kwargs)


def read_dataset_from_zip(
    fname: str, hour: Optional[int] = None, **kwargs: dict
) -> xr.Dataset:
    """Read KNMI data platfrom .zip file to xarray Dataset."""
    if fname.endswith(".zip"):
        with ZipFile(fname) as zipfo:
            fnames = sorted([x for x in zipfo.namelist() if not x.endswith("/")])
            ds = read_dataset(fnames=fnames, zipfo=zipfo, **kwargs)

    elif fname.endswith(".tar"):
        with tarfile.open(fname) as tarfo:
            tempdir = TemporaryDirectory()
            logger.info(f"Created temporary dir {tempdir}")
            tarfo.extractall(tempdir.name)
            fnames = sorted(
                [
                    os.path.join(tempdir.name, x)
                    for x in tarfo.getnames()
                    if not x.endswith("/")
                ]
            )
            ds = read_dataset(fnames=fnames, zipfo=tarfo, hour=hour, **kwargs)
    return ds


def read_dataset(
    fnames: List[str],
    zipfo: Union[None, ZipFile, tarfile.TarFile] = None,
    hour: Optional[int] = None,
    **kwargs: dict,
) -> xr.Dataset:
    """Read xarray dataset from different file types; .nc, .h5 or grib file."""
    if hour is not None:
        if hour == 24:
            hour = 0
        fnames = [x for x in fnames if get_timestamp_from_fname(x).hour == hour]

    data = []
    for file in tqdm(fnames):
        if zipfo is not None:
            if isinstance(zipfo, ZipFile):
                fo = zipfo.open(file)
        else:
            fo = file
        if file.endswith(".nc"):
            data.append(read_nc(fo, **kwargs))
        elif file.endswith(".h5"):
            data.append(read_h5(fo, **kwargs))
        elif "_GB" in file:
            if isinstance(zipfo, tarfile.TarFile):
                # memb = zipfo.getmember(file)
                # fo = zipfo.extractfile(memb)
                # yields TypeError: 'ExFileObject' object is not subscriptable
                # alternative is to unpack in termporary directory
                data.append(read_grib(file, **kwargs))
            elif isinstance(zipfo, ZipFile):
                data.append(read_grib(fo, **kwargs))
        else:
            raise ValueError(f"Can't read/handle file {file}")

    return xr.concat(data, dim="time")
