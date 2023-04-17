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
from h5py import Dataset as h5Dataset
from h5py import File as h5File
from numpy import arange, array, ndarray
from pandas import Timedelta, Timestamp, read_html
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
) -> List[str]:
    if api_key is None:
        api_key = get_anonymous_api_key()
    # Make sure to send the API key with every HTTP request
    files = []
    is_trucated = True
    start_after_filename = None
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
    fname: str,
    dirname: str = ".",
    api_key: Optional[str] = None,
) -> None:
    if api_key is None:
        api_key = get_anonymous_api_key()
    url = (
        f"{base_url}/datasets/{dataset_name}/versions/"
        f"{dataset_version}/files/{fname}/url"
    )
    r = requests.get(url, headers={"Authorization": api_key})
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    logger.info(f"Download {fname} to {dirname}")
    fname = os.path.join(dirname, fname)
    data = r.json()
    if "temporaryDownloadUrl" not in data:
        raise (Exception(f"{fname} not found"))
    with requests.get(data["temporaryDownloadUrl"], stream=True) as r:
        r.raise_for_status()
        with open(fname, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def download_files(
    dataset_name: str,
    dataset_version: str,
    fnames: list,
    **kwargs: dict,
) -> None:
    data = []
    for fname in tqdm(fnames):
        data.append(
            download_file(
                dataset_name=dataset_name,
                dataset_version=dataset_version,
                fname=fname,
                **kwargs,
            )
        )


def read_nc_knmi(fo: Union[str, FileIO], **kwargs: dict) -> xr.Dataset:
    # could help to provide argument: engine="h5netcdf"
    return xr.open_dataset(fo, **kwargs)


def get_timestamp_from_fname(fname: str) -> Union[Timestamp, None]:
    """Get the Timestamp from a filename (with some assumptions about the formatting)"""
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
        raise Exception("Could not fine timestamp formatted as YYYYMMDDHHMM from fname")


def check_hour(fnames: List[str], hour: int) -> List[str]:
    if hour == 24:
        hour = 0
    return [x for x in fnames if get_timestamp_from_fname(x).hour == hour]


def add_h5_meta(meta: Dict[str, Any], h5obj: Any, orig_ky: str = "") -> Dict[str, Any]:
    def cleanup(val: Any) -> Any:
        if isinstance(val, (ndarray, list)):
            if len(val) == 1:
                val = val[0]

        if isinstance(val, (bytes, bytearray)):
            val = str(val, encoding="utf-8")

        return val

    if hasattr(h5obj, "attrs"):
        attrs = getattr(h5obj, "attrs")
        submeta = {f"{orig_ky}/{ky}": cleanup(val) for ky, val in attrs.items()}
        return meta | submeta
    else:
        return meta


def read_h5_contents(h5fo: h5File) -> Tuple[ndarray, Dict[str, Any]]:
    data = None
    meta = {}
    for ky in h5fo.keys():
        group = h5fo[ky]
        meta = add_h5_meta(meta, group, f"{ky}")
        for gky in group.keys():
            member = group[gky]
            meta = add_h5_meta(meta, member, f"{ky}/{gky}")
            if isinstance(member, h5Dataset):
                if data is None:
                    data = member[:]
                else:
                    raise Exception("h5 contains multiple Datasets")
    return data, meta


def read_h5_knmi(fo: Union[str, FileIO]) -> xr.Dataset:
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
        data_vars=dict(data=(["y", "x"], array(data, dtype=float))),
        coords=dict(
            x=x,
            y=y,
            time=t,
        ),
        attrs=meta,
    )

    return ds


def read_grib_knmi(
    fo: Union[str, FileIO], filter_by_keys=None, **kwargs: dict
) -> xr.Dataset:
    if kwargs is None:
        kwargs = {}

    if filter_by_keys is not None:
        if "backend_kwargs" not in kwargs.keys():
            kwargs["backend_kwargs"] = {}
        kwargs["backend_kwargs"]["filter_by_keys"] = filter_by_keys
        if "errors" not in kwargs["backend_kwargs"]:
            kwargs["backend_kwargs"]["errors"] = "ignore"

    return xr.open_dataset(fo, engine="cfgrib", **kwargs)


def read_dataset_from_zip(
    fname: str, hour: Optional[int] = None, **kwargs: dict
) -> xr.Dataset:
    if fname.endswith(".zip"):
        with ZipFile(fname) as zipfo:
            fnames = sorted([x for x in zipfo.namelist() if not x.endswith("/")])
            if hour is not None:
                fnames = check_hour(fnames=fnames, hour=hour)
            ds = get_dataset_from_zip(zipfo=zipfo, fnames=fnames, **kwargs)

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
            if hour is not None:
                fnames = check_hour(fnames, hour=hour)
            ds = get_dataset_from_zip(zipfo=tarfo, fnames=fnames, **kwargs)
    return ds


def get_dataset_from_zip(
    zipfo: Union[ZipFile, tarfile.TarFile],
    fnames: List[str],
    **kwargs: dict,
) -> xr.Dataset:
    data = []
    for file in tqdm(fnames):
        if file.endswith(".nc"):
            with zipfo.open(file) as fo:
                data.append(read_nc_knmi(fo, **kwargs))
        elif file.endswith(".h5"):
            with zipfo.open(file) as fo:
                data.append(read_h5_knmi(fo, **kwargs))
        elif "_GB" in file:
            if isinstance(zipfo, tarfile.TarFile):
                # memb = zipfo.getmember(file)
                # fo = zipfo.extractfile(memb)
                # yields TypeError: 'ExFileObject' object is not subscriptable
                # alternative is to unpack in termporary directory
                data.append(read_grib_knmi(file, **kwargs))
            elif isinstance(zipfo, ZipFile):
                with zipfo.open(file) as fo:
                    data.append(read_grib_knmi(fo, **kwargs))
        else:
            raise Exception(f"Can't read file {file}")

    return xr.concat(data, dim="time")
