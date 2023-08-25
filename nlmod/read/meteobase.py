import re
from enum import Enum
from io import FileIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from zipfile import ZipFile

import numpy as np
from pandas import Timestamp
from xarray import DataArray


class MeteobaseType(Enum):
    """Enum class to couple folder names to observation type (from in
    LEESMIJ.txt)"""

    NEERSLAG = "Neerslagradargegevens in Arc/Info-formaat."
    MAKKINK = "Verdampingsgegevens volgens Makkink."
    PENMAN = "Verdampingsgegevens volgens Penman-Monteith."
    EVAPOTRANSPIRATIE = "Actuele evapotranspiratie volgens SATDATA 3.0."
    VERDAMPINGSTEKORT = "Verdampingstekort (Epot - Eact) volgens SATDATA 3.0."


def read_leesmij(fo: FileIO) -> Dict[str, Dict[str, str]]:
    """Read LEESMIJ.TXT file.

    Parameters
    ----------
    fo : FileIO
        File object

    Returns
    -------
    Dict[str, Dict[str, str]]
        Dicionary with metadata per observation type
    """
    meta = {}  # meta dict
    submeta = {}  # 1 meta dict per gegevens
    line = str(fo.readline(), encoding="utf-8")
    while line:
        if any(x for x in [e.value for e in MeteobaseType] if x in line):
            mtype = line.strip()
            submeta["type"] = mtype
            meta_idx = MeteobaseType(mtype).name
        elif ":" in line:  # regel met metadata
            l1, l2 = line.split(":")
            if "coordinaat" in l1:
                submeta[l1] = float(l2.strip())
            else:
                submeta[l1] = l2.strip()
        elif len(line) == 2:  # lege regel
            meta[meta_idx] = submeta  # sla submeta op in meta
            submeta = {}
        line = str(fo.readline(), encoding="utf-8")
    return meta


def get_timestamp_from_fname(fname: str) -> Timestamp:
    """Get the Timestamp from a filename (with some assumptions about the
    formatting)"""
    datestr = re.search("([0-9]{8})", fname)  # assumes YYYYMMDD
    if datestr is not None:
        match = datestr.group(0)
        year = int(match[0:4])
        month = int(match[4:6])
        day = int(match[6:8])

    hour = 0
    fname_wo_date = fname.replace(match, "")
    hourstr = re.search("(_[0-9]{2})", fname_wo_date)  # assumes _HH
    if hourstr is not None:
        match = hourstr.group(0)
        hour = int(match.replace("_", ""))

    dtime = Timestamp(year=year, month=month, day=day, hour=hour)
    return dtime


def read_ascii(fo: FileIO) -> Union[np.ndarray, dict]:
    """Read Esri ASCII raster format file.

    Parameters
    ----------
    fo : FileIO
        File object

    Returns
    -------
    Union[np.ndarray, dict]
        Numpy array with data and header meta
    """
    ascii_header_keys = [
        "ncols",
        "nrows",
        "nodata_value",
        "xllcorner",
        "yllcorner",
        "cellsize",
        "xllcenter",
        "yllcenter",
    ]

    # read file
    lines = fo.readlines()

    # extract header
    meta = {}
    line_cnt = 0
    for line in lines:
        linestr = str(line, encoding="utf-8").lower()
        if not any((x for x in ascii_header_keys if x in str(linestr))):
            break
        l1, l2 = linestr.split()
        if l1.lower() in ("ncols", "nrows", "nodata_value"):
            meta[l1] = int(l2)
        elif l1.lower() in (
            "xllcorner",
            "yllcorner",
            "cellsize",
            "xllcenter",
            "yllcenter",
        ):
            meta[l1] = float(l2)
        line_cnt += 1

    # extract data
    data = np.array([x.split() for x in lines[line_cnt:]], dtype=float)

    return data, meta


def get_xy_from_ascii_meta(
    meta: Dict[str, Union[int, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the xy coordinates Esri ASCII raster format header.

    Parameters
    ----------
    meta : dict
        dictonary with the following keys and value types:
        {cellsize: int,
         nrows: int,
         ncols: int,
         xllcorner/xllcenter: float,
         yllcorner/yllcenter: float}

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple with the the x and y coordinates as numpy array
    """
    if "xllcorner" in meta.keys():
        xstart = meta["xllcorner"] + meta["cellsize"] / 2
    elif "xllcenter" in meta.keys():
        xstart = meta["xllcenter"]

    x = np.linspace(
        xstart,
        xstart + meta["cellsize"] * meta["ncols"],
        meta["ncols"],
        endpoint=False,
    )

    if "yllcorner" in meta.keys():
        ystart = meta["yllcorner"] + meta["cellsize"] / 2
    elif "yllcenter" in meta.keys():
        ystart = meta["yllcenter"]

    y = np.flip(
        np.linspace(
            ystart,
            ystart + meta["cellsize"] * meta["nrows"],
            meta["nrows"],
            endpoint=True,
        )
    )
    return x, y


def read_meteobase_ascii(
    zfile: ZipFile, foldername: str, meta: Dict[str, str], replace_na: bool = True
) -> DataArray:
    """Read list of .asc files in a meteobase zipfile.

    Parameters
    ----------
    zfile : ZipFile
        meteobase zipfile
    foldername : str
        foldername where specific observation type is stored
    meta : Dict[str, str]
        relevant metadata for DataArray
    replace_na : bool
        replace nodata_value with numpy.nan

    Returns
    -------
    DataArray
    """
    fnames = [
        x
        for x in zfile.namelist()
        if f"{foldername}/" in x and x.upper().endswith(".ASC")
    ]
    if meta["Bestandsformaat"] == ".ASC (Arc/Info-raster)":
        times = []
        data_array = None
        for i, fname in enumerate(fnames):
            with zfile.open(fname) as fo:
                data, ascii_meta = read_ascii(fo)

                if data_array is None:
                    meta.update(ascii_meta)
                    data_array = np.zeros(
                        shape=(len(fnames), ascii_meta["nrows"], ascii_meta["ncols"]),
                        dtype=float,
                    )
                data_array[i] = data

                times.append(get_timestamp_from_fname(fname))

        if "Eenheid gegevens" in meta.keys():
            meta["units"] = meta["Eenheid gegevens"]

        if "nodata_value" in meta.keys() and replace_na:
            data_array[data_array == meta["nodata_value"]] = np.nan
            meta["nodata_value"] = str(np.nan)

        x, y = get_xy_from_ascii_meta(ascii_meta)

        da = DataArray(
            data_array,
            dims=["time", "y", "x"],
            coords={
                "time": times,
                "x": x,
                "y": y,
            },
            attrs=meta,
            name=foldername,
        )

        return da

    else:
        raise ValueError(f"Can't read bestandsformaat '{meta['Bestandsformaat']}'")


def read_meteobase(
    path: Union[Path, str],
    meteobase_type: Optional[str] = None,
    replace_na: bool = True,
) -> List[DataArray]:
    """Read Meteobase zipfile with ASCII data.

    Parameters
    ----------
    path : Union[Path,str]
        Path to meteobase .zipfile
    meteobase_type : Optional[str], optional
        Must be one of 'NEERSLAG', 'MAKKINK', 'PENMAN', 'EVAPOTRANSPIRATIE',
        'VERDAMPINGSTEKORT', by default None which reads all data from the
        zipfile.
    replace_na : bool
        replace nodata_value with numpy.nan

    Returns
    -------
    List[DataArray]
    """

    with ZipFile(Path(path)) as zfile:
        with zfile.open("LEESMIJ.TXT") as fo:
            meta = read_leesmij(fo)

        if meteobase_type is None:
            meteo_basetype = list(meta.keys())

        da_list = []
        for mb_type in meteo_basetype:
            da = read_meteobase_ascii(
                zfile, mb_type.upper(), meta[mb_type.upper()], replace_na=replace_na
            )
            if "Projectie" in meta[mb_type.upper()].keys():
                if (
                    meta[mb_type.upper()]["Projectie"]
                    == "RD new (Amersfoort, rijksdriehoekstelsel)"
                ):
                    import rioxarray  # noqa # pylint: disable=unused-import

                    da.rio.write_crs("EPSG:28992", inplace=True)

            da_list.append(da)

    return da_list
