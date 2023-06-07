import os
from pathlib import Path

from nlmod.read import knmi_data_platform

data_path = Path(__file__).parent / "data"


def test_download_multiple_nc_files() -> None:
    dataset_name = "EV24"
    dataset_version = "2"

    # list files from the start of 2023
    start_after_filename = (
        "INTER_OPER_R___EV24____L3__20221231T000000_20230101T000000_0003.nc"
    )
    files = knmi_data_platform.get_list_of_files(
        dataset_name, dataset_version, start_after_filename=start_after_filename
    )

    # download the last 10 files
    fnames = files[-10:]
    dirname = "download"
    knmi_data_platform.download_files(
        dataset_name, dataset_version, files[-10:], dirname=dirname
    )

    ds = knmi_data_platform.read_nc(os.path.join(dirname, fnames[0]))

    # plot the mean evaporation
    ds["prediction"].mean("time").plot()


def test_download_read_zip_file() -> None:
    dataset_name = "rad_nl25_rac_mfbs_24h_netcdf4"
    dataset_version = "2.0"

    # list the files
    files = knmi_data_platform.get_list_of_files(dataset_name, dataset_version)

    # download the last file
    dirname = "download"
    fname = files[-1]
    knmi_data_platform.download_file(
        dataset_name, dataset_version, fname=fname, dirname=dirname
    )


def test_read_zip_file() -> None:
    fname = data_path / "KNMI_Data_Platform_NETCDF.zip"
    _ = knmi_data_platform.read_dataset_from_zip(str(fname), hour=24)


def test_read_h5() -> None:
    fname = data_path / "KNMI_Data_Platform_H5.zip"
    _ = knmi_data_platform.read_dataset_from_zip(str(fname))


def test_read_grib() -> None:
    fname = data_path / "KNMI_Data_Platform_GRIB.tar"
    _ = knmi_data_platform.read_dataset_from_zip(
        str(fname),
        filter_by_keys={"stepType": "instant", "typeOfLevel": "heightAboveGround"},
    )
