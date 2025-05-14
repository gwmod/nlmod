# ruff: noqa: D103
from pathlib import Path

from xarray import Dataset

from nlmod.read import knmi_data_platform

data_path = Path(__file__).parent / "data"


def test_download_multiple_nc_files() -> None:
    dataset_name = "EV24"
    dataset_version = "2"

    try:
        # list files from the start of 2025
        start_after_filename = (
            "INTER_OPER_R___EV24____L3__20250427T000000_20250428T000000_0003.nc"
        )
        files = knmi_data_platform.get_list_of_files(
            dataset_name, dataset_version, start_after_filename=start_after_filename
        )
        assert len(files) > 0, "No files found"

        # download the first file
        fnames = files[0:1]
        dirname = "download"
        knmi_data_platform.download_files(
            dataset_name, dataset_version, fnames, dirname=dirname
        )
        file = Path(dirname) / fnames[0]
        assert file.exists(), f"File {file} was not downloaded properly"

        ds = knmi_data_platform.read_nc(file)
        assert isinstance(ds, Dataset), f"The downloaded file {file} could not be read"
    except knmi_data_platform.KNMIDataPlatformError as e:
        print(f"Error in knmi_data_platform test: {e}")


def test_download_read_zip_file() -> None:
    dataset_name = "rad_nl25_rac_mfbs_24h_netcdf4"
    dataset_version = "2.0"
    try:
        # list the files
        files = knmi_data_platform.get_list_of_files(dataset_name, dataset_version)
        assert len(files) > 0, "No files found"

        # download the last file
        dirname = "download"
        fname = files[1]
        knmi_data_platform.download_file(
            dataset_name, dataset_version, fname=fname, dirname=dirname
        )
        file = Path(dirname) / fname
        assert file.exists(), f"File {file} was not downloaded properly"
    except knmi_data_platform.KNMIDataPlatformError as e:
        print(f"Error in knmi_data_platform test: {e}")


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
