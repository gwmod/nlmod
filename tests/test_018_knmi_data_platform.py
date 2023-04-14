from nlmod.read import knmi_data_platform


def test_download_multiple_nc_files():
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
    ds = knmi_data_platform.download_files(
        dataset_name, dataset_version, files[-10:], dirname="download"
    )

    # plot the mean evaporation
    ds["prediction"].mean("time").plot()


def test_download_zip_file():
    dataset_name = "rad_nl25_rac_mfbs_24h_netcdf4"
    dataset_version = "2.0"

    # list the files
    files = knmi_data_platform.get_list_of_files(dataset_name, dataset_version)

    # download the last file and only read the last hour of every day
    # as the data represents the precipitation in the last 24 hours
    ds = knmi_data_platform.download_file(
        dataset_name, dataset_version, files[-1], hour=24, dirname="download"
    )

    # plot the mean precipitation
    ds["image1_image_data"].mean("time").plot(size=10)
