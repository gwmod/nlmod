import datetime as dt
import logging
import os
import pathlib
from shutil import copyfile

import flopy
import numpy as np
import pandas as pd

from .. import util
from ..dims.time import _pd_timestamp_to_cftime

logger = logging.getLogger(__name__)


def write_and_run(sim, ds, write_ds=True, script_path=None, silent=False):
    """Write modflow files and run the model. Extra options include writing the model
    dataset to a netcdf file in the model workspace and copying the modelscript to the
    model workspace.

    Parameters
    ----------
    sim : flopy.mf6.MFSimulation or flopy.mf6.ModflowGwf
        MF6 Simulation or MF6 Groundwater Flow object.
    ds : xarray.Dataset
        dataset with model data.
    write_ds : bool, optional
        if True the model dataset is written to a NetCDF-file (.nc) in the
        model workspace the name of the .nc file is used from the attribute
        "model_name". The default is True.
    script_path : str or None, optional
        full path of the Jupyter Notebook (.ipynb) or the module (.py) with the
        modelscript. The default is None. Preferably this path does not have to
        be given manually but there is currently no good option to obtain the
        filename of a Jupyter Notebook from within the notebook itself.
    silent : bool, optional
        write and run model silently
    """
    if isinstance(sim, flopy.mf6.ModflowGwf):
        sim = sim.simulation

    if script_path is not None:
        new_script_fname = (
            f'{dt.datetime.now().strftime("%Y%m%d")}' + os.path.split(script_path)[-1]
        )
        dst = os.path.join(ds.model_ws, new_script_fname)
        logger.info(f"write script {new_script_fname} to model workspace")
        copyfile(script_path, dst)

    if write_ds:
        logger.info("write model dataset to cache")
        ds.attrs["model_dataset_written_to_disk_on"] = dt.datetime.now().strftime(
            "%Y%m%d_%H:%M:%S"
        )
        if isinstance(ds.attrs["model_ws"], pathlib.PurePath):
            ds.to_netcdf(ds.attrs["model_ws"] / f"{ds.model_name}.nc")
        else:
            ds.to_netcdf(os.path.join(ds.attrs["model_ws"], f"{ds.model_name}.nc"))

    logger.info("write modflow files to model workspace")
    sim.write_simulation(silent=silent)
    ds.attrs["model_data_written_to_disk_on"] = dt.datetime.now().strftime(
        "%Y%m%d_%H:%M:%S"
    )

    logger.info("run model")
    assert sim.run_simulation(silent=silent)[0], "Modflow run not succeeded"
    ds.attrs["model_ran_on"] = dt.datetime.now().strftime("%Y%m%d_%H:%M:%S")


def get_tdis_perioddata(ds, nstp="nstp", tsmult="tsmult"):
    """Get tdis_perioddata from ds.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with time variant model data

    Returns
    -------
    tdis_perioddata : [perlen, nstp, tsmult]
        - perlen (double) is the length of a stress period.
        - nstp (integer) is the number of time steps in a stress period.
        - tsmult (double) is the multiplier for the length of successive time
          steps. The length of a time step is calculated by multiplying the
          length of the previous time step by TSMULT. The length of the first
          time step, :math:`\\Delta t_1`, is related to PERLEN, NSTP, and
          TSMULT by the relation :math:`\\Delta t_1= perlen \frac{tsmult -
          1}{tsmult^{nstp}-1}`.
    """
    deltat = pd.to_timedelta(1, ds.time.time_units)
    if ds.time.dtype.kind == "M":
        # dtype is pandas timestamps
        perlen = [
            (pd.to_datetime(ds["time"].data[0]) - pd.to_datetime(ds.time.start))
            / deltat
        ]
        if len(ds["time"]) > 1:
            perlen.extend(np.diff(ds["time"]) / deltat)
    elif ds.time.dtype.kind == "O":
        perlen = [
            (ds["time"].data[0] - _pd_timestamp_to_cftime(pd.Timestamp(ds.time.start)))
            / deltat
        ]
        if len(ds["time"]) > 1:
            perlen.extend(np.diff(ds["time"]) / deltat)
    elif ds.time.dtype.kind in ["i", "f"]:
        perlen = [ds["time"][0]]
        perlen.extent(np.diff(ds["time"].values))

    nstp = util._get_value_from_ds_datavar(ds, "nstp", nstp, return_da=False)

    if isinstance(nstp, (int, np.integer)):
        nstp = [nstp] * len(perlen)

    tsmult = util._get_value_from_ds_datavar(ds, "tsmult", tsmult, return_da=False)

    if isinstance(tsmult, float):
        tsmult = [tsmult] * len(perlen)

    tdis_perioddata = list(zip(perlen, nstp, tsmult))

    return tdis_perioddata


def sim(ds, exe_name=None, version_tag=None, **kwargs):
    """Create sim from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data. Should have the dimension 'time' and the
        attributes: model_name, mfversion, model_ws, time_units, start,
        perlen, nstp, tsmult
    exe_name: str, optional
        path to modflow executable, default is None. If None, the path is
        obtained from the flopy metadata that respects `version_tag`. If not
        found, the executables are downloaded. Not compatible with version_tag.
    version_tag : str, default None
        GitHub release ID: for example "18.0" or "latest". If version_tag is provided,
        the most recent installation location of MODFLOW is found in flopy metadata
        that respects `version_tag`. If not found, the executables are downloaded.
        Not compatible with exe_name.

    Returns
    -------
    sim : flopy MFSimulation
        simulation object.
    """
    # start creating model
    logger.info("creating mf6 SIM")

    # Most likely exe_name was previously set with to_model_ds()
    if exe_name is not None:
        exe_name = util.get_exe_path(exe_name=exe_name, version_tag=version_tag)
    elif "exe_name" in ds.attrs:
        exe_name = util.get_exe_path(
            exe_name=ds.attrs["exe_name"], version_tag=version_tag
        )
    elif "mfversion" in ds.attrs:
        exe_name = util.get_exe_path(
            exe_name=ds.attrs["mfversion"], version_tag=version_tag
        )
    else:
        raise ValueError("No exe_name provided and no exe_name found in ds.attrs")

    # Create the Flopy simulation object
    sim = flopy.mf6.MFSimulation(
        sim_name=ds.model_name,
        exe_name=exe_name,
        version=ds.mfversion,
        sim_ws=ds.model_ws,
        **kwargs,
    )

    return sim


def tdis(ds, sim, pname="tdis", nstp="nstp", tsmult="tsmult", **kwargs):
    """Create tdis package from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data. Should have the dimension 'time' and the
        attributes: time_units, start, perlen, nstp, tsmult
    sim : flopy MFSimulation
        simulation object.
    pname : str, optional
        package name
    **kwargs
        passed on to flopy.mft.ModflowTdis

    Returns
    -------
    dis : flopy TDis
        tdis object.
    """
    # start creating model
    logger.info("creating mf6 TDIS")

    tdis_perioddata = get_tdis_perioddata(ds, nstp=nstp, tsmult=tsmult)

    # Create the Flopy temporal discretization object
    tdis = flopy.mf6.ModflowTdis(
        sim,
        pname=pname,
        time_units=ds.time.time_units,
        nper=len(ds.time),
        start_date_time=pd.Timestamp(ds.time.start).isoformat(),
        perioddata=tdis_perioddata,
        **kwargs,
    )

    return tdis


def ims(sim, complexity="MODERATE", pname="ims", model=None, **kwargs):
    """Create implicit model solution (IMS) package.

    Parameters
    ----------
    sim : flopy MFSimulation
        simulation object.
    complexity : str, optional
        solver complexity for default settings. The default is "MODERATE".
    pname : str, optional
        package name

    Returns
    -------
    ims : flopy ModflowIms
        ims object.
    """
    logger.info("creating mf6 IMS")

    print_option = kwargs.pop("print_option", "summary")

    # Create the Flopy iterative model solver (ims) Package object
    ims = flopy.mf6.ModflowIms(
        sim,
        pname=pname,
        print_option=print_option,
        complexity=complexity,
        **kwargs,
    )
    if model is not None:
        register_solution_package(sim, model, ims)
    return ims


def ems(sim, pname="ems", model=None, **kwargs):
    """Create explicit model solution (EMS) package.

    Parameters
    ----------
    sim : flopy MFSimulation
        simulation object.
    pname : str, optional
        package name

    """
    ems = flopy.mf6.ModflowEms(sim, pname=pname, **kwargs)
    if model is not None:
        register_solution_package(sim, model, ems)
    return ems


def register_ims_package(sim, model, ims):
    sim.register_ims_package(ims, [model.name])


def register_solution_package(sim, model, solver):
    sim.register_solution_package(solver, [model.name])
