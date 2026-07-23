import datetime as dt
import logging
import os
import pathlib
from shutil import copyfile

import flopy
import numpy as np
import pandas as pd
import xarray as xr

from .. import util
from ..dims.grid import get_idomain, modelgrid_from_ds
from ..dims.time import get_perlen

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
            f"{dt.datetime.now().strftime('%Y%m%d')}_" + os.path.split(script_path)[-1]
        )
        dst = os.path.join(ds.model_ws, new_script_fname)
        logger.info(f"write script {new_script_fname} to model workspace")
        copyfile(script_path, dst)

    if write_ds:
        logger.info("write model dataset to cache")
        for attr, value in ds.attrs.items():
            if isinstance(value, pathlib.PurePath):
                ds.attrs[attr] = str(value)

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
    perlen = get_perlen(ds)

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
        sim_ws=kwargs.pop("sim_ws", ds.model_ws),
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


def get_parent_child_exchange_gdf(ds_parent, ds_child, boundnames="angldegx"):
    gdf_parent = modelgrid_from_ds(ds_parent).to_geodataframe()
    gdf_child = modelgrid_from_ds(ds_child).to_geodataframe()

    outer_ring = gdf_parent.loc[gdf_parent.touches(gdf_child.union_all())]
    candidates = outer_ring.sjoin(gdf_child, how="inner", predicate="intersects")
    candidates["shared_face_geom"] = candidates.apply(
        lambda row: row["geometry"].boundary.intersection(
            gdf_child.loc[row["index_right"], "geometry"]
        ),
        axis=1,
    )
    shared_faces = candidates.loc[
        candidates["shared_face_geom"].geom_type.isin(["LineString", "MultiLineString"])
    ].copy()
    shared_faces["child_geom"] = gdf_child.loc[
        shared_faces["index_right"], "geometry"
    ].values

    # compute exchange variables
    shared_faces["hwva"] = shared_faces.shared_face_geom.length
    shared_faces["cl1"] = shared_faces.apply(
        lambda row: row["geometry"].centroid.distance(row["shared_face_geom"]), axis=1
    )
    shared_faces["cl2"] = shared_faces.apply(
        lambda row: row["shared_face_geom"].distance(row["child_geom"].centroid), axis=1
    )

    shared_faces = shared_faces.reset_index(names="parent_cellid").rename(
        columns={"index_right": "child_cellid", "geometry": "parent_geom"}
    )
    dx = shared_faces["parent_geom"].centroid.x - shared_faces["child_geom"].centroid.x
    dy = shared_faces["parent_geom"].centroid.y - shared_faces["child_geom"].centroid.y

    shared_faces["angldegx"] = np.degrees(np.atan2(dy, dx)) % 360

    shared_faces = shared_faces.loc[
        :,
        [
            "parent_cellid",
            "parent_geom",
            "shared_face_geom",
            "child_cellid",
            "child_geom",
            "cl1",
            "cl2",
            "hwva",
            "angldegx",
        ],
    ]

    if boundnames == "angldegx":

        def angle_to_compass(angle):
            if (315 <= angle < 360) or (0 <= angle < 45):
                return "E"
            elif 45 <= angle < 135:
                return "N"
            elif 135 <= angle < 225:
                return "W"
            elif 225 <= angle < 315:
                return "S"

        shared_faces["boundnames"] = shared_faces["angldegx"].apply(angle_to_compass)
        shared_faces = shared_faces.sort_values(by="angldegx")
    elif boundnames:
        shared_faces["boundnames"] = boundnames
        shared_faces = shared_faces.sort_values(by="boundnames")

    return shared_faces


def gwfgwf(
    sim, ds_parent, ds_child, exgtype="GWF6-GWF6", boundnames="angldegx", **kwargs
):
    """Create GWF-GWF exchange package from the model datasets.

    Parameters
    ----------
    sim : flopy MFSimulation
        simulation object.
    ds_parent : xarray.Dataset
        dataset with model data for the parent model.
    ds_child : xarray.Dataset
        dataset with model data for the child model.
    exgtype : str, optional
        exchange type. The default is "GWF6-GWF6".
    boundnames : str, optional
        name of the boundary condition. The default is "angldegx". If None, no
        boundname is added to the exchangedata.
    **kwargs
        passed on to flopy.mf6.ModflowGwfgwf

    Returns
    -------
    gwfgwf : flopy ModflowGwfgwf
        gwfgwf exchange object.
    """
    # get single layer
    exch_gdf = get_parent_child_exchange_gdf(
        ds_parent,
        ds_child,
        boundnames=boundnames,
    )
    if exch_gdf.empty:
        raise ValueError("No shared faces found between parent and child model grids.")

    exch_gdf = exch_gdf.rename(
        columns={
            "parent_cellid": "cellidm1",
            "child_cellid": "cellidm2",
        }
    )
    exch_gdf["ihc"] = 1  # exchange is always horizontal
    # ensure layers are equal
    xr.testing.assert_equal(ds_parent.layer, ds_child.layer)

    idomain_parent = get_idomain(ds_parent)
    idomain_child = get_idomain(ds_child)
    exchangedata = []
    usecols = ["cellidm1", "cellidm2", "ihc", "cl1", "cl2", "hwva", "angldegx"]
    if boundnames is not None:
        usecols.append("boundnames")
    for ilay in range(ds_parent.sizes["layer"]):
        idf = exch_gdf.loc[:, usecols].copy()
        mask_active_parent = (
            idomain_parent.isel(layer=ilay, icell2d=idf["cellidm1"]) > 0
        ).values
        mask_active_child = (
            idomain_child.isel(layer=ilay, icell2d=idf["cellidm2"]) > 0
        ).values
        idf = idf.loc[mask_active_parent & mask_active_child].copy()
        idf["cellidm1"] = [(ilay, cid) for cid in idf["cellidm1"]]
        idf["cellidm2"] = [(ilay, cid) for cid in idf["cellidm2"]]
        exchangedata.append(idf)

    exchangedata = pd.concat(exchangedata, axis=0)

    return flopy.mf6.ModflowGwfgwf(
        sim,
        exgmnamea=ds_parent.model_name,
        exgmnameb=ds_child.model_name,
        exgtype=exgtype,
        nexg=len(exchangedata),
        exchangedata=exchangedata.to_records(index=False),
        auxiliary=["ANGLDEGX"],
        boundnames=True if boundnames is not None else None,
        **kwargs,
    )
