import logging
from pathlib import Path
from pandas import read_csv, DataFrame
import flopy as fp
from flopy.plot.plotutil import PRT_PATHLINE_DTYPE

from ..gwf.gwf import _dis, _disv, _set_record
from ..util import _get_value_from_ds_datavar

logger = logging.getLogger(__name__)


def prt(ds, sim, modelname=None, save_flows=True, **kwargs):
    """Create particle tracking model from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data. Should have the dimension 'time' and the
        attributes: model_name, mfversion, model_ws, time_units, start,
        perlen, nstp, tsmult
    sim : flopy MFSimulation
        simulation object.
    modelname : str
        name of the particle tracking model

    Returns
    -------
    prt : flopy ModflowPrt
        particle tracking model object
    """
    # start creating model
    logger.info("creating mf6 PRT")

    if modelname is None:
        modelname = f"{ds.model_name}_prt"
    model_nam_file = f"{modelname}.nam"

    prt = fp.mf6.ModflowPrt(
        sim,
        modelname=modelname,
        model_nam_file=model_nam_file,
        save_flows=save_flows,
        **kwargs,
    )
    return prt


def dis(ds, prt, length_units="METERS", pname="dis", **kwargs):
    """Create discretisation package from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    prt : flopy ModflowPrt
        particle tracking object
    length_units : str, optional
        length unit. The default is 'METERS'.
    pname : str, optional
        package name

    Returns
    -------
    dis : flopy ModflowPrtdis
        discretisation package.
    """
    return _dis(ds, prt, length_units, pname, **kwargs)


def disv(ds, prt, length_units="METERS", pname="disv", **kwargs):
    """Create discretisation vertices package from the model dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    prt : flopy ModflowPrt
        particle tracking object
    length_units : str, optional
        length unit. The default is 'METERS'.
    pname : str, optional
        package name

    Returns
    -------
    disv : flopy ModflowPrtdisv
        disv package
    """
    return _disv(ds, prt, length_units, pname, **kwargs)


def mip(ds, prt, porosity=None, **kwargs):
    """Create model input package for particle tracking model.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data.
    prt : ModflowPrt
        modflow particle tracking object
    porosity : str, float, array
        porosity array/attribute name, value or array. The default is None.

    Returns
    -------
    mip : flopy ModflowGwtmip
        model input package
    """
    logger.info("creating mf6 MIP")

    # NOTE: attempting to look for porosity in attributes first, then data variables.
    # If both are defined, the attribute value will be used. The log message in this
    # case is not entirely correct. This is something we may need to sort out, and
    # also think about the order we do this search.
    # if isinstance(porosity, str):
    # value = None
    # else:
    # value = porosity
    # porosity = _get_value_from_ds_attr(
    #     ds, "porosity", porosity, value=value, warn=False
    # )
    if not isinstance(porosity, float):
        porosity = _get_value_from_ds_datavar(ds, "porosity", porosity)
    return fp.mf6.ModflowPrtmip(prt, pname="mip", porosity=porosity, **kwargs)


def prp(ds, prt, packagedata, perioddata, pname="prp", **kwargs):
    """Create particle release point package for particle tracking model.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data
    prt : ModflowPrt
        modflow particle tracking object
    releasepts : list
        list of tuples containing release point data

    Returns
    -------
    prp : flopy ModflowGwtprp
        particle release point package
    """
    logger.info("creating mf6 PRP")
    prp_track_file = kwargs.pop("prp_track_file", f"{ds.model_name}.prp.trk")
    prp = fp.mf6.ModflowPrtprp(
        prt,
        pname=pname,
        filename=f"{ds.model_name}_prt.prp",
        nreleasepts=len(packagedata),
        packagedata=packagedata,
        perioddata=perioddata,
        track_filerecord=[prp_track_file],
        stop_at_weak_sink=kwargs.pop("stop_at_weak_sink", False),
        boundnames=kwargs.pop("boundnames", False),
        exit_solve_tolerance=kwargs.pop("exit_solve_tolerance", 1e-5),
        extend_tracking=kwargs.pop("extend_tracking", True),
        **kwargs,
    )
    return prp


def fmi(ds, prt, packagedata=None, **kwargs):
    """Create flow model interface package for particle tracking model.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with model data
    prt : ModflowPrt
        modflow particle tracking object
    packagedata : list, optional
        list of tuples containing package name and file path to heads and budget files.
        The default is None, which will derive heads and buget file names from the model
        dataset.
    **kwargs : dict
        additional keyword arguments passed to flopy.mf6.ModflowPrtfmi()

    Returns
    -------
    fmi : flopy ModflowGwtfmi
        flow model interface package

    """
    logger.info("creating mf6 FMI")

    if packagedata is None:
        gwf_head_file = Path(ds.model_ws) / f"{ds.model_name}.hds"
        gwf_budget_file = Path(ds.model_ws) / f"{ds.model_name}.cbc"
        packagedata = [
            ("GWFHEAD", gwf_head_file.absolute().resolve()),
            ("GWFBUDGET", gwf_budget_file.absolute().resolve()),
        ]

    fmi = fp.mf6.ModflowPrtfmi(prt, packagedata=packagedata, pname="fmi", **kwargs)
    return fmi


def oc(ds, prt, save_budget=True, print_budget=False, **kwargs):
    logger.info("creating mf6 OC")

    budget_filerecord = kwargs.pop("budget_filerecord", [f"{ds.model_name}_prt.cbc"])
    track_filerecord = kwargs.pop("track_filerecord", [f"{ds.model_name}_prt.trk"])

    saverecord = _set_record(False, save_budget, output="budget")
    printrecord = _set_record(False, print_budget, output="budget")
    oc = fp.mf6.ModflowPrtoc(
        prt,
        pname="oc",
        filename=f"{ds.model_name}_prt.oc",
        saverecord=saverecord,
        printrecord=printrecord,
        budget_filerecord=budget_filerecord,
        track_filerecord=track_filerecord,
        **kwargs,
    )
    return oc


def read_pathlines(path: str | Path, icell2d: int | None = None) -> DataFrame:
    """Read PRT pathlines from (csv-)file and add particle ID.

    The columns in the pathlines file are:
    - kper: stress period number
    - kstp: time step number
    - imdl: number of the model the particle originated in
    - iprp: number of the particle release point (PRP) package the particle originated in
    - irpt: release point number
    - ilay: layer number
    - icell: cell number
    - izone: zone number
    - istatus: particle status code
        0: particle was released
        1: particle is being actively tracked
        2: particle terminated at a boundary face
        3: particle terminated in a weak sink cell
        4: unused
        5: particle terminated in a cell with no exit face
        6: particle terminated in a stop zone
        7: particle terminated in an inactive cell
        8: particle terminated immediately upon release into a dry cell
        9: particle terminated in a subcell with no exit face
    - ireason: reporting reason code (why the particle track record was saved)
        0: particle was released
        1: particle exited a cell
        2: time step ended
        3: particle terminated
        4: particle entered a weak sink cell
        5: user-specified tracking time
    - trelease: particle release time
    - t: particle tracking time
    - x: particle x coordinate
    - y: particle y coordinate
    - z: particle z coordinate
    - name: name of the particle release point

    Parameters
    ----------
    path : Path or str
        Path to the PRT pathlines file.
    icell2d : int or None
        Number of cells in the vertex grid. If provided, it will be used to
        calculate the vertex cell index. E.g. `ds.icell2d.size`

    Returns
    -------
    DataFrame
        DataFrame containing the pathlines data.
    """
    df = read_csv(path, dtype=PRT_PATHLINE_DTYPE)
    df["ilay"] -= 1
    df["icell"] -= 1
    if icell2d is not None:
        df["icell2d"] = df["icell"] - df["ilay"] * icell2d

    # identify particle ID and add as first column to the DataFrame
    pid_cols = ["imdl", "iprp", "irpt", "trelease"]
    pid = df.sort_values(pid_cols).groupby(pid_cols).ngroup()
    df.insert(0, "pid", pid)
    return df
