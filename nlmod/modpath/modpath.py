import datetime as dt
import logging
import numbers
import os
from shutil import copyfile

import flopy
import geopandas as gpd
import pandas as pd

from .. import util
from ..dims.grid import xy_to_icell2d

logger = logging.getLogger(__name__)


def write_and_run(mpf, remove_prev_output=True, nb_path=None):
    """write modpath files and run the model.

    2 extra options:
        1. remove output of the previous run
        2. copy the modelscript (typically a Jupyter Notebook) to the model
           workspace with a timestamp.


    Parameters
    ----------
    mpf : flopy.modpath.mp7.Modpath7
        modpath model.
    model_ds : xarray.Dataset
        dataset with model data.
    remove_prev_output : bool, optional
        remove the output of a previous modpath run (if it exists)
    nb_path : str or None, optional
        full path of the Jupyter Notebook (.ipynb) with the modelscript. The
        default is None. Preferably this path does not have to be given
        manually but there is currently no good option to obtain the filename
        of a Jupyter Notebook from within the notebook itself.
    """
    if remove_prev_output:
        remove_output(mpf)

    if nb_path is not None:
        new_nb_fname = (
            f'{dt.datetime.now().strftime("%Y%m%d")}' + os.path.split(nb_path)[-1]
        )
        dst = os.path.join(mpf.model_ws, new_nb_fname)
        logger.info(f"write script {new_nb_fname} to modpath workspace")
        copyfile(nb_path, dst)

    logger.info("write modpath files to model workspace")

    # write modpath datasets
    mpf.write_input()

    # run modpath
    logger.info("run modpath model")
    assert mpf.run_model()[0], "Modpath run not succeeded"


def xy_to_nodes(xy_list, mpf, ds, layer=0):
    """convert a list of points, defined by x and y coordinates, to a list of
    nodes. A node is a unique cell in a model. The icell2d is a unique cell in
    a layer.

    Parameters
    ----------
    xy_list : list of tuples
        list with tuples with coordinates e.g. [(0,1),(2,5)].
    mpf : flopy.modpath.mp7.Modpath7
        modpath object.
    ds : xarary dataset
        model dataset.
    layer : int or list of ints, optional
        Layer number. If layer is an int all nodes are returned for that layer.
        If layer is a list the length should be the same as xy_list. The
        default is 0.

    Returns
    -------
    nodes : list of ints
        nodes numbers corresponding to the xy coordinates and layer.
    """
    if isinstance(layer, numbers.Number):
        layer = [layer] * len(xy_list)

    nodes = []
    for i, xy in enumerate(xy_list):
        icell2d = xy_to_icell2d(xy, ds)
        if mpf.ib[layer[i], icell2d] > 0:
            node = layer[i] * mpf.ib.shape[1] + icell2d
            nodes.append(node)

    return nodes


def package_to_nodes(gwf, package_name, mpf):
    """Return a list of nodes from the cells with certain boundary conditions.

    Parameters
    ----------
    gwf : flopy.mf6.mfmodel.MFModel
        Groundwater flow model.
    package_name : str
        name of the package.
    mpf : flopy.modpath.mp7.Modpath7
        modpath object.

    Raises
    ------
    TypeError
        when the modflow package has no stress period data.

    Returns
    -------
    nodes : list of ints
        node numbers corresponding to the cells with a certain boundary condition.
    """
    gwf_package = gwf.get_package(package_name)
    if not hasattr(gwf_package, "stress_period_data"):
        raise TypeError("only package with stress period data can be used")

    pkg_cid = gwf_package.stress_period_data.array[0]["cellid"]
    nodes = []
    for cid in pkg_cid:
        if mpf.ib[cid[0], cid[1]] > 0:
            node = cid[0] * mpf.ib.shape[1] + cid[1]
            nodes.append(node)

    return nodes


def layer_to_nodes(mpf, modellayer):
    """get the nodes of all cells in one ore more model layer(s).

    Parameters
    ----------
    mpf : flopy.modpath.mp7.Modpath7
        modpath object.
    modellayer : int, list or tuple
        if modellayer is an int there is one modellayer. If modellayer is a
        list or tuple there are multiple modellayers.

    Returns
    -------
    nodes : list of ints
        node numbers corresponding to all cells in certain model layer(s).
    """
    if not isinstance(modellayer, (list, tuple)):
        modellayer = [modellayer]
    nodes = []
    node = 0
    for lay in range(mpf.ib.shape[0]):
        for icell2d in range(mpf.ib.shape[1]):
            # only add specific layers
            if lay in modellayer:
                if mpf.ib[lay, icell2d] > 0:
                    nodes.append(node)
            node += 1

    return nodes


def mpf(gwf, exe_name=None, modelname=None, model_ws=None):
    """Create a modpath model from a groundwater flow model.

    Parameters
    ----------
    gwf : flopy.mf6.mfmodel.MFModel
        Groundwater flow model.
    exe_name: str, optional
        path to modpath executable, default is None, which assumes binaries
        are available in nlmod/bin directory. Binaries can be downloaded
        using `nlmod.util.download_mfbinaries()`.
    modelname: str or None, optional
        modelname of the modpath model. If None the name of the groundwaterflow
        model is used with mp7_ as a prefix. The default value is None.

    Raises
    ------
    ValueError
        if some settings in the groundwater flow model makes it impossible to
        add a modpath model.

    Returns
    -------
    mpf : flopy.modpath.mp7.Modpath7
        modpath object.
    """
    if modelname is None:
        modelname = "mp7_" + gwf.name

    if model_ws is None:
        model_ws = os.path.join(gwf.model_ws, "modpath")

    # check if the save flows parameter is set in the npf package
    npf = gwf.get_package("npf")
    if not npf.save_flows.array:
        raise ValueError(
            "the save_flows option of the npf package should be True not None"
        )

    # check if the tdis has a start_time
    if gwf.simulation.tdis.start_date_time.array is not None:
        logger.warning(
            "older versions of modpath cannot handle this, see https://github.com/MODFLOW-USGS/modpath-v7/issues/31"
        )

    # get executable
    if exe_name is None:
        exe_name = util.get_exe_path("mp7_2_002_provisional")

    # create mpf model
    mpf = flopy.modpath.Modpath7(
        modelname=modelname,
        flowmodel=gwf,
        exe_name=exe_name,
        model_ws=model_ws,
        verbose=True,
    )

    if model_ws != gwf.model_ws:
        mpf.grbdis_file = os.path.relpath(
            os.path.join(gwf.model_ws, mpf.grbdis_file), model_ws
        )
        mpf.headfilename = os.path.relpath(
            os.path.join(gwf.model_ws, mpf.headfilename), model_ws
        )
        mpf.budgetfilename = os.path.relpath(
            os.path.join(gwf.model_ws, mpf.budgetfilename), model_ws
        )
        mpf.tdis_file = os.path.relpath(
            os.path.join(gwf.model_ws, mpf.tdis_file), model_ws
        )

    return mpf


def bas(mpf, porosity=0.3, **kwargs):
    """Create the basic package for the modpath model.

    Parameters
    ----------
    mpf : flopy.modpath.mp7.Modpath7
        modpath object.
    porosity : float, optional
        porosity. The default is 0.3.

    Returns
    -------
    mpfbas : flopy.modpath.mp7bas.Modpath7Bas
        modpath bas package.
    """

    mpfbas = flopy.modpath.Modpath7Bas(mpf, porosity=porosity, **kwargs)

    return mpfbas


def remove_output(mpf):
    """Remove the output of a previous modpath run. Commonly used before
    starting a new modpath run to avoid loading the wrong data when a modpath
    run has failed.

    Parameters
    ----------
    mpf : flopy.modpath.mp7.Modpath7
        modpath object.

    Returns
    -------
    None.
    """
    mpffiles = [
        mpf.name + ".mppth",
        mpf.name + ".timeseries",
        mpf.name + ".mpend",
    ]

    # remove output
    for f in mpffiles:
        fname = os.path.join(mpf.model_ws, f)
        if os.path.exists(fname):
            os.remove(fname)
            logger.info(f"removed old version of '{f}'")
        else:
            logger.debug(f"no old version of '{f}'")


def load_pathline_data(
    mpf=None, model_ws=None, modelname=None, return_df=False, return_gdf=False
):
    """Read the pathline data from a modpath model.

    Parameters
    ----------
    mpf : flopy.modpath.mp7.Modpath7
        modpath object. If None the model_ws and modelname are used to load
        the pathline data. The default is None.
    model_ws : str or None, optional
        workspace of the modpath model. This is where modeldata is saved to.
        Only used if mpf is None. The default is None.
    modelname : str or None, optional
        name of the modpath model. Only used if mpf is None. The default is
        None.
    return_df : bool, optional
        if True a DataFrame with pathline data is returned. The default is
        False.
    return_gdf : bool, optional
        if True a GeoDataframe with pathline data is returned. The default is
        False.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    numpy.ndarray, DataFrame, GeoDataFrame
        pathline data. By default a numpy array is returned.
    """
    if mpf is None:
        if modelname is None:
            raise ValueError(
                "if no mpf model is provided a modelname should be provided to load pathline data"
            )
        fpth = os.path.join(model_ws, f"{modelname}.mppth")
    else:
        fpth = os.path.join(mpf.model_ws, mpf.name + ".mppth")

    p = flopy.utils.PathlineFile(fpth, verbose=False)
    if (not return_df) and (not return_gdf):
        return p._data
    elif return_df and (not return_gdf):
        pdf = pd.DataFrame(p._data)
        return pdf
    elif return_gdf and (not return_df):
        pdf = pd.DataFrame(p._data)
        geom = gpd.points_from_xy(pdf["x"], pdf["y"])
        pgdf = gpd.GeoDataFrame(pdf, geometry=geom)
        return pgdf
    else:
        raise ValueError(
            "'return_df' and 'return_gdf' are both True, while only one can be True"
        )


def pg_from_fdt(nodes, divisions=3):
    """Create a particle group using the FaceDataType.

    Parameters
    ----------
    nodes : list of ints
        node numbers.
    divisions : int, optional
        number of particle on each face. If divisions is 3 each cell will have
        3*3=9 particles starting at each cell face, 9*6=54 particles per cell.
        The default is 3.

    Returns
    -------
    pg : flopy.modpath.mp7particlegroup.ParticleGroupNodeTemplate
        Particle group.
    """
    logger.info(
        f"particle group with {divisions**2} particle per cell face, {6*divisions**2} particles per cell"
    )
    sd = flopy.modpath.FaceDataType(
        drape=0,
        verticaldivisions1=divisions,
        horizontaldivisions1=divisions,
        verticaldivisions2=divisions,
        horizontaldivisions2=divisions,
        verticaldivisions3=divisions,
        horizontaldivisions3=divisions,
        verticaldivisions4=divisions,
        horizontaldivisions4=divisions,
        rowdivisions5=divisions,
        columndivisions5=divisions,
        rowdivisions6=divisions,
        columndivisions6=divisions,
    )

    p = flopy.modpath.NodeParticleData(subdivisiondata=sd, nodes=nodes)

    pg = flopy.modpath.ParticleGroupNodeTemplate(particledata=p)

    return pg


def pg_from_pd(nodes, localx=0.5, localy=0.5, localz=0.5):
    """Create a particle group using the ParticleData.

    Parameters
    ----------
    nodes : list of ints
        node numbers.
    localx : float, list, tuple, or np.ndarray
        Local x-location of the particle in the cell. If a single value is
        provided all particles will have the same localx position. If
        a list, tuple, or np.ndarray is provided a localx position must
        be provided for each partloc. If localx is None, a value of
        0.5 (center of the cell) will be used (default is None).
    localy : float, list, tuple, or np.ndarray
        Local y-location of the particle in the cell. If a single value is
        provided all particles will have the same localy position. If
        a list, tuple, or np.ndarray is provided a localy position must
        be provided for each partloc. If localy is None, a value of
        0.5 (center of the cell) will be used (default is None).
    localz : float, list, tuple, or np.ndarray
        Local z-location of the particle in the cell. If a single value is
        provided all particles will have the same localz position. If
        a list, tuple, or np.ndarray is provided a localz position must
        be provided for each partloc. If localz is None, a value of
        0.5 (center of the cell) will be used (default is None). A localz
        value of 1.0 indicates the top of a cell.

    Returns
    -------
    pg : flopy.modpath.mp7particlegroup.ParticleGroup
        Particle group.
    """
    p = flopy.modpath.ParticleData(
        partlocs=nodes,
        structured=False,
        localx=localx,
        localy=localy,
        localz=localz,
    )
    pg = flopy.modpath.ParticleGroup(particledata=p)

    return pg


def sim(mpf, pg, direction="backward", gwf=None, ref_time=None, stoptime=None):
    """Create a modpath backward simulation from a particle group.

    Parameters
    ----------
    mpf : flopy.modpath.mp7.Modpath7
        modpath object.
    pg : flopy.modpath.mp7particlegroup.ParticleGroupNodeTemplate
        Particle group.
    gwf : flopy.mf6.mfmodel.MFModel or None, optional
        Groundwater flow model. Only used if ref_time is not None. Default is
        None
    ref_time : TYPE, optional
        DESCRIPTION. The default is None.
    stoptime : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    mpsim : flopy.modpath.mp7sim.Modpath7Sim
        modpath simulation object.
    """
    if stoptime is None:
        stoptimeoption = "extend"
    else:
        stoptimeoption = "specified"

    if ref_time is None:
        if direction == "backward":
            ref_time = (
                gwf.simulation.tdis.nper.array - 1,  # stress period
                gwf.simulation.tdis.data_list[-1].array[-1][1] - 1,  # timestep
                1.0,
            )
        elif direction == "forward":
            ref_time = 0.0
        else:
            raise ValueError("invalid direction, options are backward or forward")

    mpsim = flopy.modpath.Modpath7Sim(
        mpf,
        simulationtype="combined",
        trackingdirection=direction,
        weaksinkoption="pass_through",
        weaksourceoption="pass_through",
        referencetime=ref_time,
        stoptimeoption=stoptimeoption,
        stoptime=stoptime,
        particlegroups=pg,
    )

    return mpsim
