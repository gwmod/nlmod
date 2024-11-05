import numpy as np
from flopy.utils.binaryfile import binaryread


def _get_binary_head_data(kstpkper, fobj):
    """Get head data array for timestep (kstp, kper).

    Adapted from flopy.utils.binaryfile.HeadFile. Removed support for all other
    types of indexing (ids/totim) and only supports kstpkper.

    Parameters
    ----------
    kstpkper : tuple(int, int)
        step and period index
    filepath : str
        path to binary heads file
    hobj : flopy.utils.HeadFile
        flopy HeadFile object

    Returns
    -------
    np.array
        array containing data for timestep (kstp, kper)
    """
    kstp1 = kstpkper[0] + 1
    kper1 = kstpkper[1] + 1
    idx = np.where(
        (fobj.recordarray["kstp"] == kstp1) & (fobj.recordarray["kper"] == kper1)
    )
    totim1 = fobj.recordarray[idx]["totim"][0]

    keyindices = np.where(fobj.recordarray["totim"] == totim1)[0]
    # initialize head with nan and then fill it
    idx = keyindices[0]
    nrow = fobj.recordarray["nrow"][idx]
    ncol = fobj.recordarray["ncol"][idx]
    arr = np.empty((fobj.nlay, nrow, ncol), dtype=fobj.realtype)
    arr[:, :, :] = np.nan
    for idx in keyindices:
        ipos = fobj.iposarray[idx]
        ilay = fobj.recordarray["ilay"][idx]
        with open(fobj.filename, "rb") as f:
            f.seek(ipos, 0)
            nrow = fobj.recordarray["nrow"][idx]
            ncol = fobj.recordarray["ncol"][idx]
            shp = (nrow, ncol)
            arr[ilay - 1] = binaryread(f, fobj.realtype, shp)
    return arr


def __create3D(data, fobj, column="q", node="node"):
    """Adapted from CellBudgetFile.__create3D.

    See flopy.utils.binaryfile.CellBudgetFile.__create3D.
    """
    out = np.ma.zeros(fobj.nnodes, dtype=np.float32)
    out.mask = True
    for n, q in zip(data[node], data[column]):
        idx = n - 1
        out.data[idx] += q
        out.mask[idx] = False
    return np.ma.reshape(out, fobj.shape)


def _select_data_indices_budget(fobj, text, kstpkper):
    """Select data indices for budgetfile.

    Parameters
    ----------
    fobj : flopy.utils.CellBudgetFile
        CellBudgetFile object
    text : str
        text indicating which dataset to load
    kstpkper : tuple(int, int)
        step and period index

    Returns
    -------
    select_indices : np.array of int
        array with indices of data to load
    """
    # check and make sure that text is in file
    text16 = None
    if text is not None:
        text16 = fobj._find_text(text)

    # get kstpkper indices
    kstp1 = kstpkper[0] + 1
    kper1 = kstpkper[1] + 1
    if text is None:
        select_indices = np.where(
            (fobj.recordarray["kstp"] == kstp1) & (fobj.recordarray["kper"] == kper1)
        )
    else:
        if text is not None:
            select_indices = np.where(
                (fobj.recordarray["kstp"] == kstp1)
                & (fobj.recordarray["kper"] == kper1)
                & (fobj.recordarray["text"] == text16)
            )

    # build and return the record list
    if isinstance(select_indices, tuple):
        select_indices = select_indices[0]
    return select_indices


def _get_binary_budget_data(kstpkper, fobj, text, column="q", node="node"):
    """Get budget data from binary CellBudgetFile.

    Code copied from flopy.utils.binaryfile.CellBudgetFile and adapted to
    open binary file for each function call instead of relying on one open file.
    All support for totim/idx selection is dropped, only providing kstpkper will
    work.

    Parameters
    ----------
    kstpkper : tuple(int, int)
        tuple with timestep and stressperiod indices
    fobj : flopy.utils.HeadFile or flopy.utils.CellBudgetFile
        file object
    text : str
        text indicating which dataset to read
    column : str
        name of column in rec-array to read, default is 'q' which contains the fluxes
        for most budget datasets.

    Returns
    -------
    data : np.array
        array containing data for a specific timestep
    """
    # select indices to read
    idx = _select_data_indices_budget(fobj, text, kstpkper)

    # idx must be an ndarray, so if it comes in as an integer then convert
    if np.isscalar(idx):
        idx = np.array([idx])

    header = fobj.recordarray[idx]

    t = header["text"][0]
    if isinstance(t, bytes):
        t = t.decode("utf-8")

    data = []
    for ipos in fobj.iposarray[idx]:
        data.append(_get_binary_budget_record(fobj, ipos, header, column, node))

    if len(data) == 1:
        return data[0]
    else:
        return np.ma.sum(data, axis=0)


def _get_binary_budget_record(fobj, ipos, header, column, node):
    """Get a single data record from the budget file."""
    imeth = header["imeth"][0]
    nlay = abs(header["nlay"][0])
    nrow = header["nrow"][0]
    ncol = header["ncol"][0]

    # default method
    with open(fobj.filename, "rb") as f:
        f.seek(ipos, 0)

        # imeth 0
        if imeth == 0:
            return binaryread(f, fobj.realtype(1), shape=(nlay, nrow, ncol))
        # imeth 1
        elif imeth == 1:
            return binaryread(f, fobj.realtype(1), shape=(nlay, nrow, ncol))

        # imeth 2
        elif imeth == 2:
            nlist = binaryread(f, np.int32)[0]
            dtype = np.dtype([("node", np.int32), ("q", fobj.realtype)])
            data = binaryread(f, dtype, shape=(nlist,))
            return __create3D(data, fobj)

        # imeth 3
        elif imeth == 3:
            ilayer = binaryread(f, np.int32, shape=(nrow, ncol))
            data = binaryread(f, fobj.realtype(1), shape=(nrow, ncol))
            out = np.ma.zeros(fobj.nnodes, dtype=np.float32)
            out.mask = True
            vertical_layer = ilayer.flatten() - 1
            # create the 2D cell index and then move it to
            # the correct vertical location
            idx = np.arange(0, vertical_layer.shape[0])
            idx += vertical_layer * nrow * ncol
            out[idx] = data.flatten()
            return out.reshape(fobj.shape)

        # imeth 4
        elif imeth == 4:
            return binaryread(f, fobj.realtype(1), shape=(nrow, ncol))

        # imeth 5
        elif imeth == 5:
            nauxp1 = binaryread(f, np.int32)[0]
            naux = nauxp1 - 1
            dtype_list = [("node", np.int32), ("q", fobj.realtype)]
            for _ in range(naux):
                auxname = binaryread(f, str, charlen=16)
                if not isinstance(auxname, str):
                    auxname = auxname.decode()
                dtype_list.append((auxname.strip(), fobj.realtype))
            dtype = np.dtype(dtype_list)
            nlist = binaryread(f, np.int32)[0]
            data = binaryread(f, dtype, shape=(nlist,))
            return __create3D(data, fobj)

        # imeth 6
        elif imeth == 6:
            # read rest of list data
            nauxp1 = binaryread(f, np.int32)[0]
            naux = nauxp1 - 1
            dtype_list = [("node", np.int32), ("node2", np.int32), ("q", fobj.realtype)]
            for _ in range(naux):
                auxname = binaryread(f, bytes, charlen=16)
                if not isinstance(auxname, str):
                    auxname = auxname.decode()
                dtype_list.append((auxname.strip(), fobj.realtype))
            dtype = np.dtype(dtype_list)
            nlist = binaryread(f, np.int32)[0]
            data = binaryread(f, dtype, shape=(nlist,))
            data = __create3D(data, fobj, column=column, node=node)
            if fobj.modelgrid is not None:
                return np.reshape(data, fobj.shape)
            else:
                return data
        else:
            raise ValueError(f"invalid imeth value - {imeth}")
