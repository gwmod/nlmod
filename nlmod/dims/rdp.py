"""
rdp.

This module is only used in the deprecated get_vertices_arr function in the grid module.
As soon as that function is removed this module can be removed as well.
~~~
Python implementation of the Ramer-Douglas-Peucker algorithm.
:copyright: 2014-2016 Fabian Hirschmann <fabian@hirschmann.email>
:license: MIT, see LICENSE.txt for more details.
"""

import sys
from functools import partial

import numpy as np

if sys.version_info[0] >= 3:
    xrange = range


def pldist(point, start, end):
    """Calculates the distance from point to the line given by start and end.

    Parameters
    ----------
    point : numpy array
        A point.
    start : numpy array
        A point of the line.
    end : numpy array
        Another point of the line.

    Returns
    -------
    float
        The distance from point to the line.
    """
    if np.all(np.equal(start, end)):
        return np.linalg.norm(point - start)

    return np.divide(
        np.abs(np.linalg.norm(np.cross(end - start, start - point))),
        np.linalg.norm(end - start),
    )


def rdp_rec(M, epsilon, dist=pldist):
    """Simplifies a given array of points. Recursive version.

    Parameters
    ----------
    M : numpy array
        An array of points.
    epsilon : float
        Epsilon in the rdp algorithm.
    dist : function, optional
        Distance function with signature f(point, start, end).
        See :func:`rdp.pldist`. The default is pldist.

    Returns
    -------
    numpy array
        Simplified array of points.
    """
    dmax = 0.0
    index = -1

    for i in xrange(1, M.shape[0]):
        d = dist(M[i], M[0], M[-1])

        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        r1 = rdp_rec(M[: index + 1], epsilon, dist)
        r2 = rdp_rec(M[index:], epsilon, dist)

        return np.vstack((r1[:-1], r2))
    else:
        return np.vstack((M[0], M[-1]))


def _rdp_iter(M, start_index, last_index, epsilon, dist=pldist):
    stk = []
    stk.append([start_index, last_index])
    global_start_index = start_index
    indices = np.ones(last_index - start_index + 1, dtype=bool)

    while stk:
        start_index, last_index = stk.pop()

        dmax = 0.0
        index = start_index

        for i in xrange(index + 1, last_index):
            if indices[i - global_start_index]:
                d = dist(M[i], M[start_index], M[last_index])
                if d > dmax:
                    index = i
                    dmax = d

        if dmax > epsilon:
            stk.append([start_index, index])
            stk.append([index, last_index])
        else:
            for i in xrange(start_index + 1, last_index):
                indices[i - global_start_index] = False

    return indices


def rdp_iter(M, epsilon, dist=pldist, return_mask=False):
    """Simplifies a given array of points. Iterative version.

    Parameters
    ----------
    M : numpy array
        An array of points.
    epsilon : float
        Epsilon in the rdp algorithm.
    dist : function, optional
        Distance function with signature f(point, start, end).
        See :func:`rdp.pldist`. The default is pldist.
    return_mask : bool, optional
        Return the mask of points to keep instead. The default is False.

    Returns
    -------
    numpy array or bool array
        Simplified array of points or mask of points to keep.
    """
    mask = _rdp_iter(M, 0, len(M) - 1, epsilon, dist)

    if return_mask:
        return mask

    return M[mask]


def rdp(M, epsilon=0, dist=pldist, algo="iter", return_mask=False):
    """Simplifies a given array of points using the Ramer-Douglas-Peucker algorithm.

    This is a convenience wrapper around both :func:`rdp.rdp_iter`
    and :func:`rdp.rdp_rec` that detects if the input is a numpy array
    in order to adapt the output accordingly. This means that
    when it is called using a Python list as argument, a Python
    list is returned, and in case of an invocation using a numpy
    array, a NumPy array is returned.

    The parameter ``return_mask=True`` can be used in conjunction
    with ``algo="iter"`` to return only the mask of points to keep.

    Parameters
    ----------
    M : numpy array
        A series of points with shape (n,d) where n is the number of points
        and d is their dimension.
    epsilon : float, optional
        Epsilon in the rdp algorithm. The default is 0.
    dist : function, optional
        Distance function with signature f(point, start, end).
        See :func:`rdp.pldist`. The default is pldist.
    algo : str, optional
        Either "iter" for an iterative algorithm or "rec" for a recursive
        algorithm. The default is "iter".
    return_mask : bool, optional
        Return mask instead of simplified array. The default is False.

    Returns
    -------
    numpy array or list
        Simplified array of points or mask of points to keep.

    Examples
    --------
    >>> from rdp import rdp
    >>> rdp([[1, 1], [2, 2], [3, 3], [4, 4]])
    [[1, 1], [4, 4]]
    >>> import numpy as np
    >>> arr = np.array([1, 1, 2, 2, 3, 3, 4, 4]).reshape(4, 2)
    >>> mask = rdp(arr, algo="iter", return_mask=True)
    >>> arr[mask]
    array([[1, 1], [4, 4]])
    """
    if algo == "iter":
        algo = partial(rdp_iter, return_mask=return_mask)
    elif algo == "rec":
        if return_mask:
            raise NotImplementedError('return_mask=True not supported with algo="rec"')
        algo = rdp_rec

    if "numpy" in str(type(M)):
        return algo(M, epsilon, dist)

    return algo(np.array(M), epsilon, dist).tolist()
