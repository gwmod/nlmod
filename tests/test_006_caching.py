import os
import tempfile

import nlmod


def test_cache_ahn_data_array():
    """Test caching of AHN data array. Does not have dataset as argument."""
    extent = [119_900, 120_000, 441_900, 442_000]
    cache_name = "ahn4.nc"

    with tempfile.TemporaryDirectory() as tmpdir:
        assert not os.path.exists(os.path.join(tmpdir, cache_name)), "Cache should not exist yet1"
        ahn_no_cache = nlmod.read.ahn.get_ahn4(extent)
        assert not os.path.exists(os.path.join(tmpdir, cache_name)), "Cache should not exist yet2"

        ahn_cached = nlmod.read.ahn.get_ahn4(extent, cachedir=tmpdir, cachename=cache_name)
        assert os.path.exists(os.path.join(tmpdir, cache_name)), "Cache should have existed by now"
        assert ahn_cached.equals(ahn_no_cache)
        modification_time1 = os.path.getmtime(os.path.join(tmpdir, cache_name))

        # Check if the cache is used. If not, cache is rewritten and modification time changes
        ahn_cache = nlmod.read.ahn.get_ahn4(extent, cachedir=tmpdir, cachename=cache_name)
        assert ahn_cache.equals(ahn_no_cache)
        modification_time2 = os.path.getmtime(os.path.join(tmpdir, cache_name))
        assert modification_time1 == modification_time2, "Cache should not be rewritten"

        # Different extent should not lead to using the cache
        extent = [119_800, 120_000, 441_900, 442_000]
        ahn_cache = nlmod.read.ahn.get_ahn4(extent, cachedir=tmpdir, cachename=cache_name)
        modification_time3 = os.path.getmtime(os.path.join(tmpdir, cache_name))
        assert modification_time1 != modification_time3, "Cache should have been rewritten"


def test_cache_northsea_data_array():
    """Test caching of AHN data array. Does have dataset as argument."""
    from nlmod.read.rws import get_northsea
    ds1 = nlmod.get_ds(
        [119_700, 120_000, 441_900, 442_000],
        delr=100.,
        delc=100.,
        top=0.,
        botm=[-1., -2.],
        kh=10.,
        kv=1.,
    )
    ds2 = nlmod.get_ds(
        [119_800, 120_000, 441_900, 444_000],
        delr=100.,
        delc=100.,
        top=0.,
        botm=[-1., -3.],
        kh=10.,
        kv=1.,
    )

    cache_name = "northsea.nc"

    with tempfile.TemporaryDirectory() as tmpdir:
        assert not os.path.exists(os.path.join(tmpdir, cache_name)), "Cache should not exist yet1"
        out1_no_cache = get_northsea(ds1)
        assert not os.path.exists(os.path.join(tmpdir, cache_name)), "Cache should not exist yet2"

        out1_cached = get_northsea(ds1, cachedir=tmpdir, cachename=cache_name)
        assert os.path.exists(os.path.join(tmpdir, cache_name)), "Cache should exist by now"
        assert out1_cached.equals(out1_no_cache)
        modification_time1 = os.path.getmtime(os.path.join(tmpdir, cache_name))

        # Check if the cache is used. If not, cache is rewritten and modification time changes
        out1_cache = get_northsea(ds1, cachedir=tmpdir, cachename=cache_name)
        assert out1_cache.equals(out1_no_cache)
        modification_time2 = os.path.getmtime(os.path.join(tmpdir, cache_name))
        assert modification_time1 == modification_time2, "Cache should not be rewritten"

        # Only properties of `coords_2d` determine if the cache is used. Cache should still be used.
        ds1["toppertje"] = ds1.top + 1
        out1_cache = get_northsea(ds1, cachedir=tmpdir, cachename=cache_name)
        assert out1_cache.equals(out1_no_cache)
        modification_time2 = os.path.getmtime(os.path.join(tmpdir, cache_name))
        assert modification_time1 == modification_time2, "Cache should not be rewritten"

        # Different extent should not lead to using the cache
        out2_cache = get_northsea(ds2, cachedir=tmpdir, cachename=cache_name)
        modification_time3 = os.path.getmtime(os.path.join(tmpdir, cache_name))
        assert modification_time1 != modification_time3, "Cache should have been rewritten"
        assert not out2_cache.equals(out1_no_cache)
