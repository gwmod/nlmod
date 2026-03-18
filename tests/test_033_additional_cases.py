import cftime
import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from requests.exceptions import HTTPError, RequestException
from shapely.geometry import Polygon

import nlmod
from nlmod import config
from nlmod.dims.layers import kveq_combined_layers
from nlmod.modpath.modpath import package_to_nodes
from nlmod.read import administrative, knmi_data_platform, webservices


@pytest.fixture(autouse=True)
def _reset_config_options():
    config.reset_options()
    yield
    config.reset_options()


def test_config_reset_unknown_key_raises_keyerror():
    with pytest.raises(KeyError):
        config.reset_options(["does_not_exist"])


def test_config_get_options_shape_contract():
    all_opts = config.get_options()
    one_opt = config.get_options("nc_hash")

    assert isinstance(all_opts, dict)
    assert set(one_opt) == {"nc_hash"}
    assert isinstance(one_opt["nc_hash"], bool)


def test_set_ds_time_mixed_string_formats():
    ds = nlmod.get_ds([0, 100, 0, 100])
    out = nlmod.time.set_ds_time(
        ds,
        start="2000-01-01",
        time=["2000-02-01", "2000/03/01"],
    )

    assert out.sizes["time"] == 2
    assert out.time.values[1] > out.time.values[0]


def test_set_ds_time_mixed_formats_with_cftime_start():
    ds = nlmod.get_ds([0, 100, 0, 100])
    out = nlmod.time.set_ds_time(
        ds,
        start=cftime.datetime(1000, 1, 1),
        time=["2000-02-01", "2000/03/01"],
    )

    assert isinstance(out.time.values[0], cftime.datetime)
    assert out.time.values[1] > out.time.values[0]


def test_kveq_combined_layers_zero_denominator_returns_nan():
    kv = xr.DataArray(
        np.array(
            [
                [[1.0, np.inf]],
                [[1.0, np.inf]],
            ]
        ),
        dims=("layer", "y", "x"),
    )
    thickness = xr.DataArray(np.ones_like(kv), dims=kv.dims)

    out = kveq_combined_layers(kv, thickness, {0: (0, 1)})

    assert np.isfinite(out.data[0, 0, 0])
    assert np.isnan(out.data[0, 0, 1])


def test_webservices_get_data_non_json_propagates_value_error(monkeypatch):
    class DummyResponse:
        ok = True
        url = "https://example.test"

        def json(self):
            raise ValueError("not json")

    monkeypatch.setattr(
        webservices.requests, "get", lambda *args, **kwargs: DummyResponse()
    )

    with pytest.raises(ValueError, match="not json"):
        webservices._get_data("https://example.test", {})


def test_webservices_get_data_http_error_on_bad_status(monkeypatch):
    class DummyResponse:
        ok = False
        url = "https://example.test"

    monkeypatch.setattr(
        webservices.requests, "get", lambda *args, **kwargs: DummyResponse()
    )

    with pytest.raises(HTTPError, match="Request not successful"):
        webservices._get_data("https://example.test", {})


def test_webservices_get_data_error_payload_raises(monkeypatch):
    class DummyResponse:
        ok = True
        url = "https://example.test"

        def json(self):
            return {"error": {"code": 500, "message": "boom"}}

    monkeypatch.setattr(
        webservices.requests, "get", lambda *args, **kwargs: DummyResponse()
    )

    with pytest.raises(ValueError, match="Error code 500"):
        webservices._get_data("https://example.test", {})


def test_administrative_wfs_request_exception_propagates(monkeypatch):
    def _raise(*args, **kwargs):
        raise RequestException("network down")

    monkeypatch.setattr(administrative.webservices, "wfs", _raise)

    with pytest.raises(RequestException, match="network down"):
        administrative.download_municipalities_gdf(source="cbs", cachename="muni_exc")
    with pytest.raises(RequestException, match="network down"):
        administrative.download_provinces_gdf(source="cbs", cachename="prov_exc")
    with pytest.raises(RequestException, match="network down"):
        administrative.download_netherlands_gdf(source="cbs", cachename="nl_exc")


def test_administrative_waterboards_request_exception_propagates(monkeypatch):
    def _raise(*args, **kwargs):
        raise RequestException("network down")

    monkeypatch.setattr(administrative.waterboard, "download_polygons", _raise)

    with pytest.raises(RequestException, match="network down"):
        administrative.download_waterboards_gdf(cachename="wb_exc")


def test_knmi_download_file_accepts_pathlike(monkeypatch, tmp_path):
    class JsonResponse:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    class StreamResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield b"abc"

    def fake_get(url, *args, **kwargs):
        if str(url).endswith("/url"):
            return JsonResponse({"temporaryDownloadUrl": "https://download.test/file"})
        return StreamResponse()

    monkeypatch.setattr(knmi_data_platform.requests, "get", fake_get)

    knmi_data_platform.download_file(
        dataset_name="dummy",
        dataset_version="1",
        fname="file.bin",
        dirname=tmp_path,
        api_key="token",
    )

    assert (tmp_path / "file.bin").exists()


def test_knmi_download_files_forwards_pathlike(monkeypatch, tmp_path):
    calls = []

    def fake_download_file(*, dirname, **kwargs):
        calls.append(dirname)

    monkeypatch.setattr(knmi_data_platform, "download_file", fake_download_file)

    knmi_data_platform.download_files(
        dataset_name="dummy",
        dataset_version="1",
        fnames=["a.bin", "b.bin"],
        dirname=tmp_path,
        api_key="token",
    )

    assert calls == [tmp_path, tmp_path]


def test_surface_water_discretize_missing_columns_raises_keyerror():
    ds = nlmod.get_ds(
        [0, 100, 0, 100],
        delr=50.0,
        delc=50.0,
        top=1.0,
        botm=[0.0],
        kh=1.0,
        kv=1.0,
    )
    gdf = gpd.GeoDataFrame(
        {"geometry": [Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])]},
        geometry="geometry",
        crs=28992,
    )

    with pytest.raises(KeyError):
        nlmod.read.rws.discretize_surface_water(ds, gdf, da_basename="sw")


def test_modpath_package_to_nodes_without_ibound_structured_and_vertex():
    class DummyConnData:
        def __init__(self, cellids):
            self.array = {"cellid": cellids}

    class DummyPackage:
        def __init__(self, cellids):
            self.connectiondata = DummyConnData(cellids)

    class DummyGrid:
        def __init__(self, grid_type, shape):
            self.grid_type = grid_type
            self.shape = shape

    class DummyGwf:
        def __init__(self, grid_type, shape, cellids):
            self.modelgrid = DummyGrid(grid_type, shape)
            self._pkg = DummyPackage(cellids)

        def get_package(self, name):
            return self._pkg

    gwf_struct = DummyGwf("structured", (1, 2, 3), [(0, 0, 0), (0, 1, 2)])
    nodes_struct = package_to_nodes(gwf_struct, "GHB")
    assert nodes_struct == [0, 5]

    gwf_vertex = DummyGwf("vertex", (2, 4), [(0, 0), (1, 3)])
    nodes_vertex = package_to_nodes(gwf_vertex, "GHB")
    assert nodes_vertex == [0, 7]


def test_show_versions_propagates_missing_dependency(monkeypatch):
    from importlib import metadata
    from nlmod import version as version_module

    def fake_version(name):
        if name == "numpy":
            raise metadata.PackageNotFoundError("numpy")
        return "1.0"

    monkeypatch.setattr(version_module.metadata, "version", fake_version)

    with pytest.raises(metadata.PackageNotFoundError):
        version_module.show_versions()
