# %%
from pathlib import Path
from typing import Literal

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import xarray as xr

from nlmod.dims.grid import modelgrid_from_ds
from nlmod.epsg28992 import EPSG_28992

# %%

# Rebuild with QGIS:
# Load LGN.qlr file > Properties > Symbology > Export Color Map to File...
# Run nlmod.read.lgn._load_clr_file(<path to clr file>) to get this dictionary
LGN_COLOR_DICT = {
    1: {"color": "#73df1f", "rgb": (115, 223, 31), "label": "agrarisch gras"},
    2: {"color": "#e89919", "rgb": (232, 153, 25), "label": "mais"},
    3: {"color": "#b26600", "rgb": (178, 102, 0), "label": "aardappelen"},
    4: {"color": "#e51f7f", "rgb": (229, 31, 127), "label": "bieten"},
    5: {"color": "#ffff00", "rgb": (255, 255, 0), "label": "granen"},
    6: {"color": "#ff00c5", "rgb": (255, 0, 197), "label": "overige landbouwgewassen"},
    8: {"color": "#46ffcf", "rgb": (70, 255, 207), "label": "glastuinbouw"},
    9: {"color": "#3cef45", "rgb": (60, 239, 69), "label": "boomgaarden"},
    10: {"color": "#ac81a8", "rgb": (172, 129, 168), "label": "bloembollen"},
    11: {"color": "#33c800", "rgb": (51, 200, 0), "label": "loofbos"},
    12: {"color": "#009900", "rgb": (0, 153, 0), "label": "naaldbos"},
    16: {"color": "#2473ff", "rgb": (36, 115, 255), "label": "zoet water"},
    17: {"color": "#000099", "rgb": (0, 0, 153), "label": "zout water"},
    18: {
        "color": "#ff0000",
        "rgb": (255, 0, 0),
        "label": "bebouwing in primair bebouwd gebied",
    },
    19: {
        "color": "#730000",
        "rgb": (115, 0, 0),
        "label": "bebouwing in secundair bebouwd gebied",
    },
    20: {
        "color": "#93d600",
        "rgb": (147, 214, 0),
        "label": "bos in primair bebouwd gebied",
    },
    22: {
        "color": "#93aa00",
        "rgb": (147, 170, 0),
        "label": "bos in secundair bebouwd gebied",
    },
    23: {
        "color": "#93ff00",
        "rgb": (147, 255, 0),
        "label": "gras in primair bebouwd gebied",
    },
    24: {
        "color": "#ffff66",
        "rgb": (255, 255, 102),
        "label": "kale grond in bebouwd gebied",
    },
    26: {
        "color": "#761818",
        "rgb": (118, 24, 24),
        "label": "bebouwing in buitengebied",
    },
    27: {
        "color": "#ff645a",
        "rgb": (255, 100, 90),
        "label": "overig grondgebruik in buitengebied",
    },
    28: {
        "color": "#a8ef44",
        "rgb": (168, 239, 68),
        "label": "gras in secundair bebouwd gebied",
    },
    29: {"color": "#000000", "rgb": (0, 0, 0), "label": "zonneparken"},
    30: {"color": "#b03060", "rgb": (176, 48, 96), "label": "kwelders"},
    31: {"color": "#e6fb00", "rgb": (230, 251, 0), "label": "open zand in kustgebied"},
    32: {
        "color": "#89d42b",
        "rgb": (137, 212, 43),
        "label": "duinen met lage vegetatie",
    },
    33: {
        "color": "#5aba40",
        "rgb": (90, 186, 64),
        "label": "duinen met hoge vegetatie",
    },
    34: {"color": "#750075", "rgb": (117, 0, 117), "label": "duinheide"},
    35: {
        "color": "#ffff00",
        "rgb": (255, 255, 0),
        "label": "open stuifzand en/of rivierzand",
    },
    36: {"color": "#750075", "rgb": (117, 0, 117), "label": "heide"},
    37: {"color": "#a42353", "rgb": (164, 35, 83), "label": "matig vergraste heide"},
    38: {"color": "#ad8b06", "rgb": (173, 139, 6), "label": "sterk vergraste heide"},
    39: {"color": "#249996", "rgb": (36, 153, 150), "label": "hoogveen"},
    40: {"color": "#065a4c", "rgb": (6, 90, 76), "label": "bos in hoogveengebied"},
    41: {
        "color": "#ffc0cb",
        "rgb": (255, 192, 203),
        "label": "overige moeras vegetatie",
    },
    42: {"color": "#ffa500", "rgb": (255, 165, 0), "label": "rietvegetatie"},
    43: {"color": "#006400", "rgb": (0, 100, 0), "label": "bos in moerasgebied"},
    45: {
        "color": "#b6b639",
        "rgb": (182, 182, 57),
        "label": "natuurlijk beheerde agrarische graslanden",
    },
    46: {"color": "#f5e10f", "rgb": (245, 225, 15), "label": "gras in het kustgebied"},
    47: {"color": "#969639", "rgb": (150, 150, 57), "label": "overig gras"},
    61: {"color": "#ffb3a8", "rgb": (255, 179, 168), "label": "boomkwekerijen"},
    62: {"color": "#e3ff70", "rgb": (227, 255, 112), "label": "fruitkwekerijen"},
    251: {
        "color": "#871b00",
        "rgb": (135, 27, 0),
        "label": "hoofdinfrastructuur en spoorbaanlichamen",
    },
    252: {
        "color": "#b02300",
        "rgb": (176, 35, 0),
        "label": "halfverharde wegen, infrastructuur langzaam verkeer en overige infrastructuur",
    },
    253: {"color": "#a80000", "rgb": (168, 0, 0), "label": "smalle wegen"},
    321: {
        "color": "#89d42b",
        "rgb": (137, 212, 43),
        "label": "struikvegetatie in hoogveengebied (laag)",
    },
    322: {
        "color": "#89d42b",
        "rgb": (137, 212, 43),
        "label": "struikvegetatie in moerasgebied (laag)",
    },
    323: {
        "color": "#89d42b",
        "rgb": (137, 212, 43),
        "label": "overige struikvegetatie (laag)",
    },
    331: {
        "color": "#5aba40",
        "rgb": (90, 186, 64),
        "label": "struikvegetatie in hoogveengebied (hoog)",
    },
    332: {
        "color": "#5aba40",
        "rgb": (90, 186, 64),
        "label": "struikvegetatie in moerasgebied (hoog)",
    },
    333: {
        "color": "#5aba40",
        "rgb": (90, 186, 64),
        "label": "overige struikvegetatie (hoog)",
    },
}


def _load_clr_file(file_path):
    """Parses a QGIS .clr file into a Python dictionary and Matplotlib colormap."""
    lgn_color_dict = {}

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split line at '-' to separate numerical data from the description label
            parts = line.split(" - ")
            label = parts[1].strip() if len(parts) > 1 else ""

            # Parse numbers: [Code, Red, Green, Blue, Alpha, Duplicate_Code]
            num_parts = parts[0].split()
            if len(num_parts) >= 4:
                code = int(num_parts[0])
                r, g, b = int(num_parts[1]), int(num_parts[2]), int(num_parts[3])

                # Convert RGB to HEX format (#RRGGBB)
                hex_color = f"#{r:02x}{g:02x}{b:02x}"

                lgn_color_dict[code] = {
                    "color": hex_color,
                    "rgb": (r, g, b),
                    "label": label,
                }

    return lgn_color_dict


def remap_lgn(lgn: xr.DataArray, mappings: dict, fill_value=-999):
    """Remap LGN values to a simplified set of values.

    Parameters
    ----------
    lgn : xr.DataArray
        LGN data array with original values.
    mappings : dict
        Dictionary mapping original LGN values to new values.
    fill_value : int, optional
        Value to use for unmapped LGN values. Default is -999.

    Returns
    -------
    xr.DataArray
        Remapped LGN data array.
    """
    # Fast path for integer codes: use lookup array
    max_code = int(max(mappings))
    lut = np.full(max_code + 1, fill_value, dtype=np.int32)
    for c, v in mappings.items():
        lut[c] = v

    arr = lgn.values.astype(np.int64, copy=False)
    out = np.full(arr.shape, fill_value, dtype=np.int32)

    valid = (arr >= 0) & (arr <= max_code)
    out[valid] = lut[arr[valid]]
    return xr.DataArray(
        out,
        coords=lgn.coords,
        dims=lgn.dims,
        name="lgn_simplified",
        attrs=lgn.attrs,
    )


def download_lgn():
    """Download LGN data from the LGN website."""
    print("Download LGN data from: https://lgn.nl/bestanden")
    return


def load_lgn_within_extent(extent, lgn_tif_path: str | Path):
    """Load LGN tif file and subset to the given extent.

    Parameters
    ----------
    extent : list
        list of [xmin, xmax, ymin, ymax]
    lgn_tif_path : str or Path
        Path to the LGN tiff file.

    Returns
    -------
    xr.DataArray
        LGN data array.
    """
    lgn = xr.open_dataset(lgn_tif_path, engine="rasterio")
    lgnsel = lgn.sel(x=slice(extent[0], extent[1]), y=slice(extent[3], extent[2]))[
        "band_data"
    ]
    return lgnsel.sel(band=1, drop=True)


def get_lgn_cmap_norm(lgn_color_dict=LGN_COLOR_DICT):
    """Builds a ListedColormap and BoundaryNorm covering the exact pixel codes.

    Parameters
    ----------
    lgn_color_dict : dict
        Dictionary mapping LGN codes to color and label information.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        Colormap for LGN codes.
    norm : matplotlib.colors.BoundaryNorm
        Normalization for LGN codes.
    """
    max_code = max(lgn_color_dict.keys())

    # Default unmapped values to fully transparent white (#FFFFFF00)
    colors = ["#FFFFFF00"] * (max_code + 1)

    for code, data in lgn_color_dict.items():
        colors[code] = data["color"]

    cmap = mcolors.ListedColormap(colors)

    # Boundaries centered around integer raster codes
    bounds = np.arange(-0.5, max_code + 1.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm


def get_lgn_legend(lgn: xr.DataArray):
    """Get a legend for the LGN data array.

    Parameters
    ----------
    lgn : xr.DataArray
        LGN data array.

    Returns
    -------
    list
        List of matplotlib.patches.Patch objects for the legend.

    Example
    -------
    Use to build matplotlib legend::

        lgn = get_lgn(ds)
        f, ax = plt.subplots()
        lgn.plot(ax=ax)
        legend_patches = get_lgn_legend(lgn)
        ax.legend(handles=legend_patches, loc=(1, 0), fontsize="small")

    """
    return [
        mpatches.Patch(color=info["color"], label=f"{code}: {info['label']}")
        for code, info in LGN_COLOR_DICT.items()
        if code in np.unique(lgn)
    ]


def lgn_to_grid(
    lgn: xr.DataArray,
    ds: xr.Dataset,
    engine: Literal["xrspatial", "exactextract"] = "xrspatial",
    crs: str = EPSG_28992,
) -> gpd.GeoDataFrame:
    """Compute area fractions per model cell from LGN.

    Parameters
    ----------
    lgn : xr.DataArray
        LGN raster data array.
    ds : xr.Dataset
        Model dataset.
    engine : str, optional
        Engine to use for mapping. Options are "xrspatial" or "exactextract".
        Default is "xrspatial". `xrspatial.zonal_crosstab` is faster but only counts
        pixels, while `exact_extract.exactextract` calculates area fractions.

    Returns
    -------
    gpd.GeoDataFrame
        lgn area fractions per model cell
    """
    grid = modelgrid_from_ds(ds).to_geodataframe()
    grid = grid.set_crs(crs)

    if engine == "xrspatial":
        import xrspatial as xrs

        lgn_grid = (
            xrs.zonal_crosstab(grid, lgn, column="node").iloc[:, 1:].astype(float)
        )  # drop node/zone column
        lgn_grid /= lgn_grid.sum(axis=1)  # convert counts to area
        lgn_grid.columns = lgn_grid.columns.astype(int)
        lgn_grid.columns.name = "LGN_code"
        lgn_grid.index.name = "icell2d"
        lgn_grid = gpd.GeoDataFrame(lgn_grid, geometry=grid["geometry"])

    elif engine == "exactextract":
        from exactextract import exact_extract

        lgn_grid = exact_extract(
            lgn,
            grid,
            ["unique", "frac"],
            include_cols=["node"],
            include_geom=True,
            strategy="raster-sequential",  # faster
            output="pandas",
        )
        # unstack unique LGN codes into columns and fill in fracs
        lgn_grid_dict = {}
        for node, row in lgn_grid.iterrows():
            lgn_grid_dict[node] = {
                int(code): frac
                for code, frac in zip(row["unique"], row["frac"], strict=True)
            }
        lgn_grid = gpd.GeoDataFrame(
            pd.DataFrame(lgn_grid_dict).T.sort_index(axis=1),
            geometry=grid["geometry"],
        )
        lgn_grid.columns.name = "LGN_code"
        lgn_grid.index.name = "icell2d"

    else:
        raise ValueError(f"Unknown engine: {engine}")

    return lgn_grid
