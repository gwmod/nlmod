import matplotlib
import pytest
import nlmod

# def test_download_polygons(): # is tested in test_024_administrative.test_get_waterboards
#     nlmod.read.waterboard.get_polygons()


def test_get_config():
    nlmod.read.waterboard.get_configuration()


def test_bgt_waterboards():
    extent = [116500, 120000, 439000, 442000]
    bgt = nlmod.read.bgt.get_bgt(extent)

    la = nlmod.gwf.surface_water.download_level_areas(
        bgt, extent=extent, raise_exceptions=False
    )
    bgt = nlmod.gwf.surface_water.add_stages_from_waterboards(bgt, la=la)


@pytest.mark.skip("too slow")
def test_download_peilgebieden(plot=True):
    waterboards = nlmod.read.waterboard.get_polygons()
    data_kind = "level_areas"

    gdf = {}
    for wb in waterboards.index:
        print(wb)
        try:
            # xmin, ymin, xmax, ymax = waterboards.at[wb, "geometry"].bounds
            # extent = [xmin, xmax, ymin, ymax]
            gdf[wb] = nlmod.read.waterboard.get_data(
                wb, data_kind, max_record_count=1000
            )
        except Exception as e:
            if str(e) == f"{data_kind} not available for {wb}":
                print(e)
            else:
                raise

    if plot:
        # plot the winter_stage
        f, ax = nlmod.plot.get_map([9000, 279000, 304000, 623000], base=100000)
        waterboards.plot(edgecolor="k", facecolor="none", ax=ax)
        norm = matplotlib.colors.Normalize(-10.0, 20.0)
        cmap = "viridis"
        for wb in waterboards.index:
            if wb in gdf:
                try:
                    # gdf[wb].plot(ax=ax, zorder=0)
                    gdf[wb].plot("winter_stage", ax=ax, zorder=0, norm=norm, cmap=cmap)
                except Exception as e:
                    print(f"plotting of {data_kind} for {wb} failed: {e}")
            c = waterboards.at[wb, "geometry"].centroid
            ax.text(c.x, c.y, wb.replace(" ", "\n"), ha="center", va="center")
        nlmod.plot.colorbar_inside(
            ax=ax,
            norm=norm,
            cmap=cmap,
            bounds=[0.05, 0.55, 0.02, 0.4],
            label="Summer stage (m NAP)",
        )


@pytest.mark.skip("too slow")
def test_download_waterlopen(plot=True):
    def get_extent(waterboards, wb, buffer=1000.0):
        c = waterboards.at[wb, "geometry"].centroid
        extent = [c.x - buffer, c.x + buffer, c.y - buffer, c.y + buffer]
        if wb == "Vallei & Veluwe":
            extent = [170000, 172000, 460000, 462000]
        # elif wb == "Aa en Maas":
        #    extent = [132500, 147500, 408000, 416000]
        # elif wb == "HH Hollands Noorderkwartier":
        #    extent = [120000, 123000, 510000, 513000]
        # elif wb == "Waterschap Limburg":
        #    extent = [190000, 196000, 358000, 364000]
        # elif wb == "Brabantse Delta":
        #    extent = [100000, 105000, 405000, 410000]
        # elif wb == "Waterschap Scheldestromen":
        #    extent = [57000, 58000, 378000, 379000]
        return extent

    data_kind = "watercourses"
    # data_kind = "level_areas"
    waterboards = nlmod.read.waterboard.get_polygons()
    gdf = {}
    for wb in waterboards.index:
        print(wb)
        extent = get_extent(waterboards, wb)
        try:
            gdf[wb] = nlmod.read.waterboard.get_data(wb, data_kind, extent)
        except Exception as e:
            if str(e) == f"{data_kind} not available for {wb}":
                print(e)
            else:
                print(e)
                raise

    if plot:
        for wb in gdf:
            ax = gdf[wb].plot()
            ax.axis(get_extent(waterboards, wb))
            ax.set_title(wb)
