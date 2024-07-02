"""
NOTE: this is the correct epsg:28992 definition for plotting backgroundmaps in RD
More info (in Dutch) here:
https://qgis.nl/2011/12/05/epsg28992-of-rijksdriehoekstelsel-verschuiving/
This was still a problem in October 2023
"""

EPSG_28992 = (
    "+proj=sterea +lat_0=52.15616055555555 +lon_0=5.38763888888889 +k=0.9999079 "
    "+x_0=155000 +y_0=463000 +ellps=bessel "
    "+towgs84=565.417,50.3319,465.552,-0.398957,0.343988,-1.8774,4.0725 +units=m "
    "+no_defs"
)
