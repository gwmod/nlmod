import os
import requests
import nlmod
from lxml import html
from pandas import to_datetime


def test_bgt_bronhoudername_up_to_date():
    url = "https://www.kadaster.nl/-/bgt-bronhoudercodes"
    response = requests.get(url, timeout=10)
    tree = html.fromstring(response.content)
    dates = to_datetime(
        [
            "-".join(x.rstrip("-").rsplit("-", 3)[-3:])
            for x in tree.xpath("//a/@href")
            if "bronhoudercodes" in x
        ],
        format="%d-%m-%Y",
    )
    msg = "Bronhoudercodes are not up to date. Update `nlmod.read.bgt.get_bronhouder_names()`."
    assert max(dates) == to_datetime("2025-01-01"), msg


def test_bgt_layers():
    layers = nlmod.read.bgt.get_bgt_layers()
    assert isinstance(layers, list)
    assert "waterdeel" in layers


def test_bgt_zipfile():
    pathname = "download"
    if not os.path.isdir(pathname):
        os.makedirs(pathname)
    fname = os.path.join(pathname, "test_bgt_zipfile.zip")

    # download data from 2 layers within extent, and also save data to zipfile
    extent = [119900, 120000, 440000, 440100]
    bgt = nlmod.read.bgt.get_bgt(extent, layer=["waterdeel", "wegdeel"], fname=fname)
    assert isinstance(bgt, dict)
    assert "waterdeel" in bgt
    assert "bronhouder_name" in bgt["waterdeel"].columns

    # read data again from zipfile
    bgt_from_zip = nlmod.read.bgt.read_bgt_zipfile(fname)
    assert isinstance(bgt_from_zip, dict)
    assert "waterdeel" in bgt_from_zip
    assert bgt.keys() == bgt_from_zip.keys()
