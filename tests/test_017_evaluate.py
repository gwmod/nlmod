import nlmod
import test_001_model


def test_gxg():
    ds = test_001_model.get_ds_from_cache("basic_sea_model")
    head = nlmod.gwf.get_heads_da(ds)
    nlmod.evaluate.calculate_gxg(head)
