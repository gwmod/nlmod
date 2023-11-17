import nlmod


def test_prepare():
    ds = nlmod.get_ds([0, 1000, 0, 2000])
    ds = nlmod.gwt.prepare.set_default_transport_parameters(
        ds, transport_type="chloride"
    )
