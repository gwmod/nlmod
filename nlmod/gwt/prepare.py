def set_default_transport_parameters(ds, transport_type):
    """Set default transport parameters based on type of transport model.

    Convenience function for setting several variables at once for which
    default values are often used.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset
    transport_type : str
        type of transport model, currently supports "chloride" or "tracer".

    Returns
    -------
    ds : xarray.Dataset
        dataset with transport parameters added to attributes.
    """
    if transport_type == "chloride":
        # buy
        ds.attrs["drhodc"] = 25.0 / 18_000.0  # delta density / delta concentration
        ds.attrs["denseref"] = 1000.0  # reference density
        ds.attrs["crhoref"] = 0.0  # reference concentration

        # mst
        if "porosity" not in ds:
            ds.attrs["porosity"] = 0.3

        # adv
        ds.attrs["adv_scheme"] = "UPSTREAM"  # advection scheme

        # dsp
        # ds.attrs["dsp_diffc"] = None  # Diffusion coefficient
        ds.attrs["dsp_alh"] = 1.0  # Longitudinal dispersivity ($m$)
        ds.attrs["dsp_ath1"] = 0.1  # Transverse horizontal dispersivity ($m$)
        ds.attrs["dsp_atv"] = 0.1  # Transverse vertical dispersivity ($m$)

        # ssm
        ds.attrs["ssm_sources"] = []

        # general
        ds.attrs["gwt_units"] = "mg Cl- /L"

    elif transport_type == "tracer":
        # mst
        if "porosity" not in ds:
            ds.attrs["porosity"] = 0.3
        # adv
        ds.attrs["adv_scheme"] = "UPSTREAM"

        # dsp
        # ds.attrs["dsp_diffc"] = None  # Diffusion coefficient
        ds.attrs["dsp_alh"] = 1.0  # Longitudinal dispersivity ($m$)
        ds.attrs["dsp_ath1"] = 0.1  # Transverse horizontal dispersivity ($m$)
        ds.attrs["dsp_atv"] = 0.1  # Transverse vertical dispersivity ($m$)

    else:
        raise ValueError("Only 'chloride' and 'tracer' transport types are defined.")

    return ds
