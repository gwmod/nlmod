import numpy as np

import nlmod


def test_estimate_nstp():
    forcing = [0.0, 10.0] + 100 * [0.0]
    nstp_min, nstp_max = 1, 25
    tsmult = 1.01
    nstp, dt_arr = nlmod.time.estimate_nstp(
        forcing=forcing,
        tsmult=tsmult,
        nstp_min=nstp_min,
        nstp_max=nstp_max,
        return_dt_arr=True,
    )

    assert nstp[0] == nstp_min
    assert nstp[-1] == nstp_min
    assert max(nstp) == nstp_max
    assert min(nstp) == nstp_min


def test_ds_time_from_tdis_settings():
    tidx = nlmod.time.ds_time_idx_from_tdis_settings(
        "2000", [100, 100, 100], nstp=[1, 2, 2], tsmult=[1.0, 1.0, 2.0]
    )

    elapsed = (tidx.to_numpy() - np.datetime64("2000")) / np.timedelta64(1, "D")
    assert np.allclose(elapsed, [100, 150, 200, 233.33333333, 300.0])
