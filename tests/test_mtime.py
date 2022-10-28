import nlmod


def test_estimate_nstp():
    forcing = [0.0, 10.0] + 100 * [0.0]
    nstp_min, nstp_max = 1, 25
    tsmult = 1.01
    nstp, dt_arr = nlmod.mtime.estimate_nstp(
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
