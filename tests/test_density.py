"""Port of MITK-GEM BoneDensity / PowerLaw Catch tests."""

import numpy as np

from opengem.core.density import (
    BoneDensityParams,
    PowerLaw,
    PowerLawSet,
    fit_calibration,
)


def test_bone_density_functor():
    p = BoneDensityParams(
        slope=0.0087,
        offset=-0.00159,
        ash_enabled=True,
        ash_offset=0.09,
        ash_divisor=1.14,
        app_enabled=True,
        app_divisor=0.6,
    )

    def expected(x):
        ct_val = 0.0087 * x - 0.00159
        ash_val = (ct_val + 0.09) / 1.14
        return ash_val / 0.6

    for nr in (0, 0.0001, -0.0001, 1000, -1000):
        assert abs(float(p.evaluate(nr)) - expected(nr)) < 1e-12


def test_power_law_set():
    s = PowerLawSet()
    s.add(PowerLaw(1, 1, 7), 0)
    s.add(PowerLaw(6850, 1.49, 8), 200)
    s.add(PowerLaw(3, 6, 9), 300)

    def expected(x):
        if x < 0:
            return 1 * x**1 + 7
        if x < 200:
            return 6850 * x**1.49 + 8
        return 3 * x**6 + 9

    for nr in (-100, -0.00001, 0, 0.0001, 99, 199, 200, 201, 299, 300, 301):
        got = float(s.evaluate(nr))
        exp = expected(nr)
        assert abs(got - exp) / (abs(exp) + 1e-9) < 1e-9 or abs(got - exp) < 1e-6


def test_fit_calibration_mg():
    # two points in mgHA/cm3 → fit in gHA/cm3
    slope, offset = fit_calibration([0, -5], [10_000, 0], unit="mgHA/cm3")
    # rho_g = 10 at HU=0, 0 at HU=-5 → slope = (0-10)/(-5-0) wait:
    # points (HU, rho_g): (0, 10), (-5, 0)
    # slope = (0-10)/(-5-0) = 2, offset = 10
    assert abs(slope - 2.0) < 1e-9
    assert abs(offset - 10.0) < 1e-9
