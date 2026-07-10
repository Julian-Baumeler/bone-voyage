"""Bone density chain and power-law Young's modulus (MITK-GEM parity).

ρ_ct  = slope * HU + offset
ρ_ash = (ρ_ct + ash_offset) / ash_divisor
ρ_app = ρ_ash / app_divisor
E     = factor * ρ_app^exponent + offset   (piecewise by upper bound on ρ)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass
class BoneDensityParams:
    """MITK-GEM BoneDensityParameters / BoneDensityFunctor."""

    slope: float = 1.0
    offset: float = 0.0
    ash_enabled: bool = True
    ash_offset: float = 0.0
    ash_divisor: float = 1.0
    app_enabled: bool = True
    app_divisor: float = 1.0

    def evaluate(self, hu: np.ndarray | float) -> np.ndarray | float:
        rho_ct = self.slope * hu + self.offset
        if self.ash_enabled:
            dens = (rho_ct + self.ash_offset) / (self.ash_divisor or 1.0)
        else:
            dens = rho_ct
        if self.app_enabled:
            dens = dens / (self.app_divisor or 1.0)
        return dens


@dataclass(frozen=True)
class PowerLaw:
    factor: float
    exponent: float
    offset: float = 0.0

    def evaluate(self, rho: np.ndarray | float) -> np.ndarray | float:
        return self.factor * np.power(rho, self.exponent) + self.offset


@dataclass
class PowerLawSet:
    """Piecewise power laws keyed by exclusive upper bound on ρ (MITK-GEM map)."""

    laws: list[tuple[float, PowerLaw]] = field(default_factory=list)

    def add(self, law: PowerLaw, upper_bound: float) -> None:
        self.laws.append((upper_bound, law))
        self.laws.sort(key=lambda x: x[0])

    def evaluate(self, rho: np.ndarray | float) -> np.ndarray | float:
        if not self.laws:
            raise ValueError("No power laws defined")
        scalar = np.isscalar(rho)
        r = np.atleast_1d(np.asarray(rho, dtype=float))
        out = np.empty_like(r)
        bounds = np.array([b for b, _ in self.laws], dtype=float)
        # For each rho, pick first law with upper_bound > rho (upper_bound map)
        for i, val in enumerate(r):
            idx = int(np.searchsorted(bounds, val, side="right"))
            if idx >= len(self.laws):
                idx = len(self.laws) - 1
            out[i] = self.laws[idx][1].evaluate(val)
        return float(out[0]) if scalar else out

    @classmethod
    def default_bone(cls) -> PowerLawSet:
        """Default from MITK-GEM FAQ example (simplified)."""
        s = cls()
        s.add(PowerLaw(1.0, 1.0, 0.0), 0.0)
        s.add(PowerLaw(6850.0, 1.49, 0.0), float("inf"))
        return s


def fit_calibration(
    hu_values: Sequence[float],
    rho_values: Sequence[float],
    unit: str = "mgHA/cm3",
) -> tuple[float, float]:
    """Least-squares fit ρ = slope * HU + offset.

    If unit is mgHA/cm3, rho is converted to gHA/cm3 (÷1000) before fit,
    matching MITK-GEM CalibrationDataModel behaviour.
    """
    hu = np.asarray(hu_values, dtype=float)
    rho = np.asarray(rho_values, dtype=float)
    if unit.lower().startswith("mg"):
        rho = rho / 1000.0
    if hu.size < 2:
        raise ValueError("Need at least two calibration points")
    # ρ = slope * HU + offset  →  polyfit degree 1: ρ = a*HU + b
    slope, offset = np.polyfit(hu, rho, 1)
    return float(slope), float(offset)


def apply_density(hu: np.ndarray, params: BoneDensityParams) -> np.ndarray:
    return np.asarray(params.evaluate(hu), dtype=float)


def apply_power_law(rho: np.ndarray, laws: PowerLawSet) -> np.ndarray:
    rho = np.maximum(rho, 0.0)
    return np.asarray(laws.evaluate(rho), dtype=float)
