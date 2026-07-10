"""Load/save MITK-GEM .matmap XML (MaterialMapping Version 2016.2)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opengem.core.density import BoneDensityParams, PowerLaw, PowerLawSet


@dataclass
class MatMapConfig:
    calibration_unit: str = "mgHA/cm3"
    calibration_points: list[tuple[float, float]] = field(default_factory=list)  # (HU, rho)
    density: BoneDensityParams = field(default_factory=BoneDensityParams)
    auto_fit_rho_ct: bool = True
    power_laws: PowerLawSet = field(default_factory=PowerLawSet.default_bone)
    do_peel: bool = True
    number_of_extends: int = 3
    min_value: float = 0.0


def load_matmap(path: str | Path) -> MatMapConfig:
    tree = ET.parse(path)
    root = tree.getroot()
    cfg = MatMapConfig()

    cal = root.find("Calibration")
    if cal is not None:
        cfg.calibration_unit = cal.get("unit", cfg.calibration_unit)
        for dp in cal.findall("DataPoint"):
            cfg.calibration_points.append((float(dp.get("HU", 0)), float(dp.get("rho", 0))))

    bdp = root.find("BoneDensityParameters")
    if bdp is not None:
        rho_ct = bdp.find("RhoCT")
        if rho_ct is not None:
            cfg.auto_fit_rho_ct = rho_ct.get("AutomaticFit", "1") in ("1", "true", "True")
            cfg.density.slope = float(rho_ct.get("slope", 0) or 0)
            cfg.density.offset = float(rho_ct.get("offset", 0) or 0)
        rho_ash = bdp.find("RhoAsh")
        if rho_ash is not None:
            cfg.density.ash_enabled = rho_ash.get("enabled", "1") in ("1", "true", "True")
            cfg.density.ash_offset = float(rho_ash.get("offset", 0) or 0)
            cfg.density.ash_divisor = float(rho_ash.get("divisor", 1) or 1)
        rho_app = bdp.find("RhoApp")
        if rho_app is not None:
            cfg.density.app_enabled = rho_app.get("enabled", "1") in ("1", "true", "True")
            cfg.density.app_divisor = float(rho_app.get("divisor", 1) or 1)

    pls = root.find("PowerLaws")
    if pls is not None:
        laws = PowerLawSet()
        for pl in pls.findall("PowerLawParameters"):
            law = PowerLaw(
                factor=float(pl.get("factor", 1)),
                exponent=float(pl.get("exponent", 1)),
                offset=float(pl.get("offset", 0)),
            )
            rmax = float(pl.get("rangeMax", "inf") or "inf")
            if rmax > 1e30:
                rmax = float("inf")
            laws.add(law, rmax)
        if laws.laws:
            cfg.power_laws = laws

    opts = root.find("Options")
    if opts is not None:
        cfg.do_peel = opts.get("doPeel", "1") in ("1", "true", "True")
        cfg.number_of_extends = int(float(opts.get("numberOfExtends", 3)))
        cfg.min_value = float(opts.get("minValue", 0))

    return cfg


def save_matmap(path: str | Path, cfg: MatMapConfig) -> None:
    root = ET.Element("MaterialMapping", Version="2016.2")
    cal = ET.SubElement(root, "Calibration", unit=cfg.calibration_unit)
    for hu, rho in cfg.calibration_points:
        ET.SubElement(cal, "DataPoint", HU=str(hu), rho=str(rho))

    bdp = ET.SubElement(root, "BoneDensityParameters")
    ET.SubElement(
        bdp,
        "RhoCT",
        AutomaticFit="1" if cfg.auto_fit_rho_ct else "0",
        slope=str(cfg.density.slope),
        offset=str(cfg.density.offset),
    )
    ET.SubElement(
        bdp,
        "RhoAsh",
        enabled="1" if cfg.density.ash_enabled else "0",
        offset=str(cfg.density.ash_offset),
        divisor=str(cfg.density.ash_divisor),
    )
    ET.SubElement(
        bdp,
        "RhoApp",
        enabled="1" if cfg.density.app_enabled else "0",
        divisor=str(cfg.density.app_divisor),
    )

    pls = ET.SubElement(root, "PowerLaws")
    prev = float("-inf")
    for upper, law in cfg.power_laws.laws:
        rmax = upper if upper != float("inf") else "-3.40282346638529e+38"
        ET.SubElement(
            pls,
            "PowerLawParameters",
            factor=str(law.factor),
            exponent=str(law.exponent),
            offset=str(law.offset),
            rangeMin=str(prev if prev != float("-inf") else "-3.40282346638529e+38"),
            rangeMax=str(rmax),
        )
        prev = upper

    ET.SubElement(
        root,
        "Options",
        doPeel="1" if cfg.do_peel else "0",
        numberOfExtends=str(cfg.number_of_extends),
        minValue=str(cfg.min_value),
    )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def config_to_dict(cfg: MatMapConfig) -> dict[str, Any]:
    return {
        "calibration_unit": cfg.calibration_unit,
        "calibration_points": [{"hu": h, "rho": r} for h, r in cfg.calibration_points],
        "density": {
            "slope": cfg.density.slope,
            "offset": cfg.density.offset,
            "ash_enabled": cfg.density.ash_enabled,
            "ash_offset": cfg.density.ash_offset,
            "ash_divisor": cfg.density.ash_divisor,
            "app_enabled": cfg.density.app_enabled,
            "app_divisor": cfg.density.app_divisor,
        },
        "auto_fit_rho_ct": cfg.auto_fit_rho_ct,
        "power_laws": [
            {
                "upper_bound": b if b != float("inf") else None,
                "factor": law.factor,
                "exponent": law.exponent,
                "offset": law.offset,
            }
            for b, law in cfg.power_laws.laws
        ],
        "do_peel": cfg.do_peel,
        "number_of_extends": cfg.number_of_extends,
        "min_value": cfg.min_value,
    }


def config_from_dict(d: dict[str, Any]) -> MatMapConfig:
    dens = d.get("density", {})
    density = BoneDensityParams(
        slope=float(dens.get("slope", 1)),
        offset=float(dens.get("offset", 0)),
        ash_enabled=bool(dens.get("ash_enabled", True)),
        ash_offset=float(dens.get("ash_offset", 0)),
        ash_divisor=float(dens.get("ash_divisor", 1)),
        app_enabled=bool(dens.get("app_enabled", True)),
        app_divisor=float(dens.get("app_divisor", 1)),
    )
    laws = PowerLawSet()
    for pl in d.get("power_laws") or []:
        ub = pl.get("upper_bound")
        if ub is None:
            ub = float("inf")
        laws.add(
            PowerLaw(float(pl["factor"]), float(pl["exponent"]), float(pl.get("offset", 0))),
            float(ub),
        )
    if not laws.laws:
        laws = PowerLawSet.default_bone()
    pts = []
    for p in d.get("calibration_points") or []:
        pts.append((float(p["hu"]), float(p["rho"])))
    return MatMapConfig(
        calibration_unit=d.get("calibration_unit", "mgHA/cm3"),
        calibration_points=pts,
        density=density,
        auto_fit_rho_ct=bool(d.get("auto_fit_rho_ct", True)),
        power_laws=laws,
        do_peel=bool(d.get("do_peel", True)),
        number_of_extends=int(d.get("number_of_extends", 3)),
        min_value=float(d.get("min_value", 0)),
    )
