"""2050 reservoir state-of-charge over the system-price background, the two
climate years side by side, for the three figures that share these networks:

    images/2050/soc_price.png
        original PyPSA baseline, GloFAS-SABER, EFAS-SABER
    images/2050_samevol_baseline/soc_price.png
        same-volume PyPSA baseline, GloFAS-SABER, EFAS-SABER
    images/2050_samevol_baseline/soc_price_with_original_baseline.png
        both baselines, GloFAS-SABER, EFAS-SABER; the price background stays
        the one of the same-volume figure, so the original baseline only adds
        its SOC line and its maximum

The four networks are loaded once; the rest of the 2050 post-processing is not
rerun.
"""

from pathlib import Path
import pypsa

from hydro_inflow.utils import setup_logging
from postprocessing_workflow import processing_2050 as p50

RESULTS = Path(__file__).resolve().parents[1] / "hydro_results"
IMAGES = RESULTS / "images"

GLOFAS = "PyPSA + GloFAS-SABER"
EFAS = "PyPSA + EFAS-SABER"
ORIGINAL = "PyPSA baseline (original volume)"
SAMEVOL = "PyPSA baseline (same volume)"
NETWORK_PATHS = {
    ORIGINAL: RESULTS / "PyPSA_2050" / "base_s_100___2050.nc",
    SAMEVOL: RESULTS / "PyPSA_2050_samevol" / "base_s_100___2050.nc",
    GLOFAS: RESULTS / "GloFAS-SABER_2050" / "base_s_100___2050.nc",
    EFAS: RESULTS / "EFAS-SABER_2050" / "base_s_100___2050.nc",
}

p50.SCENARIO_COLORS[ORIGINAL] = "#8B1E2D"
p50.SCENARIO_COLORS[SAMEVOL] = p50.SCENARIO_COLORS[p50.BASELINE_NAME]
p50.SCENARIO_SHORT_NAMES[ORIGINAL] = "PyPSA baseline"
p50.SCENARIO_SHORT_NAMES[SAMEVOL] = "same-volume baseline"


def main():
    setup_logging()
    missing = [str(p) for p in NETWORK_PATHS.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing 2050 networks:\n" + "\n".join(missing))

    nets = {name: pypsa.Network(path, engine="netcdf4")
            for name, path in NETWORK_PATHS.items()}

    original_trio = {p50.BASELINE_NAME: nets[ORIGINAL], GLOFAS: nets[GLOFAS], EFAS: nets[EFAS]}
    samevol_trio = {p50.BASELINE_NAME: nets[SAMEVOL], GLOFAS: nets[GLOFAS], EFAS: nets[EFAS]}

    figures = [
        (original_trio, None, IMAGES / "2050" / "soc_price.png"),
        (samevol_trio, None, IMAGES / "2050_samevol_baseline" / "soc_price.png"),
        (nets, samevol_trio,
         IMAGES / "2050_samevol_baseline" / "soc_price_with_original_baseline.png"),
    ]
    for networks, price_networks, output in figures:
        p50.make_soc_price_figure(networks=networks, output_path=output,
                                  price_networks=price_networks)
        print(f"written {output}")


if __name__ == "__main__":
    main()
