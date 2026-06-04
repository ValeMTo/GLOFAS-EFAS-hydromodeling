from __future__ import annotations

from pathlib import Path
import logging
import os
import urllib.request
import numpy as np


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def get_repo_root() -> Path:
    return Path(
        os.environ.get(
            "HYDRO_REPO_ROOT",
            Path(__file__).resolve().parents[1],
        )
    )


def get_data_root(repo_root: Path | None = None) -> Path:
    if repo_root is None:
        repo_root = get_repo_root()

    return Path(
        os.environ.get(
            "HYDRO_DATA_ROOT",
            repo_root / "data" / "hydro_workflow",
        )
    )


def ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path, overwrite: bool = False) -> None:
    logger = logging.getLogger(__name__)

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        logger.info("Already exists, skipping: %s", destination)
        return

    temporary_destination = destination.with_suffix(destination.suffix + ".tmp")

    logger.info("Downloading: %s", url)
    logger.info("Destination: %s", destination)

    try:
        urllib.request.urlretrieve(url, temporary_destination)
        temporary_destination.replace(destination)
    except Exception:
        temporary_destination.unlink(missing_ok=True)
        logger.exception("Download failed for %s", destination)
        raise

    logger.info("Downloaded: %s", destination)


def require_file(path: Path, label: str | None = None) -> None:
    if not path.exists():
        name = label or "required file"
        raise FileNotFoundError(f"Missing {name}: {path}")


def require_directory(path: Path, label: str | None = None) -> None:
    if not path.exists():
        name = label or "required directory"
        raise FileNotFoundError(f"Missing {name}: {path}")
    
def km_to_deg_lat(km: float) -> float:
    return km / 111.0


def km_to_deg_lon(km: float, lat: float) -> float:
    cos_lat = np.cos(np.deg2rad(lat))

    if np.isclose(cos_lat, 0.0):
        return np.inf

    return km / (111.0 * cos_lat)


def approx_dist_km(lon: float, lat: float, lon2, lat2):
    dlon = lon2 - lon
    dlat = lat2 - lat

    dx_km = dlon * 111.0 * np.cos(np.deg2rad(lat))
    dy_km = dlat * 111.0

    return np.sqrt(dx_km**2 + dy_km**2)