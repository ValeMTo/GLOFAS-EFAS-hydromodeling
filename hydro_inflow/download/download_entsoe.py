from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import logging
import os
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from hydro_inflow.hydro_config import get_config
from hydro_inflow.utils import setup_logging


logger = logging.getLogger(__name__)

BASE_URL = "https://web-api.tp.entsoe.eu/api"

COUNTRIES = {
    "AL": "10YAL-KESH-----5",
    "AT": "10YAT-APG------L",
    "BA": "10YBA-JPCC-----D",
    "BE": "10YBE----------2",
    "BG": "10YCA-BULGARIA-R",
    "CH": "10YCH-SWISSGRIDZ",
    "CZ": "10YCZ-CEPS-----N",
    "DE": "10Y1001A1001A83F",
    "DK": "10Y1001A1001A65H",
    "EE": "10Y1001A1001A39I",
    "ES": "10YES-REE------0",
    "FI": "10YFI-1--------U",
    "FR": "10YFR-RTE------C",
    "GB": "10YGB----------A",
    "GR": "10YGR-HTSO-----Y",
    "HR": "10YHR-HEP------M",
    "HU": "10YHU-MAVIR----U",
    "IE": "10YIE-1001A00010",
    "IT": "10YIT-GRTN-----B",
    "LT": "10YLT-1001A0008Q",
    "LU": "10YLU-CEGEDEL-NQ",
    "LV": "10YLV-1001A00074",
    "ME": "10YCS-CG-TSO---S",
    "MK": "10YMK-MEPSO----8",
    "NL": "10YNL----------L",
    "NO": "10YNO-0--------C",
    "PL": "10YPL-AREA-----S",
    "PT": "10YPT-REN------W",
    "RO": "10YRO-TEL------P",
    "RS": "10YCS-SERBIATSOV",
    "SE": "10YSE-1--------K",
    "SI": "10YSI-ELES-----O",
    "SK": "10YSK-SEPS-----K",
    "XK": "10Y1001A1001A885",
}

ITALY_ZONES = {
    "IT_North": "10Y1001A1001A73I",
}

PSR_TYPES = {
    "B10": "Pumped",
    "B11": "RoR",
    "B12": "Reservoir",
}


def parse_years(value: str) -> list[int]:
    if ":" in value:
        start, end = value.split(":", maxsplit=1)
        return list(range(int(start), int(end) + 1))

    return [int(item.strip()) for item in value.split(",") if item.strip()]


def strip_ns(tag: str) -> str:
    return tag.split("}")[-1]


def parse_a75_timeseries_points(root: ET.Element, target_psr: str) -> pd.DataFrame:
    rows = []

    for ts in root.iter():
        if strip_ns(ts.tag) != "TimeSeries":
            continue

        current_psr = None

        for child in ts.iter():
            if strip_ns(child.tag) == "psrType":
                current_psr = child.text
                break

        if current_psr != target_psr:
            continue

        for period in ts.iter():
            if strip_ns(period.tag) != "Period":
                continue

            start = None
            resolution = None

            for el in period.iter():
                tag = strip_ns(el.tag)

                if tag == "start":
                    start = pd.to_datetime(el.text, utc=True)
                elif tag == "resolution":
                    resolution = el.text

            if start is None:
                continue

            for point in period.iter():
                if strip_ns(point.tag) != "Point":
                    continue

                position = None
                quantity = None

                for el in point:
                    tag = strip_ns(el.tag)

                    if tag == "position":
                        position = int(el.text)
                    elif tag == "quantity":
                        try:
                            quantity = float(el.text)
                        except (TypeError, ValueError):
                            quantity = None

                if position is None or quantity is None:
                    continue

                if resolution == "PT15M":
                    timestamp = start + pd.Timedelta(minutes=15 * (position - 1))
                else:
                    timestamp = start + pd.Timedelta(hours=position - 1)

                rows.append((timestamp, current_psr, quantity))

    df = pd.DataFrame(rows, columns=["datetime", "psrType", "MW"])

    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("datetime")


def fetch_month_a75(
    token: str,
    country_code: str,
    psr_type: str,
    start: datetime,
    end: datetime,
    session: requests.Session,
) -> pd.DataFrame:
    params = {
        "securityToken": token,
        "documentType": "A75",
        "processType": "A16",
        "in_Domain": country_code,
        "psrType": psr_type,
        "periodStart": start.strftime("%Y%m%d%H%M"),
        "periodEnd": end.strftime("%Y%m%d%H%M"),
    }

    response = session.get(BASE_URL, params=params, timeout=120)

    if response.status_code != 200:
        logger.warning(
            "ENTSO-E request failed: country_domain=%s psr=%s start=%s end=%s status=%s",
            country_code,
            psr_type,
            start,
            end,
            response.status_code,
        )
        return pd.DataFrame()

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
        logger.warning(
            "Could not parse ENTSO-E XML: country_domain=%s psr=%s start=%s end=%s",
            country_code,
            psr_type,
            start,
            end,
        )
        return pd.DataFrame()

    return parse_a75_timeseries_points(root, psr_type)


def download_country_year(
    token: str,
    country_code: str,
    year: int,
    sleep_s: float,
) -> pd.DataFrame:
    start_year = datetime(year, 1, 1)
    end_year = datetime(year + 1, 1, 1)

    session = requests.Session()
    all_rows = []

    current = start_year

    while current < end_year:
        next_month = min(current + relativedelta(months=1), end_year)

        for psr_type in PSR_TYPES:
            month = fetch_month_a75(
                token=token,
                country_code=country_code,
                psr_type=psr_type,
                start=current,
                end=next_month,
                session=session,
            )

            if not month.empty:
                all_rows.append(month)

            time.sleep(sleep_s)

        current = next_month

    if not all_rows:
        return pd.DataFrame()

    raw = pd.concat(all_rows, ignore_index=True)

    wide = raw.pivot_table(
        index="datetime",
        columns="psrType",
        values="MW",
        aggfunc="mean",
    ).sort_index()

    for psr_type in PSR_TYPES:
        if psr_type not in wide.columns:
            wide[psr_type] = pd.NA

    wide = wide[list(PSR_TYPES.keys())]
    wide = wide.rename(columns=PSR_TYPES)

    wide = wide.resample("1h").mean()

    start_cut = pd.Timestamp(f"{year}-01-01 00:00:00", tz="UTC")
    end_cut = pd.Timestamp(f"{year + 1}-01-01 00:00:00", tz="UTC")

    wide = wide[(wide.index >= start_cut) & (wide.index < end_cut)]

    wide["Total"] = (
        wide[["Pumped", "RoR", "Reservoir"]]
        .astype(float)
        .fillna(0)
        .sum(axis=1)
    )

    return wide


def download_entsoe_hydro(
    years: str = "2015:2019",
    token: str | None = None,
    output_dir: Path | None = None,
    sleep_s: float = 0.12,
    overwrite: bool = False,
) -> list[Path]:
    token = token or os.environ.get("ENTSOE_API_TOKEN")

    if not token:
        raise RuntimeError(
            "ENTSOE_API_TOKEN is not set. "
            "Set it with: export ENTSOE_API_TOKEN='<your-token>'"
        )

    cfg = get_config()

    if output_dir is None:
        output_dir = Path(cfg["entsoe_hydro_hourly_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)

    parsed_years = parse_years(years)
    output_files: list[Path] = []

    logger.info("Downloading ENTSO-E hydro production for years: %s", parsed_years)
    logger.info("Output directory: %s", output_dir)

    for year in tqdm(parsed_years, desc="Years"):
        output = output_dir / f"Europe_Hydro_{year}.csv"

        if output.exists() and not overwrite:
            logger.info("Already exists, skipping: %s", output)
            output_files.append(output)
            continue

        year_data = pd.DataFrame()

        for country, bidding_zone_code in tqdm(
            COUNTRIES.items(),
            desc=f"Countries {year}",
            leave=False,
        ):
            country_data = download_country_year(
                token=token,
                country_code=bidding_zone_code,
                year=year,
                sleep_s=sleep_s,
            )

            if country_data.empty:
                logger.warning("%s %s returned no ENTSO-E hydro data.", country, year)
                continue

            country_data = country_data.rename(
                columns={
                    "Pumped": f"{country}_Pumped",
                    "RoR": f"{country}_RoR",
                    "Reservoir": f"{country}_Reservoir",
                    "Total": f"{country}_Total",
                }
            )

            year_data = pd.concat([year_data, country_data], axis=1)

        if year_data.empty:
            logger.warning("%s produced an empty ENTSO-E hydro dataframe.", year)
            continue

        year_data = year_data.sort_index()
        year_data.to_csv(output)
        output_files.append(output)

        logger.info("Saved %s with shape %s", output, year_data.shape)

    logger.info("ENTSO-E hydro download completed.")

    return output_files

def download_entsoe_italy_zones(
    years: str = "2015:2019",
    token: str | None = None,
    output_dir: Path | None = None,
    sleep_s: float = 0.12,
    overwrite: bool = False,
) -> list[Path]:
    token = token or os.environ.get("ENTSOE_API_TOKEN")

    if not token:
        raise RuntimeError(
            "ENTSOE_API_TOKEN is not set. "
            "Set it with: export ENTSOE_API_TOKEN='<your-token>'"
        )

    cfg = get_config()

    if output_dir is None:
        output_dir = Path(cfg["italy_bidding_zone_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)

    parsed_years = parse_years(years)
    output_files: list[Path] = []

    logger.info("Downloading ENTSO-E Italian bidding-zone hydro for years: %s", parsed_years)
    logger.info("Output directory: %s", output_dir)

    for year in tqdm(parsed_years, desc="Italy zones years"):
        for zone_name, zone_code in ITALY_ZONES.items():
            output = output_dir / f"Italy_{zone_name}_{year}.csv"

            if output.exists() and not overwrite:
                logger.info("Already exists, skipping: %s", output)
                output_files.append(output)
                continue

            zone_data = download_country_year(
                token=token,
                country_code=zone_code,
                year=year,
                sleep_s=sleep_s,
            )

            if zone_data.empty:
                logger.warning("%s %s returned no ENTSO-E hydro data.", zone_name, year)
                continue

            zone_data = zone_data.sort_index()
            zone_data.to_csv(output)
            output_files.append(output)

            logger.info("Saved %s with shape %s", output, zone_data.shape)

    logger.info("ENTSO-E Italian bidding-zone download completed.")

    return output_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download hourly hydro generation from ENTSO-E Transparency Platform."
    )

    parser.add_argument(
        "--years",
        default="2015:2019",
        help="Years to download. Example: '2015:2019' or '2019'.",
    )

    parser.add_argument(
        "--sleep-s",
        type=float,
        default=0.12,
        help="Sleep time between ENTSO-E API requests.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing yearly ENTSO-E CSV files.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Use DEBUG logging.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    download_entsoe_hydro(
        years=args.years,
        sleep_s=args.sleep_s,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()