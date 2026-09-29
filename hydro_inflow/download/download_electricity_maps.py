#!/usr/bin/env python3
"""Download the daily electricity mix from Electricity Maps (API v4) and save
one JSON file per year, matching the existing CH_<year>_electricity_maps.json
format used by hydro_inflow/build_entsoe_hydro_annual_production.py.

Usage:
    export EM_TOKEN="your-api-key"
    python3 download_electricity_maps.py
    python3 download_electricity_maps.py --zone IT --years 2018 2019 2020
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_URL = "https://api.electricitymaps.com/v4/electricity-mix/past-range"

# Max range per request, in days, per API granularity.
MAX_WINDOW_DAYS = {
    "5_minutes": 1,
    "15_minutes": 2,
    "hourly": 10,
    "daily": 100,
    "monthly": 365,
    "quarterly": 365,
    "yearly": 365,
}


def windows(start, end, step_days):
    cursor = start
    while cursor < end:
        next_cursor = min(cursor + timedelta(days=step_days), end)
        yield cursor, next_cursor
        cursor = next_cursor


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def call_api(session, token, zone, start, end, granularity, breakdown_type, retries=4):
    params = {
        "zone": zone,
        "start": iso(start),
        "end": iso(end),
        "temporalGranularity": granularity,
        "breakdownType": breakdown_type,
    }
    headers = {"auth-token": token}

    for attempt in range(1, retries + 1):
        try:
            r = session.get(BASE_URL, params=params, headers=headers, timeout=60)
        except requests.RequestException as e:
            if attempt == retries:
                raise
            print(f"    network error ({e}); retrying in {2 ** attempt}s", file=sys.stderr)
            time.sleep(2 ** attempt)
            continue

        if r.status_code == 200:
            return r.json()

        if r.status_code == 429 or r.status_code >= 500:
            if attempt == retries:
                r.raise_for_status()
            wait = int(r.headers.get("Retry-After", 2 ** attempt))
            print(f"    HTTP {r.status_code}; retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue

        raise SystemExit(
            f"HTTP {r.status_code} on {start.date()} -> {end.date()}: {r.text[:500]}"
        )

    raise SystemExit("retries exhausted")


def download_year(session, token, zone, year, granularity, breakdown_type, pause):
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    step = MAX_WINDOW_DAYS[granularity]

    points = {}
    meta = None

    for win_start, win_end in windows(start, end, step):
        print(f"  {win_start.date()} -> {win_end.date()}")
        response = call_api(session, token, zone, win_start, win_end, granularity, breakdown_type)

        if meta is None:
            meta = {
                "zone": response.get("zone", zone),
                "temporalGranularity": response.get("temporalGranularity", granularity),
                "unit": response.get("unit"),
            }

        for point in response.get("data", []):
            points[point["datetime"]] = point

        time.sleep(pause)

    if not points:
        return None, 0

    document = dict(meta)
    document["data"] = [points[k] for k in sorted(points)]
    return document, len(points)


def download_electricity_maps(
    years,
    zone="CH",
    granularity="daily",
    breakdown_type="normal",
    output_dir=".",
    token=None,
    pause=0.5,
):
    """Download one electricity_maps JSON file per year. Returns the list of
    written paths (years with no data returned are skipped)."""
    token = token or os.environ.get("EM_TOKEN")
    if not token:
        raise RuntimeError('Missing API key. Set it with: export EM_TOKEN="your-api-key"')

    os.makedirs(output_dir, exist_ok=True)
    session = requests.Session()
    written = []

    for year in years:
        print(f"{zone} {year} ({granularity}):")
        document, n = download_year(session, token, zone, year, granularity, breakdown_type, pause)

        if document is None:
            print("  no data returned, skipping\n")
            continue

        filename = f"{zone}_{year}_electricity_maps.json"
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(document, f, indent=2, ensure_ascii=False)

        methods = {d.get("estimationMethod") for d in document["data"] if d.get("estimationMethod")}
        if any(m and "SANDBOX" in m.upper() for m in methods):
            print(f"  WARNING: sandbox data, not real (estimationMethod = {sorted(methods)})")

        print(f"  {n} points -> {filename}\n")
        written.append(Path(path))

    return written


def main():
    parser = argparse.ArgumentParser(description="Download electricity mix data from Electricity Maps, one file per year.")
    parser.add_argument("--zone", default="CH")
    parser.add_argument("--years", nargs="+", type=int, default=[2017, 2018, 2019, 2020, 2021, 2022, 2023])
    parser.add_argument("--granularity", default="daily", choices=sorted(MAX_WINDOW_DAYS))
    parser.add_argument(
        "--breakdown-type", default="normal", choices=["normal", "flow-traced"],
        help="'normal' matches the existing CH_<year>_electricity_maps.json files; the API default is 'flow-traced'.",
    )
    parser.add_argument("--out", default=".")
    parser.add_argument("--pause", type=float, default=0.5)
    args = parser.parse_args()

    download_electricity_maps(
        years=args.years,
        zone=args.zone,
        granularity=args.granularity,
        breakdown_type=args.breakdown_type,
        output_dir=args.out,
        pause=args.pause,
    )


if __name__ == "__main__":
    main()
