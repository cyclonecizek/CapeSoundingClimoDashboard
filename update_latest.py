#!/usr/bin/env python3
"""
update_latest.py
================

Find the most recent Cape Canaveral (XMR, WMO 74794) sounding, compute every
parameter with xmr_sounding_features.py, and write latest.json for the
climatology dashboard.

Sources, newest wins:
  1. University of Wyoming upper-air archive (near real time, via Siphon 0.11+)
  2. NOAA NCEI IGRA2 year-to-date file (lags a day or two; very reliable)

If neither source has anything newer than the existing latest.json, the file is
left untouched, so a scheduled job only commits when a new sounding arrives.

    pip install "siphon>=0.11.0" numpy pandas
    python update_latest.py                    # normal run
    python update_latest.py --igra-file X.zip  # use a local IGRA file (testing)
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import time
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import xmr_sounding_features as xf

WMO = "74794"
Y2D_DIR = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-y2d/"
LOOKBACK_H = 36
UA = {"User-Agent": "xmr-latest-sounding (GitHub Actions)"}


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def from_wyoming(since: datetime | None):
    """Walk back hour by hour; Wyoming answers 404/no-data when no sounding exists."""
    try:
        from siphon.simplewebservice.wyoming import WyomingUpperAir
    except ImportError:
        print("Siphon not installed; skipping Wyoming")
        return None
    t = _utcnow().replace(minute=0, second=0, microsecond=0)
    stop = max(since or datetime.min, t - timedelta(hours=LOOKBACK_H))
    while t > stop:
        try:
            df = WyomingUpperAir.request_data(t, WMO)
        except Exception as exc:  # no sounding at this hour, or server hiccup
            msg = str(exc).splitlines()[0][:80]
            print(f"  Wyoming {t:%Y-%m-%d %HZ}: none ({msg})")
            t -= timedelta(hours=1)
            time.sleep(0.5)
            continue
        df = df.dropna(subset=["pressure"]).sort_values("pressure", ascending=False)
        n = len(df)
        lt2 = np.zeros(n, int)
        lt2[0] = 1  # lowest level is the surface
        arrs = [df[c].to_numpy(float) for c in ("pressure", "height", "temperature", "dewpoint", "direction")]
        ws = df["speed"].to_numpy(float) / xf.MS2KT  # knots -> m/s
        hdr = dict(year=t.year, month=t.month, day=t.day, hour=t.hour, reltime=9999)
        print(f"  Wyoming {t:%Y-%m-%d %HZ}: found ({n} levels)")
        return t, "University of Wyoming", hdr, (lt2, *arrs, ws), n
    return None


def from_igra(local: Path | None):
    if local:
        lines = xf._read_text(local)
    else:
        idx = urllib.request.urlopen(urllib.request.Request(Y2D_DIR, headers=UA), timeout=60).read().decode()
        names = sorted(set(re.findall(rf'href="({xf.IGRA_ID}-data[^"]*\.zip)"', idx)))
        if not names:
            print("  IGRA: station file not found in year-to-date directory")
            return None
        url = Y2D_DIR + names[-1]
        print(f"  IGRA: {url}")
        blob = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300).read()
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            lines = zf.read(zf.namelist()[0]).decode("ascii", "replace").splitlines()
    last = None
    for hdr, raw in xf.iter_soundings(lines, set(range(1900, 2200)), set(range(1, 13))):
        last = (hdr, raw)
    if not last:
        return None
    hdr, raw = last
    hour = hdr["hour"] if hdr["hour"] != 99 else (hdr["reltime"] // 100 if hdr["reltime"] != 9999 else 0)
    t = datetime(hdr["year"], hdr["month"], hdr["day"], hour)
    print(f"  IGRA newest: {t:%Y-%m-%d %HZ}")
    return t, "NOAA NCEI IGRA2", hdr, xf.parse_levels(raw), len(raw)


def clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    return v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("latest.json"))
    ap.add_argument("--igra-file", type=Path, help="local IGRA2 file instead of downloading")
    ap.add_argument("--no-wyoming", action="store_true")
    a = ap.parse_args(argv)

    since = None
    if a.out.exists():
        try:
            since = datetime.strptime(json.loads(a.out.read_text())["valid"], "%Y-%m-%dT%H:%MZ")
            print(f"Current latest.json: {since:%Y-%m-%d %HZ}")
        except Exception:
            pass

    cands = []
    if not a.no_wyoming:
        print("Checking University of Wyoming ...")
        try:
            r = from_wyoming(since)
            if r:
                cands.append(r)
        except Exception as exc:
            print(f"  Wyoming failed: {exc}")
    if not cands or a.igra_file:
        print("Checking NOAA IGRA2 ...")
        try:
            r = from_igra(a.igra_file)
            if r:
                cands.append(r)
        except Exception as exc:
            print(f"  IGRA failed: {exc}")

    cands = [c for c in cands if since is None or c[0] > since]
    if not cands:
        print("No newer sounding; latest.json unchanged.")
        return
    t, source, hdr, arrays, nlev = max(cands, key=lambda c: c[0])
    feats = xf.features_from_arrays(hdr, *arrays, n_levels=nlev)
    out = {
        "station": "XMR 74794",
        "valid": t.strftime("%Y-%m-%dT%H:%MZ"),
        "source": source,
        "generated": _utcnow().strftime("%Y-%m-%dT%H:%MZ"),
        "values": {k: clean(v) for k, v in feats.items()},
    }
    a.out.write_text(json.dumps(out, separators=(",", ":")))
    print(f"Wrote {a.out}: {out['valid']} from {source} ({len(out['values'])} parameters)")


if __name__ == "__main__":
    main()
