#!/usr/bin/env python3
"""
xmr_sounding_features.py
========================

Build a machine-learning feature table from 30 years of Cape Canaveral (XMR,
WMO 74794) wet-season radiosonde observations.

  * Source: NCEI IGRA 2.2 period-of-record file for station USM00074794
    (one zip file holding every sounding; no per-sounding web requests).
  * Filter: May-September (configurable), start/end years (default 1996-2025).
  * Output: one row per sounding, one column per parameter, plus a data
    dictionary CSV describing every column.

Thermodynamics are implemented here directly (Bolton 1980 vapor pressure,
LCL and theta-e; RK4 pseudoadiabat; virtual-temperature-corrected CAPE/CIN),
so the only dependencies are numpy and pandas.

Usage
-----
    pip install numpy pandas            # (pyarrow optional, for --parquet)
    python xmr_sounding_features.py                      # download + process
    python xmr_sounding_features.py --igra-file USM00074794-data.txt.zip
    python xmr_sounding_features.py --start 1996 --end 2025 --months 5-9 \
        --out xmr_wetseason.csv --workers 8

Conventions
-----------
  * Pressure hPa, temperature degC, heights in m AGL unless the name says MSL.
  * Winds: u/v in knots (meteorological convention, u>0 from the west),
    direction in degrees (from), speeds in knots, shear magnitudes in knots.
  * Layer means in pressure layers are pressure-weighted (uniform-dp grid);
    layer means in height layers are height-weighted.
  * "1000-700" layers start at min(surface pressure, 1000 hPa).
  * Parcel method: MEAN-LAYER. The Lifted Index, Thompson Index, warm-cloud
    depth and the CAPE-based composites (WMSI, EHI, BRN, STP, sig-svr) use a
    parcel with the pressure-weighted mean potential temperature and mixing
    ratio of the lowest 100 hPa (--ml-depth to change). Surface-based and
    most-unstable parcel columns are kept separately for reference; SCP and
    SHIP stay on the most-unstable parcel as those indices are defined.
  * NaN means the sounding did not reach, or did not report, what the
    parameter needs (e.g. humidity missing aloft, winds terminated early).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Station / source
# ---------------------------------------------------------------------------
IGRA_ID = "USM00074794"  # Cape Canaveral / XMR, WMO 74794
IGRA_URLS = [
    f"https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/{IGRA_ID}-data.txt.zip",
    f"https://www.ncei.noaa.gov/pub/data/igra/data/data-por/{IGRA_ID}-data.txt.zip",
]
STATION_ELEV_FALLBACK_M = 3.0  # used only if the surface level carries no height

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
RD = 287.04749
RV = 461.5
EPS = RD / RV
CPD = 1005.7
KAPPA = RD / CPD
G = 9.80665
LV = 2.501e6
T0 = 273.15
MS2KT = 1.943844
DP = 2.0  # analysis grid spacing, hPa
ML_DEPTH = 100.0  # mean-layer parcel depth, hPa (set with --ml-depth)


def _init_worker(ml_depth):
    global ML_DEPTH
    ML_DEPTH = ml_depth

# Column descriptions, filled as features are computed (see put()).
DESC: dict[str, str] = {}


# ===========================================================================
# 1. Download + parse IGRA2
# ===========================================================================
def download(dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using existing {dest}")
        return dest
    last = None
    for url in IGRA_URLS:
        try:
            print(f"Downloading {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "xmr-sounding-features"})
            with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            print(f"  saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
            return dest
        except Exception as exc:  # try the mirror
            last = exc
            print(f"  failed: {exc}")
            if dest.exists():
                dest.unlink()
    raise SystemExit(f"Could not download IGRA2 file: {last}")


def _read_text(path: Path) -> list[str]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            name = [n for n in zf.namelist() if n.endswith(".txt")][0]
            return zf.read(name).decode("ascii", errors="replace").splitlines()
    return path.read_text(errors="replace").splitlines()


def _num(s: str) -> float:
    s = s.strip()
    if not s:
        return np.nan
    v = int(s)
    return np.nan if v in (-9999, -8888) else float(v)


def iter_soundings(lines, years, months):
    """Yield (header, raw_level_lines) for soundings passing the date filter.

    IGRA2 fixed-width header (1-based columns):
      ID 2-12, YEAR 14-17, MONTH 19-20, DAY 22-23, HOUR 25-26, RELTIME 28-31,
      NUMLEV 33-36, LAT 56-62 (x1e4), LON 64-71 (x1e4)
    """
    i, n = 0, len(lines)
    while i < n:
        h = lines[i]
        if not h.startswith("#"):
            i += 1
            continue
        numlev = int(h[32:36])
        yr, mo = int(h[13:17]), int(h[18:20])
        if yr in years and mo in months:
            hdr = {
                "year": yr, "month": mo, "day": int(h[21:23]),
                "hour": int(h[24:26]), "reltime": int(h[27:31]),
                "lat": int(h[55:62]) / 1e4, "lon": int(h[63:71]) / 1e4,
            }
            yield hdr, lines[i + 1: i + 1 + numlev]
        i += 1 + numlev


def parse_levels(raw):
    """IGRA2 data record (1-based columns):
    LVLTYP1 1, LVLTYP2 2, ETIME 4-8, PRESS 10-15 (Pa), PFLAG 16, GPH 17-21 (m),
    ZFLAG 22, TEMP 23-27 (0.1 C), TFLAG 28, RH 29-33 (0.1 %), DPDP 35-39 (0.1 C),
    WDIR 41-45 (deg), WSPD 47-51 (0.1 m/s)
    """
    n = len(raw)
    lt2 = np.zeros(n, int)
    p, z, t, dpd, wd, ws = (np.full(n, np.nan) for _ in range(6))
    for k, L in enumerate(raw):
        L = L.ljust(51)
        lt2[k] = int(L[1]) if L[1].isdigit() else 0
        p[k] = _num(L[9:15]) / 100.0
        z[k] = _num(L[16:21])
        t[k] = _num(L[22:27]) / 10.0
        dpd[k] = _num(L[34:39]) / 10.0
        wd[k] = _num(L[40:45])
        ws[k] = _num(L[46:51]) / 10.0
    return lt2, p, z, t, t - dpd, wd, ws


# ===========================================================================
# 2. Thermodynamics
# ===========================================================================
def es(tc):  # saturation vapor pressure over water, hPa (Bolton 1980)
    return 6.112 * np.exp(17.67 * tc / (tc + 243.5))


def mixr(e, p):
    return EPS * e / (p - e)


def sat_mixr(tc, p):
    return mixr(es(tc), p)


def dewpoint_from_e(e):
    lg = np.log(e / 6.112)
    return 243.5 * lg / (17.67 - lg)


def vtemp(tk, w):
    return tk * (w + EPS) / (EPS * (1.0 + w))


def theta(tk, p):
    return tk * (1000.0 / p) ** KAPPA


def lcl_tk(tk, tdk):  # Bolton eq. 15
    return 1.0 / (1.0 / (tdk - 56.0) + np.log(tk / tdk) / 800.0) + 56.0


def theta_e(tc, tdc, p):  # Bolton eq. 43 (as in MetPy)
    tk = tc + T0
    e = es(tdc)
    w = mixr(e, p)
    tl = lcl_tk(tk, tdc + T0)
    th_l = tk * (1000.0 / (p - e)) ** KAPPA * (tk / tl) ** (0.28 * w)
    return th_l * np.exp((3036.0 / tl - 1.78) * w * (1.0 + 0.448 * w))


def wetbulb(tc, tdc, p):
    """Psychrometric wet-bulb temperature (Newton iteration), degC."""
    tc, tdc, p = np.broadcast_arrays(np.asarray(tc, float), np.asarray(tdc, float), np.asarray(p, float))
    e = es(tdc)
    tw = 0.5 * (tc + tdc)
    for _ in range(25):
        gam = 6.60e-4 * (1.0 + 0.00115 * tw) * p
        f = es(tw) - gam * (tc - tw) - e
        df = es(tw) * 17.67 * 243.5 / (tw + 243.5) ** 2 + gam
        tw = tw - f / df
    return tw


def _dtdp(p, t):  # pseudoadiabatic dT/dp (K per hPa), scalar/fast
    tc = t - T0
    e = 6.112 * math.exp(17.67 * tc / (tc + 243.5))
    rs = EPS * e / (p - e)
    return ((RD * t + LV * rs) / p) / (CPD + (LV * LV * rs * EPS) / (RD * t * t))


def moist_lapse(t_start_k, p_start, p_levels):
    """Integrate a saturated pseudoadiabat from (p_start, t_start) through
    p_levels (monotonic, either direction). RK4, max 5 hPa steps."""
    out = np.empty(len(p_levels))
    t, p = float(t_start_k), float(p_start)
    for i, pn in enumerate(p_levels):
        pn = float(pn)
        nsub = max(1, int(math.ceil(abs(pn - p) / 5.0)))
        h = (pn - p) / nsub
        for _ in range(nsub):
            k1 = _dtdp(p, t)
            k2 = _dtdp(p + h / 2, t + h * k1 / 2)
            k3 = _dtdp(p + h / 2, t + h * k2 / 2)
            k4 = _dtdp(p + h, t + h * k3)
            t += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            p += h
        out[i] = t
    return out


def lift(p0, t0c, w0, p_levels):
    """Lift a parcel (p0, T0, mixing ratio w0) to p_levels (<= p0).
    Returns parcel T (K), parcel virtual T (K), LCL pressure, LCL T (K)."""
    p_levels = np.asarray(p_levels, float)
    tk0 = t0c + T0
    td0 = dewpoint_from_e(w0 * p0 / (EPS + w0)) + T0
    tl = lcl_tk(tk0, td0) if td0 < tk0 else tk0
    pl = p0 * (tl / tk0) ** (1.0 / KAPPA)
    tp = np.empty_like(p_levels)
    dry = p_levels >= pl
    tp[dry] = tk0 * (p_levels[dry] / p0) ** KAPPA
    if (~dry).any():
        tp[~dry] = moist_lapse(tl, pl, p_levels[~dry])
    w = np.where(dry, w0, sat_mixr(tp - T0, p_levels))
    return tp, vtemp(tp, w), pl, tl


# ===========================================================================
# 3. Profile on a uniform pressure grid
# ===========================================================================
class Profile:
    def __init__(self, lt2, p, z, t, td, wd, ws):
        ok = ~np.isnan(p) & ~np.isnan(t)
        if ok.sum() < 5:
            raise ValueError("too few thermo levels")
        # surface: level type 1 if present, else highest pressure
        sfc = np.where(ok & (lt2 == 1))[0]
        isfc = sfc[0] if len(sfc) else np.where(ok)[0][np.argmax(p[ok])]
        psfc = p[isfc]
        if not (950.0 <= psfc <= 1050.0):
            raise ValueError(f"surface pressure {psfc} out of range")
        if np.isnan(td[isfc]):
            raise ValueError("no surface dewpoint")
        self.zsfc = z[isfc] if not np.isnan(z[isfc]) else STATION_ELEV_FALLBACK_M

        keep = ok & (p <= psfc)
        order = np.argsort(-p[keep])
        pt, tt, tdt = p[keep][order], t[keep][order], td[keep][order]
        _, uidx = np.unique(-pt, return_index=True)
        pt, tt, tdt = pt[uidx], tt[uidx], tdt[uidx]
        self.p_top = pt[-1]

        pg = np.arange(psfc, self.p_top, -DP)
        if pg[-1] - self.p_top > 0.1:
            pg = np.append(pg, self.p_top)
        self.p = pg
        self.x = -np.log(pg)  # increasing coordinate for np.interp
        xt = -np.log(pt)
        self.t = np.interp(self.x, xt, tt)

        mv = ~np.isnan(tdt)
        tdg = np.interp(self.x, xt[mv], np.minimum(tdt[mv], tt[mv]))
        self.p_top_moist = pt[mv][-1]
        tdg[pg < self.p_top_moist] = np.nan
        self.td = np.minimum(tdg, self.t)

        self.w = sat_mixr(self.td, pg)                 # NaN where no humidity
        self.tv = vtemp(self.t + T0, np.nan_to_num(self.w))
        # hypsometric heights (MSL)
        zg = np.empty_like(pg)
        zg[0] = self.zsfc
        tvm = 0.5 * (self.tv[1:] + self.tv[:-1])
        zg[1:] = self.zsfc + np.cumsum(RD / G * tvm * np.log(pg[:-1] / pg[1:]))
        self.zmsl = zg
        self.z = zg - self.zsfc                        # AGL

        # winds (pressure levels + height-only levels mapped to pressure)
        wv = ~np.isnan(wd) & ~np.isnan(ws)
        pw = p.copy()
        need = wv & np.isnan(pw) & ~np.isnan(z)
        if need.any():
            zz = z[need]
            inside = (zz >= zg[0]) & (zz <= zg[-1])
            pnew = np.full(zz.shape, np.nan)
            pnew[inside] = np.exp(-np.interp(zz[inside], zg, self.x))
            pw[need] = pnew
        wv &= ~np.isnan(pw) & (pw <= psfc + 0.01)
        self.u = np.full_like(pg, np.nan)
        self.v = np.full_like(pg, np.nan)
        self.p_top_wind = np.nan
        if wv.sum() >= 3:
            spd = ws[wv] * MS2KT
            rad = np.deg2rad(wd[wv])
            uu, vv = -spd * np.sin(rad), -spd * np.cos(rad)
            o = np.argsort(-pw[wv])
            pp, uu, vv = pw[wv][o], uu[o], vv[o]
            _, ui = np.unique(-pp, return_index=True)
            pp, uu, vv = pp[ui], uu[ui], vv[ui]
            xw = -np.log(pp)
            inr = (self.x >= xw[0]) & (self.x <= xw[-1])
            self.u[inr] = np.interp(self.x[inr], xw, uu)
            self.v[inr] = np.interp(self.x[inr], xw, vv)
            self.p_top_wind = pp[-1]

        self.psfc = psfc
        with np.errstate(invalid="ignore"):
            self.rh = 100.0 * es(self.td) / es(self.t)
            self.the = theta_e(self.t, self.td, pg)

    # ---- helpers ---------------------------------------------------------
    def at_p(self, arr, p0):
        if p0 > self.psfc + 1e-6 or p0 < self.p[-1]:
            return np.nan
        return float(np.interp(-math.log(p0), self.x, arr))

    def at_z(self, arr, h):
        if h < 0 or h > self.z[-1]:
            return np.nan
        return float(np.interp(h, self.z, arr))

    def z_at_p(self, p0):
        return self.at_p(self.z, p0)

    def pmask(self, pbot, ptop):
        pbot = min(pbot, self.psfc)
        if ptop < self.p[-1] - 1e-6:
            return None
        return (self.p <= pbot + 1e-6) & (self.p >= ptop - 1e-6)

    def layer(self, arr, pbot, ptop):
        """Exact layer bounds: (p, values) including interpolated end points."""
        pbot = min(pbot, self.psfc)
        m = self.pmask(pbot, ptop)
        if m is None or pbot <= ptop:
            return None, None
        inner = m & (self.p < pbot) & (self.p > ptop)
        pp = np.concatenate([[pbot], self.p[inner], [ptop]])
        vv = np.concatenate([[self.at_p(arr, pbot)], arr[inner], [self.at_p(arr, ptop)]])
        if np.isnan(vv).any():
            return None, None
        return pp, vv

    def pmean(self, arr, pbot, ptop):
        """Pressure-weighted (trapezoidal) layer mean."""
        pp, vv = self.layer(arr, pbot, ptop)
        if pp is None:
            return np.nan
        dp = pp[:-1] - pp[1:]
        return float(np.sum(0.5 * (vv[:-1] + vv[1:]) * dp) / dp.sum())

    def zmean(self, arr, h0, h1, dz=50.0):
        hs = np.arange(h0, h1 + 1e-6, dz)
        if hs[-1] > self.z[-1]:
            return np.nan
        vals = np.interp(hs, self.z, arr)
        return np.nan if np.isnan(vals).any() else float(vals.mean())

    def crossing(self, arr, target):
        """Lowest level (going up) where arr falls through target.
        Returns (z AGL, p)."""
        a = arr - target
        idx = np.where((a[:-1] > 0) & (a[1:] <= 0))[0]
        if not len(idx):
            return np.nan, np.nan
        k = idx[0]
        f = a[k] / (a[k] - a[k + 1])
        zc = self.z[k] + f * (self.z[k + 1] - self.z[k])
        pc = math.exp(-(self.x[k] + f * (self.x[k + 1] - self.x[k])))
        return float(zc), float(pc)


# ===========================================================================
# 4. Parcel analysis
# ===========================================================================
def parcel_analysis(pr: Profile, i0: int, t0c: float, w0: float):
    p = pr.p[i0:]
    tp, tvp, pl, tl = lift(p[0], t0c, w0, p)
    tve = pr.tv[i0:]
    b = tvp - tve
    lnp = np.log(p)
    e = RD * 0.5 * (b[:-1] + b[1:]) * (lnp[:-1] - lnp[1:])  # J/kg per layer
    z = pr.z[i0:]
    zmid = 0.5 * (z[:-1] + z[1:])
    tenv_mid = 0.5 * (pr.t[i0:][:-1] + pr.t[i0:][1:])

    out = dict(lcl_p=pl, lcl_t=tl - T0, lcl_z=pr.at_p(pr.z, pl) if pl >= p[-1] else np.nan)
    li_tp = np.interp(-math.log(500.0), -lnp, tp) if p[-1] <= 500 <= p[0] else np.nan
    out["li"] = pr.at_p(pr.t, 500.0) - (li_tp - T0)

    pos = np.where((p <= pl) & (b > 0))[0]
    nan_keys = ("lfc_p", "lfc_z", "el_p", "el_z", "el_t", "ncape")
    if not len(pos):
        out.update(cape=0.0, cin=0.0, cape_0_3km=0.0, cape_m10_m30=0.0,
                   **{k: np.nan for k in nan_keys})
        return out, tp, p

    ilfc, iel = pos[0], pos[-1]

    def _cross(k):  # interpolate zero crossing between k-1 and k
        if k == 0 or b[k - 1] == b[k]:
            return p[k], z[k]
        f = b[k - 1] / (b[k - 1] - b[k])
        return (math.exp(lnp[k - 1] + f * (lnp[k] - lnp[k - 1])),
                z[k - 1] + f * (z[k] - z[k - 1]))

    lfc_p, lfc_z = _cross(ilfc) if b[max(ilfc - 1, 0)] <= 0 else (min(pl, p[ilfc]), z[ilfc])
    if iel < len(p) - 1:
        f = b[iel] / (b[iel] - b[iel + 1])
        el_p = math.exp(lnp[iel] + f * (lnp[iel + 1] - lnp[iel]))
        el_z = z[iel] + f * (z[iel + 1] - z[iel])
    else:  # still buoyant at top of data
        el_p, el_z = p[-1], z[-1]
    layers = np.arange(len(e))
    between = (layers >= max(ilfc - 1, 0)) & (layers <= iel)
    epos = np.clip(e, 0, None)
    cape = float(epos[between].sum())
    cin = float(np.clip(e[layers < max(ilfc - 1, 0)], None, 0).sum())
    cape03 = float(epos[between & (zmid <= 3000.0)].sum())
    cape_mp = float(epos[between & (tenv_mid <= -10) & (tenv_mid >= -30)].sum())
    out.update(cape=cape, cin=cin, lfc_p=lfc_p, lfc_z=lfc_z, el_p=el_p, el_z=el_z,
               el_t=pr.at_p(pr.t, el_p), cape_0_3km=cape03, cape_m10_m30=cape_mp,
               ncape=cape / (el_z - lfc_z) if el_z - lfc_z > 100 else np.nan)
    return out, tp, p


# ===========================================================================
# 5. Feature extraction
# ===========================================================================
def put(row, name, value, desc):
    row[name] = value
    DESC.setdefault(name, desc)


def wdir_speed(u, v):
    if np.isnan(u) or np.isnan(v):
        return np.nan, np.nan
    spd = math.hypot(u, v)
    d = (270.0 - math.degrees(math.atan2(v, u))) % 360.0
    return (d if spd > 0.05 else 0.0), spd


def pw_mm(pr, pbot, ptop):
    pp, w = pr.layer(pr.w, pbot, ptop)
    if pp is None:
        return np.nan
    q = w / (1 + w)
    pp = pp * 100.0
    return float(np.sum(0.5 * (q[:-1] + q[1:]) * (pp[:-1] - pp[1:])) / G)


def features(hdr, raw):
    """Features from raw IGRA2 level records."""
    return features_from_arrays(hdr, *parse_levels(raw), n_levels=len(raw))


def features_from_arrays(hdr, lt2, p, z, t, td, wd, ws, n_levels=None):
    """Features from level arrays: pressure hPa, height m MSL, T/Td C,
    wind direction deg, wind speed m/s; lt2 == 1 marks the surface level.
    hdr needs year, month, day, hour, reltime (9999 if unknown)."""
    raw = range(n_levels if n_levels is not None else len(p))
    pr = Profile(lt2, p, z, t, td, wd, ws)
    if pr.p_top > 400.0:
        raise ValueError(f"sounding terminated at {pr.p_top:.0f} hPa")

    r: dict = {}
    reltime = hdr["reltime"]
    put(r, "date", f"{hdr['year']:04d}-{hdr['month']:02d}-{hdr['day']:02d}", "UTC date")
    put(r, "year", hdr["year"], "Year")
    put(r, "month", hdr["month"], "Month")
    put(r, "day", hdr["day"], "Day of month")
    put(r, "doy", pd.Timestamp(hdr["year"], hdr["month"], hdr["day"]).dayofyear, "Day of year")
    put(r, "nominal_hour_utc", hdr["hour"] if hdr["hour"] != 99 else np.nan, "IGRA nominal observation hour (UTC)")
    put(r, "release_time_utc", reltime if reltime not in (9999, -9999) else np.nan, "Actual release time HHMM UTC (if reported)")
    put(r, "n_levels", len(raw), "Number of reported levels")
    put(r, "p_top_thermo", pr.p_top, "Top of temperature data (hPa)")
    put(r, "p_top_moist", pr.p_top_moist, "Top of humidity data (hPa)")
    put(r, "p_top_wind", pr.p_top_wind, "Top of wind data (hPa)")

    # ---- surface -----------------------------------------------------------
    ts, tds, ps = pr.t[0], pr.td[0], pr.psfc
    put(r, "sfc_p", ps, "Surface pressure (hPa)")
    put(r, "sfc_t", ts, "Surface temperature (C)")
    put(r, "sfc_td", tds, "Surface dewpoint (C)")
    put(r, "sfc_rh", pr.rh[0], "Surface RH (%)")
    put(r, "sfc_w", pr.w[0] * 1000, "Surface mixing ratio (g/kg)")
    put(r, "sfc_thetae", pr.the[0], "Surface theta-e (K)")
    put(r, "sfc_tw", float(wetbulb(ts, tds, ps)), "Surface wet-bulb temperature (C)")
    sd, ss = wdir_speed(pr.u[0], pr.v[0])
    put(r, "sfc_u", pr.u[0], "Surface u wind (kt)")
    put(r, "sfc_v", pr.v[0], "Surface v wind (kt)")
    put(r, "sfc_wdir", sd, "Surface wind direction (deg)")
    put(r, "sfc_wspd", ss, "Surface wind speed (kt)")

    # ---- mandatory levels ----------------------------------------------------
    for lev in (1000, 925, 850, 700, 500, 400, 300, 250, 200):
        tl_, tdl = pr.at_p(pr.t, lev), pr.at_p(pr.td, lev)
        put(r, f"t{lev}", tl_, f"Temperature at {lev} hPa (C)")
        put(r, f"td{lev}", tdl, f"Dewpoint at {lev} hPa (C)")
        put(r, f"dpd{lev}", tl_ - tdl, f"Dewpoint depression at {lev} hPa (C)")
        put(r, f"rh{lev}", pr.at_p(pr.rh, lev), f"RH at {lev} hPa (%)")
        put(r, f"w{lev}", pr.at_p(pr.w, lev) * 1000, f"Mixing ratio at {lev} hPa (g/kg)")
        put(r, f"thetae{lev}", pr.at_p(pr.the, lev), f"Theta-e at {lev} hPa (K)")
        put(r, f"z{lev}", pr.at_p(pr.zmsl, lev), f"Height of {lev} hPa (m MSL)")
        uu, vv = pr.at_p(pr.u, lev), pr.at_p(pr.v, lev)
        d_, s_ = wdir_speed(uu, vv)
        put(r, f"u{lev}", uu, f"u wind at {lev} hPa (kt)")
        put(r, f"v{lev}", vv, f"v wind at {lev} hPa (kt)")
        put(r, f"wdir{lev}", d_, f"Wind direction at {lev} hPa (deg)")
        put(r, f"wspd{lev}", s_, f"Wind speed at {lev} hPa (kt)")
    put(r, "thickness_1000_500", r["z500"] - r["z1000"], "1000-500 hPa thickness (m)")

    # ---- moisture ------------------------------------------------------------
    cum = {"sfc_850": 850, "sfc_700": 700, "sfc_500": 500, "sfc_400": 400, "sfc_300": 300}
    for k, top in cum.items():
        v = pw_mm(pr, ps, top)
        put(r, f"pw_{k}_in", v / 25.4, f"Precipitable water surface to {top} hPa (in)")
    for bot, top in ((1000, 850), (850, 700), (700, 500), (500, 300)):
        v = pw_mm(pr, bot, top)
        put(r, f"pw_{bot}_{top}_in", v / 25.4, f"Precipitable water in {bot}-{top} hPa layer (in)")
    put(r, "pwat_in", r["pw_sfc_300_in"], "Total precipitable water, surface-300 hPa (in)")
    put(r, "pwat_mm", r["pw_sfc_300_in"] * 25.4, "Total precipitable water, surface-300 hPa (mm)")

    for bot, top in ((1000, 700), (1000, 850), (850, 700), (850, 500), (800, 600), (700, 500), (500, 300)):
        put(r, f"rh_{bot}_{top}", pr.pmean(pr.rh, bot, top), f"Mean RH {bot}-{top} hPa (%)")
    put(r, "w_ml", pr.pmean(pr.w, ps, ps - ML_DEPTH) * 1000, f"Mean mixing ratio of the lowest {ML_DEPTH:g} hPa (g/kg)")
    put(r, "w_0_1km", pr.zmean(pr.w, 0, 1000) * 1000, "Mean mixing ratio 0-1 km AGL (g/kg)")
    put(r, "humidity_index", r["dpd850"] + r["dpd700"] + r["dpd500"],
        "Humidity index (Litynski): DPD850+DPD700+DPD500 (C)")

    # theta-e structure
    m = pr.pmask(700, 400)
    the_min_mid = float(np.nanmin(pr.the[m])) if m is not None and (~np.isnan(pr.the[m])).any() else np.nan
    p_the_min = float(pr.p[m][np.nanargmin(pr.the[m])]) if not np.isnan(the_min_mid) else np.nan
    put(r, "thetae_min_700_400", the_min_mid, "Minimum theta-e 700-400 hPa (K)")
    put(r, "p_thetae_min_700_400", p_the_min, "Pressure of minimum theta-e 700-400 hPa (hPa)")
    put(r, "thetae_deficit", pr.the[0] - the_min_mid, "Surface theta-e minus mid-level minimum (K)")
    put(r, "thetae_850_minus_500", r["thetae850"] - r["thetae500"], "Theta-e 850 minus 500 hPa (K); >0 = convectively unstable")
    mlo = pr.pmask(ps, ps - 150)
    mmid = pr.pmask(650, 500)
    the_max_lo = float(np.nanmax(pr.the[mlo])) if mlo is not None else np.nan
    the_min_65 = float(np.nanmin(pr.the[mmid])) if mmid is not None and (~np.isnan(pr.the[mmid])).any() else np.nan
    put(r, "mdpi", (the_max_lo - the_min_65) / 30.0,
        "Microburst Day Potential Index (Wheeler & Roeder 1996): (max theta-e lowest 150 hPa - min theta-e 650-500 hPa)/30")

    # ---- temperature structure / heights -----------------------------------------
    for name, target in (("frz", 0.0), ("m10", -10.0), ("m20", -20.0), ("m30", -30.0)):
        zc, pc = pr.crossing(pr.t, target)
        lbl = "0C" if target == 0 else f"{int(target)}C"
        put(r, f"z_{name}", zc, f"Height of {lbl} (m AGL)")
        put(r, f"p_{name}", pc, f"Pressure of {lbl} (hPa)")
    tw = wetbulb(pr.t, np.where(np.isnan(pr.td), pr.t - 30, pr.td), pr.p)
    put(r, "z_wbz", pr.crossing(tw, 0.0)[0], "Wet-bulb zero height (m AGL)")
    put(r, "hgz_depth", r["z_m20"] - r["z_m10"], "Depth of -10 to -20C layer (m)")

    def lapse(pb, pt):
        zb, zt = pr.z_at_p(pb), pr.z_at_p(pt)
        return (pr.at_p(pr.t, pb) - pr.at_p(pr.t, pt)) / ((zt - zb) / 1000.0)
    put(r, "lr_850_500", lapse(850, 500), "Lapse rate 850-500 hPa (C/km)")
    put(r, "lr_700_500", lapse(700, 500), "Lapse rate 700-500 hPa (C/km)")
    put(r, "lr_sfc_850", (ts - r["t850"]) / (pr.z_at_p(850) / 1000.0), "Lapse rate surface-850 hPa (C/km)")
    for a, b_ in ((0, 1), (0, 3), (3, 6)):
        t_a, t_b = pr.at_z(pr.t, a * 1000.0), pr.at_z(pr.t, b_ * 1000.0)
        put(r, f"lr_{a}_{b_}km", (t_a - t_b) / (b_ - a), f"Lapse rate {a}-{b_} km AGL (C/km)")

    # ---- classic indices -----------------------------------------------------------
    t850, td850, t700, td700, t500 = r["t850"], r["td850"], r["t700"], r["td700"], r["t500"]
    ki = (t850 - t500) + td850 - (t700 - td700)
    tt = (t850 + td850) - 2 * t500
    put(r, "k_index", ki, "K Index")
    put(r, "total_totals", tt, "Total Totals")
    put(r, "vertical_totals", t850 - t500, "Vertical Totals")
    put(r, "cross_totals", td850 - t500, "Cross Totals")

    # ---- parcels -------------------------------------------------------------------
    sb, sb_tp, sb_p = parcel_analysis(pr, 0, ts, pr.w[0])
    # Mean-layer parcel: pressure-weighted mean potential temperature and
    # mixing ratio of the lowest ML_DEPTH hPa, lifted from the surface
    # (equivalent to lifting from mid-layer: theta and w are conserved
    # dry-adiabatically, so the LCL and moist ascent are identical).
    th_ml = pr.pmean(theta(pr.t + T0, pr.p), ps, ps - ML_DEPTH)
    w_ml = pr.pmean(pr.w, ps, ps - ML_DEPTH)
    t_ml = th_ml * (ps / 1000.0) ** KAPPA - T0
    ml, _, _ = parcel_analysis(pr, 0, t_ml, w_ml)
    mum = (pr.p >= ps - 300) & ~np.isnan(pr.the)
    imu = int(np.argmax(np.where(mum, pr.the, -np.inf)))
    mu, _, _ = parcel_analysis(pr, imu, pr.t[imu], pr.w[imu])
    for tag, d in (("sb", sb), ("ml", ml), ("mu", mu)):
        lab = {"sb": "surface-based", "ml": f"{ML_DEPTH:g}-hPa mean-layer", "mu": "most-unstable (lowest 300 hPa)"}[tag]
        put(r, f"{tag}cape", d["cape"], f"CAPE, {lab} parcel, virtual-T corrected (J/kg)")
        put(r, f"{tag}cin", d["cin"], f"CIN, {lab} parcel (J/kg, negative)")
        put(r, f"{tag}_lcl_p", d["lcl_p"], f"LCL pressure, {lab} (hPa)")
        put(r, f"{tag}_lcl_z", d["lcl_z"], f"LCL height, {lab} (m AGL)")
        put(r, f"{tag}_lcl_t", d["lcl_t"], f"LCL temperature, {lab} (C)")
        put(r, f"{tag}_lfc_p", d["lfc_p"], f"LFC pressure, {lab} (hPa)")
        put(r, f"{tag}_lfc_z", d["lfc_z"], f"LFC height, {lab} (m AGL)")
        put(r, f"{tag}_el_p", d["el_p"], f"Equilibrium level pressure, {lab} (hPa)")
        put(r, f"{tag}_el_z", d["el_z"], f"Equilibrium level height, {lab} (m AGL)")
        put(r, f"{tag}_el_t", d["el_t"], f"Equilibrium level temperature, {lab} (C)")
        put(r, f"{tag}cape_0_3km", d["cape_0_3km"], f"CAPE below 3 km AGL, {lab} (J/kg)")
        put(r, f"{tag}cape_m10_m30", d["cape_m10_m30"], f"CAPE in the -10 to -30C (mixed-phase / charging) layer, {lab} (J/kg)")
        put(r, f"{tag}_ncape", d["ncape"], f"Normalized CAPE, {lab} (J/kg/m)")
    put(r, "mu_p_origin", pr.p[imu], "Most-unstable parcel origin pressure (hPa)")
    put(r, "ml_parcel_theta", th_ml, f"Mean-layer parcel potential temperature, lowest {ML_DEPTH:g} hPa (K)")
    put(r, "ml_parcel_t", t_ml, f"Mean-layer parcel starting temperature at surface pressure (C)")
    put(r, "lifted_index", ml["li"], f"Lifted Index, {ML_DEPTH:g}-hPa mean-layer parcel to 500 hPa (C)")
    put(r, "thompson_index", ki - ml["li"], "Thompson Index = KI - LI (mean-layer LI)")
    put(r, "li_sfc", sb["li"], "Lifted Index, surface parcel (C) [reference only]")
    put(r, "thompson_index_sfc", ki - sb["li"], "Thompson Index with surface-parcel LI [reference only]")
    if not np.isnan(t850) and not np.isnan(td850):
        tp5, _, _, _ = lift(850.0, t850, float(sat_mixr(td850, 850.0)), [500.0])
        put(r, "showalter", t500 - (tp5[0] - T0), "Showalter Index (850 hPa parcel to 500 hPa) (C)")
    else:
        put(r, "showalter", np.nan, "Showalter Index (850 hPa parcel to 500 hPa) (C)")
    put(r, "warm_cloud_depth", r["z_frz"] - r["ml_lcl_z"], "Warm-cloud depth: 0C height minus mean-layer LCL height (m)")

    # convective temperature / CCL (mean mixing ratio of lowest 100 hPa)
    ws_env = sat_mixr(pr.t, pr.p)
    zc, pc = pr.crossing(ws_env, w_ml)
    put(r, "ccl_p", pc, "Convective condensation level pressure (hPa)")
    put(r, "ccl_z", zc, "Convective condensation level height (m AGL)")
    if not np.isnan(pc):
        tccl = pr.at_p(pr.t, pc) + T0
        tc = tccl * (ps / pc) ** KAPPA - T0
    else:
        tc = np.nan
    put(r, "conv_temp", tc, "Convective temperature (C)")
    put(r, "conv_temp_deficit", tc - ts, "Convective temperature minus surface temperature (C)")

    # DCAPE (origin: min 50-hPa-mean theta-e within lowest 400 hPa)
    lo = np.where(pr.p >= ps - 400)[0]
    the_s = pd.Series(pr.the[lo]).rolling(25, center=True, min_periods=13).mean().to_numpy()
    if np.isfinite(the_s).any():
        io = int(lo[np.nanargmin(the_s)])
        two = float(wetbulb(pr.t[io], pr.td[io], pr.p[io])) + T0
        pdn = pr.p[: io + 1][::-1]
        tdn = moist_lapse(two, pr.p[io], pdn)
        tvdn = vtemp(tdn, sat_mixr(tdn - T0, pdn))
        tve = pr.tv[: io + 1][::-1]
        lnp = np.log(pdn)
        bb = tve - tvdn
        dcape = float(np.sum(RD * 0.5 * (bb[:-1] + bb[1:]) * (lnp[1:] - lnp[:-1])))
        put(r, "dcape", dcape, "Downdraft CAPE (J/kg)")
        put(r, "dcape_origin_p", pr.p[io], "DCAPE parcel origin (hPa)")
        put(r, "downdraft_t_sfc", tdn[-1] - T0, "Downdraft parcel temperature at surface (C)")
        put(r, "downdraft_dt", ts - (tdn[-1] - T0), "Surface T minus downdraft T: cold-pool strength proxy (C)")
    else:
        for k in ("dcape", "dcape_origin_p", "downdraft_t_sfc", "downdraft_dt"):
            put(r, k, np.nan, k)

    # WINDEX (McCann 1994)
    hm = r["z_frz"] / 1000.0
    ql = r["w_0_1km"]
    qm = pr.at_p(pr.w, r["p_frz"]) * 1000 if not np.isnan(r["p_frz"]) else np.nan
    if np.isnan(hm) or np.isnan(ql) or np.isnan(qm) or hm <= 0:
        windex = np.nan
    else:
        gam = (ts - 0.0) / hm
        rq = min(ql / 12.0, 1.0)
        arg = hm * rq * (gam ** 2 - 30.0 + ql - 2.0 * qm)
        windex = 5.0 * math.sqrt(arg) if arg > 0 else 0.0
    put(r, "windex", windex, "WINDEX, max potential convective gust (kt), McCann 1994")
    put(r, "wmsi", ml["cape"] * r["thetae_deficit"] / 1000.0,
        "Wet Microburst Severity Index (Pryor & Ellrod): MLCAPE x theta-e deficit / 1000")

    # ---- wind layers ---------------------------------------------------------------
    def put_mean_wind(tag, u_, v_, desc):
        d_, s_ = wdir_speed(u_, v_)
        put(r, f"u_{tag}", u_, f"Mean u wind {desc} (kt)")
        put(r, f"v_{tag}", v_, f"Mean v wind {desc} (kt)")
        put(r, f"wdir_{tag}", d_, f"Mean wind direction {desc} (deg)")
        put(r, f"wspd_{tag}", s_, f"Mean (vector) wind speed {desc} (kt)")
        return d_, s_

    for bot, top in ((1000, 700), (1000, 850), (850, 700), (850, 500), (850, 300), (700, 500), (500, 300)):
        put_mean_wind(f"{bot}_{top}", pr.pmean(pr.u, bot, top), pr.pmean(pr.v, bot, top),
                      f"{bot}-{top} hPa (pressure-weighted)")
    # scalar mean speed 1000-700 (distinct from vector-mean speed)
    put(r, "wspd_scalar_1000_700", pr.pmean(np.hypot(pr.u, pr.v), 1000, 700),
        "Mean scalar wind speed 1000-700 hPa (kt)")
    d17 = r["wdir_1000_700"]
    regime = np.nan if np.isnan(d17) else ("NE" if d17 < 90 else "SE" if d17 < 180 else "SW" if d17 < 270 else "NW")
    put(r, "flow_regime_1000_700", regime, "Quadrant of 1000-700 hPa mean wind (NE/SE/SW/NW)")

    for a, b_ in ((0, 1), (0, 3), (0, 6)):
        put_mean_wind(f"{a}_{b_}km", pr.zmean(pr.u, a * 1e3, b_ * 1e3), pr.zmean(pr.v, a * 1e3, b_ * 1e3),
                      f"{a}-{b_} km AGL (height-weighted)")

    def shear(h0, h1):
        du = pr.at_z(pr.u, h1) - pr.at_z(pr.u, h0)
        dv = pr.at_z(pr.v, h1) - pr.at_z(pr.v, h0)
        return math.hypot(du, dv) if not (np.isnan(du) or np.isnan(dv)) else np.nan

    for a, b_ in ((0, 1), (0, 3), (0, 6), (0, 8)):
        put(r, f"shear_{a}_{b_}km", shear(a * 1e3, b_ * 1e3), f"Bulk shear {a}-{b_} km AGL (kt)")

    def pshear(pb, pt):
        du = pr.at_p(pr.u, pt) - pr.at_p(pr.u, pb)
        dv = pr.at_p(pr.v, pt) - pr.at_p(pr.v, pb)
        return math.hypot(du, dv) if not (np.isnan(du) or np.isnan(dv)) else np.nan

    put(r, "shear_850_500", pshear(850, 500), "Bulk shear 850-500 hPa (kt)")
    put(r, "shear_850_200", pshear(850, 200), "Bulk shear 850-200 hPa (kt)")
    put(r, "shear_sfc_500", pshear(ps, 500), "Bulk shear surface-500 hPa (kt)")

    # Bunkers storm motion and SRH
    u06, v06 = r["u_0_6km"], r["v_0_6km"]
    su = pr.zmean(pr.u, 5500, 6000) - pr.zmean(pr.u, 0, 500)
    sv = pr.zmean(pr.v, 5500, 6000) - pr.zmean(pr.v, 0, 500)
    if not np.isnan(su + sv + u06 + v06) and math.hypot(su, sv) > 0:
        mag = math.hypot(su, sv)
        dev = 7.5 * MS2KT
        rmu, rmv = u06 + dev * sv / mag, v06 - dev * su / mag
        lmu, lmv = u06 - dev * sv / mag, v06 + dev * su / mag
    else:
        rmu = rmv = lmu = lmv = np.nan
    put(r, "bunkers_rm_u", rmu, "Bunkers right-mover u (kt)")
    put(r, "bunkers_rm_v", rmv, "Bunkers right-mover v (kt)")
    put(r, "bunkers_lm_u", lmu, "Bunkers left-mover u (kt)")
    put(r, "bunkers_lm_v", lmv, "Bunkers left-mover v (kt)")

    def srh(h1, cu, cv):
        if np.isnan(cu) or h1 > pr.z[-1]:
            return np.nan
        hs = np.arange(0, h1 + 1, 50.0)
        uu = np.interp(hs, pr.z, pr.u) / MS2KT - cu / MS2KT
        vv = np.interp(hs, pr.z, pr.v) / MS2KT - cv / MS2KT
        if np.isnan(uu).any() or np.isnan(vv).any():
            return np.nan
        return float(np.sum(uu[1:] * vv[:-1] - uu[:-1] * vv[1:]))

    srh1, srh3 = srh(1000, rmu, rmv), srh(3000, rmu, rmv)
    put(r, "srh_0_1km", srh1, "Storm-relative helicity 0-1 km, Bunkers RM (m2/s2)")
    put(r, "srh_0_3km", srh3, "Storm-relative helicity 0-3 km, Bunkers RM (m2/s2)")

    # ---- severe composites ----------------------------------------------------------------
    sh06_ms = r["shear_0_6km"] / MS2KT
    put(r, "ehi_0_1km", ml["cape"] * srh1 / 160000.0, "Energy-Helicity Index 0-1 km (MLCAPE)")
    put(r, "ehi_0_3km", ml["cape"] * srh3 / 160000.0, "Energy-Helicity Index 0-3 km (MLCAPE)")
    brn_sh = 0.5 * ((u06 - pr.zmean(pr.u, 0, 500)) ** 2 + (v06 - pr.zmean(pr.v, 0, 500)) ** 2) / MS2KT ** 2
    put(r, "brn", ml["cape"] / brn_sh if brn_sh and brn_sh > 0 else np.nan, "Bulk Richardson Number (MLCAPE)")
    put(r, "sig_svr", ml["cape"] * sh06_ms, "Craven-Brooks significant severe: MLCAPE x 0-6 km shear (m3/s3)")
    sh_term = np.nan if np.isnan(sh06_ms) else (0.0 if sh06_ms < 10 else min(sh06_ms, 30.0) / 20.0)
    put(r, "scp_fixed", mu["cape"] / 1000.0 * srh3 / 50.0 * sh_term,
        "Supercell Composite (fixed-layer: MUCAPE, 0-3 km SRH, 0-6 km shear)")
    lcl_term = np.clip((2000.0 - ml["lcl_z"]) / 1000.0, 0, 1) if not np.isnan(ml["lcl_z"]) else np.nan
    stp_sh = np.nan if np.isnan(sh06_ms) else (0.0 if sh06_ms < 12.5 else min(sh06_ms, 30.0) / 20.0)
    put(r, "stp_fixed", ml["cape"] / 1500.0 * lcl_term * srh1 / 150.0 * stp_sh,
        "Significant Tornado Parameter (fixed-layer shear/SRH, MLCAPE and ML LCL)")
    # SHIP
    mumr = np.clip(float(sat_mixr(pr.td[imu], pr.p[imu])) * 1000, 11.0, 13.6)
    t5 = min(t500, -5.5)
    shs = np.clip(sh06_ms, 7.0, 27.0)
    ship = mu["cape"] * mumr * r["lr_700_500"] * (-t5) * shs / 42e6
    if mu["cape"] < 1300:
        ship *= mu["cape"] / 1300.0
    if r["lr_700_500"] < 5.8:
        ship *= r["lr_700_500"] / 5.8
    if not np.isnan(r["z_frz"]) and r["z_frz"] < 2400:
        ship *= r["z_frz"] / 2400.0
    put(r, "ship", ship, "Significant Hail Parameter")

    # SWEAT
    d850, s850 = r["wdir850"], r["wspd850"]
    d500, s500 = r["wdir500"], r["wspd500"]
    sweat = 12 * max(td850, 0) + 20 * max(tt - 49, 0) + 2 * s850 + s500
    if (130 <= d850 <= 250 and 210 <= d500 <= 310 and d500 - d850 > 0 and s850 >= 15 and s500 >= 15):
        sweat += 125 * (math.sin(math.radians(d500 - d850)) + 0.2)
    put(r, "sweat", sweat, "Severe Weather Threat Index")
    return r


def _worker(item):
    hdr, raw = item
    try:
        return features(hdr, raw), None
    except Exception as exc:
        return None, f"{hdr['year']:04d}-{hdr['month']:02d}-{hdr['day']:02d} {hdr['hour']:02d}Z: {exc}"


# ===========================================================================
# 6. Main
# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--igra-file", type=Path, default=Path(f"{IGRA_ID}-data.txt.zip"))
    ap.add_argument("--start", type=int, default=1996)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--months", default="5-9", help="e.g. 5-9 or 6,7,8")
    ap.add_argument("--hours", default="", help="optional nominal hours to keep, e.g. 10,15")
    ap.add_argument("--out", type=Path, default=Path("xmr_wetseason_soundings.csv"))
    ap.add_argument("--parquet", action="store_true", help="also write .parquet (needs pyarrow)")
    ap.add_argument("--workers", type=int, default=0, help="0 = all cores")
    ap.add_argument("--ml-depth", type=float, default=100.0,
                    help="mean-layer parcel depth in hPa (default 100; e.g. 50)")
    a = ap.parse_args(argv)

    global ML_DEPTH
    ML_DEPTH = a.ml_depth
    months = (set(range(int(a.months.split("-")[0]), int(a.months.split("-")[1]) + 1))
              if "-" in a.months else {int(m) for m in a.months.split(",")})
    years = set(range(a.start, a.end + 1))
    hours = {int(h) for h in a.hours.split(",")} if a.hours else None

    path = download(a.igra_file) if not a.igra_file.exists() else a.igra_file
    lines = _read_text(path)
    items = [(h, raw) for h, raw in iter_soundings(lines, years, months)
             if hours is None or h["hour"] in hours]
    if not items:
        raise SystemExit("No soundings matched the filters.")
    h0 = items[0][0]
    print(f"{len(items)} soundings matched ({a.start}-{a.end}, months {sorted(months)}). "
          f"Station position in file: {h0['lat']:.3f}N {h0['lon']:.3f}E")

    t_start = time.time()
    rows, errors = [], []
    workers = a.workers or None
    print(f"Parcel method: {ML_DEPTH:g}-hPa mean layer (LI, Thompson, composites)")
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(ML_DEPTH,)) as ex:
        for k, (row, err) in enumerate(ex.map(_worker, items, chunksize=32), 1):
            (rows.append(row) if row else errors.append(err))
            if k % 1000 == 0:
                print(f"  {k}/{len(items)} processed ({time.time() - t_start:.0f}s)")

    # populate DESC in this process for the data dictionary
    for hdr, raw in items:
        try:
            features(hdr, raw)
            break
        except Exception:
            continue

    df = pd.DataFrame(rows).sort_values(["date", "nominal_hour_utc"]).reset_index(drop=True)
    df.to_csv(a.out, index=False, float_format="%.4g")
    if a.parquet:
        df.to_parquet(a.out.with_suffix(".parquet"), index=False)
    dd = pd.DataFrame({"column": df.columns, "description": [DESC.get(c, "") for c in df.columns]})
    dd.to_csv(a.out.with_name(a.out.stem + "_data_dictionary.csv"), index=False)
    if errors:
        a.out.with_name(a.out.stem + "_skipped.txt").write_text("\n".join(errors))

    print(f"\nWrote {len(df)} soundings x {df.shape[1]} columns -> {a.out}  "
          f"({time.time() - t_start:.0f}s); skipped {len(errors)}")
    print("\nSoundings by nominal hour (UTC):")
    print(df["nominal_hour_utc"].value_counts(dropna=False).sort_index().to_string())
    print("\nSoundings per year:")
    print(df.groupby("year").size().to_string())


if __name__ == "__main__":
    main()
