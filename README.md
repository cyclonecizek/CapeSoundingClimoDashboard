# XMR wet-season sounding climatology

Interactive climatologies of 30 years (1996–2025; wet season and full year) of Cape Canaveral
(XMR, WMO 74794) radiosonde soundings, with the latest sounding ranked against it.

- `index.html`: wet-season dashboard (May–September; data embedded).
- `annual.html`: year-round dashboard (all months; data embedded).
- `latest.json`: shared by both pages; the most recent sounding's parameters, refreshed hourly by
  `.github/workflows/latest-sounding.yml`.
- `update_latest.py`: finds the newest sounding (University of Wyoming, with
  NOAA IGRA2 as backup) and computes its parameters.
- `xmr_sounding_features.py`: the parameter calculations used for both the
  climatology and the latest sounding.

Data: NOAA NCEI Integrated Global Radiosonde Archive (IGRA 2.2) and the
University of Wyoming upper-air archive.
