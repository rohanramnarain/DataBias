"""
Dataset Harvester + Plain-Work Visuals (No-API-Key Edition)
-----------------------------------------------------------
Pulls a set of public datasets (no API key required) and generates Work-style, bias-aware
starter visuals. Endpoints favor JSON where possible; CSV reads use defensive parsing.

INCLUDED (no key required):
- Justice & Policing: COMPAS (ProPublica), HMDA (Browser CSV path), Mapping Police Violence (stable link fallback*)
- Health & Environment: BRFSS (CDC Socrata JSON), EPA TRI (Envirofacts CSV w/ robust parsing),
                        HRSA HPSA (Socrata JSON), CMS Hospital General Info (CMS metastore -> downloadURL)
- Transportation: NYC TLC Trip sample (Socrata)
- Clinical Research: ClinicalTrials.gov v2 (filters)

EXCLUDED (require key or are too brittle without one):
- College Scorecard (requires API key)
- FBI NIBRS (key / CDE gateway, not included here)
- EJScreen (best via downloads UI or EPA services; omitted here)

*Note: Mapping Police Violence has reorganized hosting. We try a couple of common mirrors/paths.
If all fail, we skip gracefully.

USAGE
- List datasets:  python dataset_harvester_no_key.py --list
- Run all:        python dataset_harvester_no_key.py --run --max-rows 20000
- Pick some:      python dataset_harvester_no_key.py --run --only hmda,compas,mpv,brfss,tri,hrsa,cms,nyctlc,clinicaltrials

OUTPUTS
- ./data/          : CSV/JSON extracts
- ./figures/       : PNG charts (plain style; one chart per figure; default colors)
- ./logs/harvest.log: Run log
"""
import os
import io
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# ------------------------------
# Setup
# ------------------------------
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
FIG_DIR = BASE / "figures"
LOG_DIR  = BASE / "logs"
for d in [DATA_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "harvest.log"

DEFAULT_HEADERS = {
    "User-Agent": "pharmachute-workstyle-harvester/1.0",
    "Accept": "*/*",
}

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(msg)

def save_df(df: pd.DataFrame, name: str, fmt: str = "csv"):
    p = DATA_DIR / f"{name}.{fmt}"
    if fmt == "csv":
        df.to_csv(p, index=False)
    elif fmt == "parquet":
        df.to_parquet(p, index=False)
    else:
        raise ValueError("Unsupported format")
    log(f"Saved {name}.{fmt} ({len(df):,} rows)")
    return p

def fetch(url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None, timeout: int = 90) -> requests.Response:
    h = DEFAULT_HEADERS.copy()
    if headers:
        h.update(headers)
    log(f"GET {url}")
    r = requests.get(url, headers=h, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r

def work_plain_savefig(figpath: Path, tight=True):
    if tight:
        plt.tight_layout()
    plt.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close()

# ------------------------------
# Dataset modules (no-key)
# ------------------------------

def propublica_compas() -> Optional[Path]:
    """COMPAS Broward County dataset released by ProPublica (raw GitHub)."""
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    try:
        df = pd.read_csv(url)
        return save_df(df, "compas_broward")
    except Exception as e:
        log(f"COMPAS error: {e}")
        return None

def cfpb_hmda_sample(year: int = 2022, state: str = "NY", max_rows: int = 50000) -> Optional[Path]:
    """
    HMDA public Browser API CSV path (no key). We send headers and accept CSV explicitly.
    If blocked, user can switch to published downloads and filter locally.
    """
    url = f"https://ffiec.cfpb.gov/v2/data-browser-api/view/lar.csv?year={year}&state={state}"
    try:
        r = fetch(url, headers={"Accept": "text/csv"}, timeout=180)
        # Defensive parse
        df = pd.read_csv(io.BytesIO(r.content), on_bad_lines="skip", low_memory=False)
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=42)
        return save_df(df, f"hmda_{year}_{state}_sample")
    except Exception as e:
        log(f"HMDA error: {e}")
        return None

def mapping_police_violence() -> Optional[Path]:
    """
    Mapping Police Violence fatalities data.
    Try a couple of known mirrors/paths. If they shift, skip gracefully.
    """
    candidates = [
        # MPV historical raw CSV (may move)
        "https://raw.githubusercontent.com/MappingPoliceViolence/mapping-police-violence-data/main/data.csv",
        # Some forks/mirrors keep an updated copy:
        "https://raw.githubusercontent.com/ajdm/mapping-police-violence-data/main/data.csv",
    ]
    for url in candidates:
        try:
            df = pd.read_csv(url)
            if not df.empty:
                return save_df(df, "mapping_police_violence")
        except Exception as e:
            log(f"MPV try failed: {e}")
    log("MPV: all candidate URLs failed; skipping.")
    return None

def cdc_brfss_prevalence_sample(limit: int = 10000, dataset_id: str = "9nys-gcy2") -> Optional[Path]:
    """
    CDC BRFSS via Socrata JSON (no token for small pulls).
    dataset_id is configurable to target specific indicators.
    """
    url = f"https://chronicdata.cdc.gov/resource/{dataset_id}.json?$limit={limit}"
    try:
        r = fetch(url, timeout=120)
        df = pd.DataFrame(r.json())
        if df.empty:
            log("BRFSS: empty response; consider another dataset_id.")
            return None
        return save_df(df, f"cdc_brfss_{dataset_id}_sample")
    except Exception as e:
        log(f"BRFSS error: {e}")
        return None

def epa_tri_basic_sample(rows: int = 2000) -> Optional[Path]:
    """
    EPA TRI via Envirofacts efservice (CSV). Can be finicky; we use small slice + robust parsing.
    Example: TRI_RELEASE_QTY table first 2000 rows.
    """
    url = f"https://enviro.epa.gov/facts/tri/efservice/TRI_RELEASE_QTY/ROWS/0:{rows-1}/CSV"
    try:
        df = pd.read_csv(url, on_bad_lines="skip", low_memory=False)
        return save_df(df, "epa_tri_release_qty_sample")
    except Exception as e:
        log(f"EPA TRI error: {e}")
        return None

def hrsa_hpsa_sample(limit: int = 10000) -> Optional[Path]:
    """
    HRSA HPSA via Socrata JSON.
    """
    url = f"https://data.hrsa.gov/resource/hgdw-ubwe.json?$limit={limit}"
    try:
        r = fetch(url, timeout=120)
        df = pd.json_normalize(r.json())
        if df.empty:
            log("HRSA: empty response.")
            return None
        return save_df(df, "hrsa_hpsa_sample")
    except Exception as e:
        log(f"HRSA error: {e}")
        return None

def cms_hospital_general_info_download() -> Optional[Path]:
    """
    CMS Provider Data Catalog: use metastore to get current downloadURL; read CSV (no key).
    """
    try:
        meta = fetch("https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/xubh-q36u").json()
        dist = meta.get("distribution", [])
        if not dist:
            log("CMS: no distribution in metastore.")
            return None
        dl = dist[0].get("downloadURL")
        if not dl:
            log("CMS: downloadURL missing.")
            return None
        df = pd.read_csv(dl, on_bad_lines="skip", low_memory=False)
        return save_df(df, "cms_hospital_general_info")
    except Exception as e:
        log(f"CMS PDC error: {e}")
        return None

def nyc_tlc_sample(limit: int = 20000) -> Optional[Path]:
    """
    NYC TLC sample via Socrata (no key). We try multiple resource IDs.
    """
    candidates = [
        "gnke-dk5s",  # Yellow Taxi (may 404 depending on archival moves)
        "2yzn-sicd",  # For-Hire Vehicle trips (works per your run)
        "w7fs-fd9i",  # Green Taxi
    ]
    for rid in candidates:
        url = f"https://data.cityofnewyork.us/resource/{rid}.csv?$limit={limit}"
        try:
            r = fetch(url, headers={"Accept": "text/csv"})
            df = pd.read_csv(io.BytesIO(r.content), on_bad_lines="skip", low_memory=False)
            if not df.empty:
                return save_df(df, f"nyc_tlc_{rid}_sample")
        except Exception as e:
            log(f"NYC TLC try {rid} error: {e}")
            continue
    return None

def clinicaltrials_results_sample(n: int = 500) -> Optional[Path]:
    """
    ClinicalTrials.gov v2: filter-based paging (no key). Pull Completed + Interventional.
    """
    base = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "format": "json",
        "filter.overallStatus": "Completed",
        "filter.studyType": "Interventional",
        "pageSize": min(n, 100)
    }
    frames = []
    next_token = ""
    fetched = 0
    try:
        while fetched < n:
            if next_token:
                params["pageToken"] = next_token
            r = fetch(base, params=params, timeout=90)
            js = r.json()
            studies = js.get("studies", [])
            if not studies:
                break
            frames.append(pd.json_normalize(studies))
            fetched += len(studies)
            next_token = js.get("nextPageToken", "")
            if not next_token:
                break
        if not frames:
            log("ClinicalTrials: no data.")
            return None
        out = pd.concat(frames, ignore_index=True)
        return save_df(out, "clinicaltrials_completed_interventional_sample")
    except Exception as e:
        log(f"ClinicalTrials error: {e}")
        return None

# ------------------------------
# Work-style visuals (plain marks)
# ------------------------------

def visual_hmda_denial_and_highprice(sample_csv: Path, outprefix: str = "hmda_work_bars"):
    """Paired bars: denial rate & high-price loan share by race."""
    df = pd.read_csv(sample_csv, low_memory=False)
    # Denial: action_taken == 3 (denied)
    df["denied"] = (df.get("action_taken", pd.Series([np.nan]*len(df))).fillna(-1).astype(float) == 3).astype(int)
    # High-price proxy: rate_spread > 0
    if "rate_spread" in df.columns:
        df["high_price"] = pd.to_numeric(df["rate_spread"], errors="coerce").fillna(0).gt(0).astype(int)
    else:
        df["high_price"] = 0
    race = df.get("applicant_race_1", pd.Series(["Unknown"] * len(df))).astype(str)
    grp = df.assign(race=race).groupby("race").agg(
        apps=("denied", "count"),
        denial_rate=("denied", "mean"),
        high_price_share=("high_price", "mean")
    ).reset_index()
    grp = grp[grp["apps"] >= 100].sort_values("denial_rate", ascending=False)

    # Bars for denial rate
    plt.figure(figsize=(8, 5))
    plt.bar(grp["race"].astype(str), grp["denial_rate"] * 100)
    plt.ylabel("Denials per 100 applications")
    plt.title("HMDA: Denial rate by applicant race (sample)")
    work_plain_savefig(FIG_DIR / f"{outprefix}_denials.png")

    # Bars for high-price share
    plt.figure(figsize=(8, 5))
    plt.bar(grp["race"].astype(str), grp["high_price_share"] * 100)
    plt.ylabel("High-price loans per 100 originations (proxy)")
    plt.title("HMDA: High-price share by applicant race (sample)")
    work_plain_savefig(FIG_DIR / f"{outprefix}_highprice.png")

def visual_compas_fpr_fnr(compas_csv: Path, outprefix: str = "compas_work_parity"):
    """Side-by-side bars of FPR/FNR by race; plus a base-rate CSV (tables-first)."""
    df = pd.read_csv(compas_csv, low_memory=False)
    if "two_year_recid" in df.columns:
        y = pd.to_numeric(df["two_year_recid"], errors="coerce").fillna(0).astype(int)
    elif "is_recid" in df.columns:
        y = pd.to_numeric(df["is_recid"], errors="coerce").fillna(0).astype(int)
    else:
        log("COMPAS: recidivism outcome not found; skipping visual.")
        return

    if "decile_score" in df.columns:
        pred = (pd.to_numeric(df["decile_score"], errors="coerce").fillna(0) >= 5).astype(int)
    elif "score_text" in df.columns:
        pred = df["score_text"].astype(str).str.lower().isin(["high", "medium"]).astype(int)
    else:
        log("COMPAS: risk score not found; skipping visual.")
        return

    race = df.get("race", pd.Series(["Unknown"] * len(df))).astype(str)
    recs = []
    for r, sub in df.assign(race=race).groupby("race"):
        if len(sub) < 100:
            continue
        idx = sub.index
        tp = int(((pred == 1) & (y == 1)).loc[idx].sum())
        tn = int(((pred == 0) & (y == 0)).loc[idx].sum())
        fp = int(((pred == 1) & (y == 0)).loc[idx].sum())
        fn = int(((pred == 0) & (y == 1)).loc[idx].sum())
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
        base = y.loc[idx].mean()
        recs.append({"race": r, "FPR": fpr, "FNR": fnr, "base_rate": base, "n": len(sub)})
    out = pd.DataFrame(recs).dropna().sort_values("FPR", ascending=False)
    if out.empty:
        log("COMPAS: insufficient data for parity chart.")
        return

    # FPR bars
    plt.figure(figsize=(8, 5))
    plt.bar(out["race"].astype(str), out["FPR"] * 100)
    plt.ylabel("False positive rate per 100 non-recidivists")
    plt.title("COMPAS: FPR by race (sample, toy threshold)")
    work_plain_savefig(FIG_DIR / f"{outprefix}_fpr.png")

    # FNR bars
    plt.figure(figsize=(8, 5))
    plt.bar(out["race"].astype(str), out["FNR"] * 100)
    plt.ylabel("False negative rate per 100 recidivists")
    plt.title("COMPAS: FNR by race (sample, toy threshold)")
    work_plain_savefig(FIG_DIR / f"{outprefix}_fnr.png")

    # Base rates (table saved)
    save_df(out[["race", "base_rate", "n"]], f"{outprefix}_base_rates")

def visual_clinicaltrials_overdue(sample_csv: Path, outprefix: str = "ctgov_reporting"):
    """Dot plot of time from primary completion to results first posted (simplified)."""
    df = pd.read_csv(sample_csv, low_memory=False)
    # Try to locate fields in v2 flatten
    date_cols = [c for c in df.columns if "primaryCompletion" in c]
    posted_cols = [c for c in df.columns if "resultsFirstPost" in c]
    if not date_cols or not posted_cols:
        log("ClinicalTrials: date fields not found; skipping visual.")
        return
    to_dt = lambda s: pd.to_datetime(s, errors="coerce")
    primary = to_dt(df[date_cols[0]])
    posted  = to_dt(df[posted_cols[0]])
    delay_days = (posted - primary).dt.days
    df2 = pd.DataFrame({"delay_days": delay_days}).dropna()
    if df2.empty:
        log("ClinicalTrials: no computable delays.")
        return
    samp = df2.sample(n=min(len(df2), 500), random_state=42)
    y = np.zeros(len(samp))
    plt.figure(figsize=(8, 2))
    plt.scatter(samp["delay_days"], y, s=10)
    plt.yticks([])
    plt.xlabel("Days from Primary Completion to Results First Posted")
    plt.title("ClinicalTrials.gov: Reporting lags (sample)")
    work_plain_savefig(FIG_DIR / f"{outprefix}_lags.png")

# ------------------------------
# Runner
# ------------------------------

DATASETS = {
    # Justice & Policing
    "compas": ("COMPAS (ProPublica) Broward County", propublica_compas),
    "hmda": ("HMDA sample (FFIEC/CFPB Browser CSV)", cfpb_hmda_sample),
    "mpv": ("Mapping Police Violence (mirrors)", mapping_police_violence),
    # Health & Environment
    "brfss": ("CDC BRFSS (Socrata JSON sample)", cdc_brfss_prevalence_sample),
    "tri": ("EPA TRI (Envirofacts CSV sample)", epa_tri_basic_sample),
    "hrsa": ("HRSA HPSA (Socrata JSON)", hrsa_hpsa_sample),
    "cms": ("CMS Hospital General Information (metastore -> CSV)", cms_hospital_general_info_download),
    # Transportation
    "nyctlc": ("NYC TLC Trips sample (Socrata)", nyc_tlc_sample),
    # Clinical Research
    "clinicaltrials": ("ClinicalTrials.gov v2 Completed + Interventional", clinicaltrials_results_sample),
}

def run_selected(keys: List[str], max_rows: int):
    paths = {}
    for k in keys:
        desc, fn = DATASETS[k]
        log(f"=== {k}: {desc} ===")
        try:
            if k == "hmda":
                p = fn(max_rows=min(max_rows, 100000))
            elif k == "nyctlc":
                p = fn(limit=min(max_rows, 20000))
            elif k == "tri":
                p = fn(rows=min(max_rows, 2000))
            elif k == "brfss":
                p = fn(limit=min(max_rows, 10000))
            elif k == "clinicaltrials":
                p = fn(n=min(max_rows, 500))
            else:
                p = fn()
            paths[k] = p
        except Exception as e:
            log(f"{k} failed: {e}")
            paths[k] = None
    return paths

def generate_visuals(paths: Dict[str, Optional[Path]]):
    # HMDA visuals
    if paths.get("hmda"):
        try:
            visual_hmda_denial_and_highprice(paths["hmda"])
            log("HMDA visuals done.")
        except Exception as e:
            log(f"HMDA visual error: {e}")

    # COMPAS visuals
    if paths.get("compas"):
        try:
            visual_compas_fpr_fnr(paths["compas"])
            log("COMPAS visuals done.")
        except Exception as e:
            log(f"COMPAS visual error: {e}")

    # ClinicalTrials visuals
    if paths.get("clinicaltrials"):
        try:
            visual_clinicaltrials_overdue(paths["clinicaltrials"])
            log("ClinicalTrials visuals done.")
        except Exception as e:
            log(f"ClinicalTrials visual error: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Harvest datasets (no API keys) + plain visuals")
    parser.add_argument("--list", action="store_true", help="List dataset keys")
    parser.add_argument("--run", action="store_true", help="Run harvest")
    parser.add_argument("--only", type=str, default="", help="Comma-separated dataset keys to run")
    parser.add_argument("--max-rows", type=int, default=20000, help="Row cap per dataset (where applicable)")
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for k, (desc, _) in DATASETS.items():
            print(f"  - {k:14s} : {desc}")
        return

    if not args.run:
        print("Nothing to do. Use --list or --run.")
        return

    keys = list(DATASETS.keys())
    if args.only:
        requested = [s.strip().lower() for s in args.only.split(",") if s.strip()]
        keys = [k for k in keys if k in requested]
        missing = [k for k in requested if k not in DATASETS]
        if missing:
            log(f"Unknown dataset keys ignored: {', '.join(missing)}")

    paths = run_selected(keys, args.max_rows)
    generate_visuals(paths)
    log("Done. See ./data, ./figures, and ./logs/harvest.log")

if __name__ == "__main__":
    main()
