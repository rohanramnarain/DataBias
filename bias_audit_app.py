# bias_audit_app_compat_impute_robust.py
# -----------------------------------------------------------
# Streamlit app to scan a CSV for common data biases.
# Monroe-Nathan pastiche + binning for AGE (5-year) and ZIP (ZIP3).
# Run:  streamlit run bias_audit_app_compat_impute_robust.py
# -----------------------------------------------------------

import io, json, math, os, random, inspect
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import ScaledTranslation

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import chi2_contingency, entropy

# ===========================================================
# MONROE NATHAN PRESENTATION LAYER (auto-applied)
# ===========================================================

class _MNTheme:
    def __init__(self, mode="light", base_font="DejaVu Serif", dpi=150):
        self.mode = mode
        self.base_font = base_font
        self.dpi = dpi
        if mode == "dark":
            self.paper = "#d7c8a6"
            self.paper2 = "#c9b891"
            self.ink = "#131313"
            self.grid = "#5b523e"
        else:
            self.paper = "#f4e7c7"
            self.paper2 = "#eadab2"
            self.ink = "#171615"
            self.grid = "#9a8f74"
        self.tick_size = 10
        self.caption_size = 9
        self.title_size = 16
        self.label_size = 12
        self.spine_alpha = 0.55

def _mn_apply_rc(theme: _MNTheme):
    mpl.rcParams.update({
        "figure.dpi": theme.dpi,
        "savefig.dpi": theme.dpi,
        "savefig.bbox": "tight",
        "figure.facecolor": theme.paper,
        "axes.facecolor": theme.paper,
        "text.color": theme.ink,
        "axes.labelcolor": theme.ink,
        "xtick.color": theme.ink,
        "ytick.color": theme.ink,
        "axes.edgecolor": theme.ink,
        "axes.linewidth": 1.2,
        "grid.color": theme.grid,
        "grid.alpha": 0.35,
        "axes.grid": False,
        "font.family": "serif",
        "font.serif": [theme.base_font, "DejaVu Serif", "Times New Roman", "Georgia"],
        "font.size": 11,
        "axes.titlesize": theme.title_size,
        "axes.labelsize": theme.label_size,
        "axes.titlelocation": "left",
        "axes.prop_cycle": mpl.cycler(color=[theme.ink]),
        "path.sketch": (2, 150, 2),  # hand-drawn wobble
    })

def _mn_noise_texture(ax, theme: _MNTheme, strength=0.08):
    rng = np.random.default_rng(42)
    h = max(200, int(ax.figure.bbox.height))
    w = max(200, int(ax.figure.bbox.width))
    base = rng.normal(0.0, 1.0, (h, w))
    for _ in range(2):
        base = (base[:-1,:-1] + base[1:,:-1] + base[:-1,1:] + base[1:,1:]) / 4.0
    base = (base - base.min()) / (base.max() - base.min() + 1e-9)
    cm = LinearSegmentedColormap.from_list("paperfade", [theme.paper, theme.paper2])
    ax.imshow(base, cmap=cm, interpolation="bilinear",
              extent=(0,1,0,1), transform=ax.transAxes, zorder=-10, alpha=strength)

def _mn_off_kilter_labels(ax, theme: _MNTheme):
    """Slight rotation + point-jitter using ScaledTranslation (safe)."""
    rng = random.Random(7)
    fig = ax.figure
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        if not lbl.get_text():
            continue
        lbl.set_rotation(lbl.get_rotation() + rng.uniform(-2.0, 2.0))
        dx_pt = rng.uniform(-1.5, 1.5) / 72.0
        dy_pt = rng.uniform(-1.5, 1.5) / 72.0
        lbl.set_transform(lbl.get_transform() + ScaledTranslation(dx_pt, dy_pt, fig.dpi_scale_trans))
        lbl.set_fontstyle("italic")

def _mn_stylize_axes(ax, theme: _MNTheme):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_alpha(theme.spine_alpha)

    ax.tick_params(labelsize=theme.tick_size, length=0)
    _mn_off_kilter_labels(ax, theme)

    for p in ax.patches:
        try:
            p.set_edgecolor(theme.ink)
            p.set_linewidth(1.6)
            p.set_facecolor("#232323")
            p.set_alpha(0.95)
            p.set_sketch_params(2, 120, 2)
        except Exception:
            pass

    for ln in ax.lines:
        ln.set_linewidth(2.0)
        ln.set_solid_capstyle("butt")
        ln.set_sketch_params(2, 120, 2)

    # light guide lines (evokes deciles)
    try:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ticks = np.linspace(x0, x1, 6) if ax.xaxis.get_scale() == "linear" else []
        for t in ticks:
            ax.plot([t, t], [y0, y1], color=theme.grid, linewidth=1.1, alpha=0.45, zorder=-5)
    except Exception:
        pass

def _mn_stylize_figure(fig, theme: _MNTheme, caption: Optional[str]):
    for a in fig.axes:
        _mn_noise_texture(a, theme, strength=0.10)
        _mn_stylize_axes(a, theme)
    if caption:
        fig.text(0.01, 0.01, caption, fontsize=theme.caption_size, alpha=0.8)
    fig.tight_layout(pad=1.05)

# activate theme + patch Streamlit
_MONROE = _MNTheme(mode=os.environ.get("MONROE_MODE", "light").strip().lower())
_mn_caption = os.environ.get("MONROE_CAPTION", "MISSING   ☐     OUT OF SCHOOL   ▮").strip()
_mn_apply_rc = _mn_apply_rc(_MONROE)

if hasattr(st, "pyplot"):
    _orig_st_pyplot = st.pyplot
    def _mn_pyplot(fig=None, *args, **kwargs):
        if fig is None:
            fig = plt.gcf()
        _mn_stylize_figure(fig, _MONROE, _mn_caption)
        return _orig_st_pyplot(fig, *args, **kwargs)
    st.pyplot = _mn_pyplot  # type: ignore

# ===========================================================
# ORIGINAL APP LOGIC + BINNING FOR AGE & ZIP
# ===========================================================

_OHE_KW_BASE = {"handle_unknown": "ignore"}
if 'sparse_output' in inspect.signature(OneHotEncoder).parameters:
    _OHE_KW = dict(_OHE_KW_BASE, sparse_output=False)
else:
    _OHE_KW = dict(_OHE_KW_BASE, sparse=False)

# --------------------------
# Utility functions
# --------------------------

def safe_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr, dtype=float); s = arr.sum()
    if s == 0: return np.ones_like(arr) / len(arr) if len(arr) else arr
    return arr / s

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = safe_normalize(p); q = safe_normalize(q)
    m = 0.5 * (p + q)
    return float(0.5 * (entropy(p, m) + entropy(q, m)))

def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.array(expected, dtype=float); actual = np.array(actual, dtype=float)
    if len(expected) == 0 or len(actual) == 0: return np.nan
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    e_hist = np.histogram(expected, bins=qs)[0]
    a_hist = np.histogram(actual, bins=qs)[0]
    e_pct = safe_normalize(e_hist); a_pct = safe_normalize(a_hist)
    return float(np.sum((a_pct - e_pct) * np.log((a_pct + 1e-12) / (e_pct + 1e-12))))

def compute_group_distribution(df: pd.DataFrame, group_col: str) -> pd.Series:
    return df[group_col].astype("category").value_counts(dropna=False).sort_index()

def other_or_unknown_rate(df: pd.DataFrame, col: str, keywords=("other","unknown","unspecified","n/a","na","none")) -> float:
    s = df[col].astype(str).str.strip().str.lower()
    return float(s.isin(keywords).mean())

def missingness_by_group(df: pd.DataFrame, group_col: str) -> Dict[str, float]:
    rates = {}
    for g, sub in df.groupby(group_col, dropna=False):
        cols = [c for c in df.columns if c != group_col]
        rates[str(g)] = float(sub[cols].isna().mean().mean())
    return rates

# --------------------------
# Fairness helpers
# --------------------------

def disparate_impact_ratio(y: pd.Series, group: pd.Series, positive_label=None) -> Tuple[float, Dict[str, float]]:
    if positive_label is None:
        vals = y.dropna().unique().tolist()
        if len(vals) == 2: positive_label = vals[1]
        elif len(vals) > 0: positive_label = vals[0]
        else: return (np.nan, {})
    rates = {}
    for g, sub in y.groupby(group):
        if len(sub) == 0: continue
        rates[str(g)] = float((sub == positive_label).mean())
    if not rates: return (np.nan, {})
    mn, mx = min(rates.values()), max(rates.values())
    if mx == 0: return (np.nan, rates)
    return (mn / mx, rates)

def collapse_rare_labels(y: pd.Series, min_count: int = 5, rare_label: str = "__RARE__") -> pd.Series:
    vc = y.value_counts(); rare = vc[vc < min_count].index
    return y if len(rare) == 0 else y.where(~y.isin(rare), rare_label)

def predictability_of_sensitive(df: pd.DataFrame, sensitive_col: str, feature_cols: List[str]) -> Tuple[float, List[Tuple[str, float]]]:
    data = df.copy(); y = data[sensitive_col].astype(str); X = data[feature_cols]
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    num_cols = [c for c in X.columns if c not in cat_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler(with_mean=False))]), num_cols),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(**_OHE_KW))]), cat_cols),
        ], remainder="drop"
    )
    clf = LogisticRegression(max_iter=1000, multi_class="auto")
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    mask = ~y.isna(); X2, y2 = X[mask], y[mask]
    if len(X2) < 50: return (np.nan, [])

    y2c = collapse_rare_labels(y2, min_count=5, rare_label="__RARE__")
    vc = y2c.value_counts()
    if len(vc) < 2: return (np.nan, [])
    if vc.min() < 2:
        keep = y2c.isin(vc[vc >= 2].index)
        X2, y2c = X2[keep], y2c[keep]
        if len(y2c.value_counts()) < 2: return (np.nan, [])
    strat_arg = y2c if y2c.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(X2, y2c, test_size=0.3, random_state=42, stratify=strat_arg)
    pipe.fit(X_train, y_train)
    try:
        proba = pipe.predict_proba(X_test)
        classes = pipe.named_steps["clf"].classes_
        y_bin = pd.get_dummies(y_test).reindex(columns=classes, fill_value=0).values
        auc = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
    except Exception:
        pred = pipe.predict(X_test); auc = accuracy_score(y_test, pred)

    mi_scores = []
    for col in feature_cols:
        try:
            if X[col].dtype.kind in "bifc":
                xx = X[[col]].copy()
                xx[col] = xx[col].fillna(xx[col].median())
                mi = mutual_info_classif(xx, y2c, discrete_features=[False], random_state=42)
                mi_scores.append((col, float(mi[0])))
            else:
                enc = OneHotEncoder(**_OHE_KW)
                xx = X[[col]].copy()
                xx[col] = xx[col].fillna("missing")
                xx = enc.fit_transform(xx)
                mi = mutual_info_classif(xx, y2c, discrete_features=True, random_state=42)
                mi_scores.append((col, float(mi.sum())))
        except Exception:
            continue
    mi_scores.sort(key=lambda t: t[1], reverse=True)
    return (float(auc), mi_scores[:10])

# --------------------------
# Scoring rubric (0-100 risk)
# --------------------------

def score_representation(jsd: float) -> float:
    if np.isnan(jsd): return np.nan
    return float(min(100.0, (jsd / 0.5) * 100.0))

def score_missingness(max_gap: float) -> float:
    if np.isnan(max_gap): return np.nan
    return float(np.interp(max_gap, [0.01, 0.4], [0, 100]))

def score_disparate_impact(diratio: float) -> float:
    if np.isnan(diratio): return np.nan
    diratio = max(min(diratio, 1.0), 0.0); return float((1.0 - diratio) * 100)

def score_proxy_leakage(auc: float) -> float:
    if np.isnan(auc): return np.nan
    if auc <= 0.5: return 0.0
    return float(np.interp(auc, [0.5, 0.8, 0.9, 1.0], [0, 80, 95, 100]))

def score_other_bucket(rate: float) -> float:
    if np.isnan(rate): return np.nan
    return float(np.interp(rate, [0.01, 0.3], [0, 100]))

def score_temporal_drift(psi: float) -> float:
    if np.isnan(psi): return np.nan
    return float(np.interp(psi, [0.0, 0.1, 0.25, 0.5], [0, 30, 70, 100]))

# --------------------------
# NEW: Heuristic detectors for additional bias types
# --------------------------

def _find_col(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for k in keywords:
        for lc, orig in low.items():
            if k in lc: return orig
    return None

def _binaryify(s: pd.Series) -> Optional[pd.Series]:
    try:
        x = s.astype(str).str.lower().str.strip()
        if set(x.unique()) <= {"0","1","true","false","yes","no","y","n","nan"}:
            m = x.map({"1":1,"0":0,"true":1,"false":0,"yes":1,"no":0,"y":1,"n":0})
            return m.astype(float)
    except Exception:
        pass
    return None

def detect_construct_bias(df: pd.DataFrame, sensitive_cols: List[str], target_col: Optional[str]) -> Optional[Dict]:
    if not target_col or target_col not in df.columns: return None
    y = df[target_col]
    if y.dtype.kind in "bifc":
        std = float(pd.to_numeric(y, errors="coerce").std(skipna=True) or 0.0)
        risk = float(np.interp(std, [0.0, 1e-6, 1.0], [100, 95, 0])) if not np.isnan(std) else np.nan
        detail = f"Outcome std≈{std:.3g} (near-constant -> higher risk)"
    else:
        share = float(y.value_counts(normalize=True, dropna=False).max()) if len(y) else np.nan
        risk = float(np.interp(share, [0.5, 0.95, 1.0], [10, 90, 100])) if not np.isnan(share) else np.nan
        detail = f"Dominant label share≈{share:.3f}"
    return {"bias": "Construct bias", "score": round(risk,1) if not np.isnan(risk) else np.nan, "detail": detail}

def detect_nonresponse_bias(df: pd.DataFrame, sensitive_cols: List[str]) -> Optional[Dict]:
    cand = _find_col(df, ["respond","response","completed","finished"])
    if not cand: return None
    b = _binaryify(df[cand])
    if b is None: return None
    gaps = []
    for s in sensitive_cols:
        try:
            diratio, _ = disparate_impact_ratio(b, df[s], positive_label=1)
            if not np.isnan(diratio):
                gaps.append(1 - diratio)
        except Exception:
            continue
    if not gaps: return None
    risk = float(np.interp(max(gaps), [0.0, 0.2, 0.5, 1.0], [0, 40, 80, 100]))
    return {"bias": "Nonresponse bias", "score": round(risk,1), "detail": f"Max response-rate disparity≈{max(gaps):.3f}", "col": cand, "series": b}

def detect_enumerator_bias(df: pd.DataFrame, sensitive_cols: List[str], target_col: Optional[str]) -> Optional[Dict]:
    enum_col = _find_col(df, ["enumerator","interviewer","collector","agent","staff"])
    if not enum_col or not target_col or target_col not in df.columns: return None
    try:
        tbl = pd.crosstab(df[enum_col], df[target_col])
        if tbl.shape[0] >= 2 and tbl.shape[1] >= 2:
            chi2, p, _, _ = chi2_contingency(tbl.fillna(0))
            risk = float(np.interp(-np.log10(max(p,1e-12)), [0, 3, 6], [10, 60, 100]))
            return {"bias": "Enumerator bias", "score": round(risk,1), "detail": f"chi2 p≈{p:.2e} ({enum_col} × {target_col})", "table": tbl, "enum_col": enum_col, "target_col": target_col}
    except Exception:
        pass
    return None

def detect_consent_bias(df: pd.DataFrame, sensitive_cols: List[str], target_col: Optional[str]) -> Optional[Dict]:
    ccol = _find_col(df, ["consent","assent","opt_in","optout","opt-out","opt out"])
    if not ccol: return None
    b = _binaryify(df[ccol])
    if b is None: return None
    gaps = []
    per_group = {}
    for s in sensitive_cols:
        try:
            diratio, rates = disparate_impact_ratio(b, df[s], positive_label=1)
            if rates:
                per_group[s] = rates
            if not np.isnan(diratio):
                gaps.append(1 - diratio)
        except Exception:
            continue
    if not gaps: return None
    risk = float(np.interp(max(gaps), [0.0, 0.15, 0.4, 0.8], [0, 40, 80, 100]))
    return {"bias": "Consent bias", "score": round(risk,1), "detail": f"Max consent disparity≈{max(gaps):.3f} by group", "col": ccol, "per_group": per_group}

def detect_labeling_bias(df: pd.DataFrame, target_col: Optional[str]) -> Optional[Dict]:
    if not target_col or target_col not in df.columns: return None
    lab = _find_col(df, ["labeler","annotator","rater","coder","source"])
    if not lab: return None
    try:
        tbl = pd.crosstab(df[lab], df[target_col])
        if tbl.shape[0] >= 2 and tbl.shape[1] >= 2:
            chi2, p, _, _ = chi2_contingency(tbl.fillna(0))
            risk = float(np.interp(-np.log10(max(p,1e-12)), [0, 3, 6], [10, 60, 100]))
            return {"bias": "Labeling bias", "score": round(risk,1), "detail": f"chi2 p≈{p:.2e} ({lab} × {target_col})", "table": tbl, "labeler_col": lab, "target_col": target_col}
    except Exception:
        pass
    return None

def detect_archival_erasure_bias(df: pd.DataFrame, time_col: Optional[str], sensitive_cols: List[str]) -> List[Dict]:
    out = []
    if not time_col or time_col not in df.columns: return out
    t = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    dfx = df.copy(); dfx["_t"] = t
    dfx = dfx.dropna(subset=["_t"])
    if len(dfx) < 50: return out
    q1, q3 = dfx["_t"].quantile([0.25, 0.75]).values
    early = dfx[dfx["_t"] <= q1]
    late  = dfx[dfx["_t"] >= q3]
    for s in sensitive_cols:
        e_counts = early[s].astype(str).value_counts(dropna=False).sort_index()
        l_counts = late[s].astype(str).value_counts(dropna=False).sort_index()
        e_groups = set(e_counts.index.tolist())
        l_groups = set(l_counts.index.tolist())
        vanished = e_groups - l_groups
        risk = 0.0 if len(e_groups)==0 else (len(vanished)/max(1,len(e_groups))) * 100.0
        if len(e_groups | l_groups) > 0:
            out.append({"bias": "Archival/Erasure bias", "sensitive_attr": s,
                        "score_0to100": round(risk,1),
                        "detail": f"Groups present early but missing late: {sorted(list(vanished))[:6]}...",
                        "early_counts": e_counts, "late_counts": l_counts})
    return out

def detect_aggregation_bias(df: pd.DataFrame, target_col: Optional[str], sensitive_cols: List[str]) -> Optional[Dict]:
    if not target_col or target_col not in df.columns: return None
    y = df[target_col]
    zero_var_groups = 0; big_groups = 0
    entropies = []
    labels = []
    for s in sensitive_cols:
        for g, sub in df.groupby(s, dropna=False):
            if len(sub) >= 20:
                big_groups += 1
                ysub = y.loc[sub.index]
                ent = float(entropy(ysub.value_counts(normalize=True, dropna=False).values + 1e-12))
                entropies.append(ent); labels.append(f"{s}={g}")
                if ysub.nunique(dropna=True) <= 1:
                    zero_var_groups += 1
    if big_groups == 0: return None
    prop = zero_var_groups / big_groups
    risk = float(np.interp(prop, [0.0, 0.2, 0.5, 1.0], [0, 40, 80, 100]))
    return {"bias": "Aggregation bias", "score": round(risk,1),
            "detail": f"{zero_var_groups}/{big_groups} sizeable groups with zero outcome variation",
            "entropy_labels": labels, "entropy_values": entropies}

def detect_linkage_bias(df: pd.DataFrame) -> Optional[Dict]:
    key = _find_col(df, ["id","subject_id","patient_id","record_id","uuid","guid"])
    if not key: return None
    try:
        dup_mask = df[key].astype(str).duplicated(keep=False)
        dup_rate = float(dup_mask.mean())
        dup_groups = df[key].astype(str).value_counts()
        risk = float(np.interp(dup_rate, [0.0, 0.05, 0.2, 0.5], [0, 25, 70, 100]))
        return {"bias": "Linkage bias", "score": round(risk,1), "detail": f"Duplicate-key rate≈{dup_rate:.3f} in {key}",
                "key": key, "dup_counts": dup_groups}
    except Exception:
        return None

def detect_evaluation_bias(df: pd.DataFrame, target_col: Optional[str], score_col: Optional[str], pred_col: Optional[str], sensitive_cols: List[str]) -> Optional[Dict]:
    if not target_col or target_col not in df.columns: return None
    if score_col and score_col in df.columns:
        y = df[target_col]
        s = pd.to_numeric(df[score_col], errors="coerce")
        per_group_auc = {}
        for c in sensitive_cols:
            vals = []
            labs = []
            try:
                mask = (~y.isna()) & (~s.isna()) & (~df[c].isna())
                yc = y[mask].astype(str)
                sc = s[mask]
                for g, sub in yc.groupby(df[c][mask]):
                    if len(sub) >= 30 and sub.nunique() == 2:
                        pos = sub.unique().tolist()[1]
                        ybin = (yc.loc[sub.index] == pos).astype(int)
                        auc = roc_auc_score(ybin, sc.loc[sub.index])
                        vals.append(auc); labs.append(str(g))
                if vals:
                    per_group_auc[c] = (labs, vals)
            except Exception:
                continue
        if per_group_auc:
            spreads = [max(v)-min(v) for _, v in per_group_auc.values() if len(v) >= 2]
            spread = max(spreads) if spreads else 0.0
            risk = float(np.interp(spread, [0.0, 0.05, 0.15, 0.3], [0, 30, 70, 100]))
            return {"bias": "Evaluation bias", "score": round(risk,1), "detail": f"AUC spread across groups≈{spread:.3f}",
                    "per_group_auc": per_group_auc}
    elif pred_col and pred_col in df.columns:
        y = df[target_col].astype(str); yhat = df[pred_col].astype(str)
        per_group_acc = {}
        for c in sensitive_cols:
            vals = []
            labs = []
            try:
                for g, sub in y.groupby(df[c]):
                    idx = sub.index
                    if len(idx) >= 30:
                        acc = float((y.loc[idx] == yhat.loc[idx]).mean())
                        vals.append(acc); labs.append(str(g))
                if vals:
                    per_group_acc[c] = (labs, vals)
            except Exception:
                continue
        if per_group_acc:
            spreads = [max(v)-min(v) for _, v in per_group_acc.values() if len(v) >= 2]
            spread = max(spreads) if spreads else 0.0
            risk = float(np.interp(spread, [0.0, 0.05, 0.15, 0.3], [0, 30, 70, 100]))
            return {"bias": "Evaluation bias", "score": round(risk,1), "detail": f"Accuracy spread across groups≈{spread:.3f}",
                    "per_group_acc": per_group_acc}
    return None

def detect_visualization_bias(df: pd.DataFrame, sensitive_cols: List[str]) -> Optional[Dict]:
    high_card = [s for s in sensitive_cols if df[s].nunique(dropna=False) >= 25]
    if not high_card: return None
    risk = float(np.interp(len(high_card), [1, 3, 5], [40, 70, 90]))
    card_counts = {s: int(df[s].nunique(dropna=False)) for s in high_card}
    return {"bias": "Visualization/Presentation bias", "score": round(risk,1),
            "detail": f"High-cardinality sensitive fields: {', '.join(high_card[:5])}",
            "card_counts": card_counts}

def detect_interpretive_bias(df: pd.DataFrame, sensitive_cols: List[str]) -> Optional[Dict]:
    rates = []
    per = {}
    for s in sensitive_cols:
        try:
            r = other_or_unknown_rate(df, s)
            rates.append(r); per[s] = r
        except Exception:
            continue
    if not rates: return None
    r = float(np.nanmax(rates))
    risk = score_other_bucket(r)
    return {"bias": "Interpretive/Label choice bias", "score": round(risk,1),
            "detail": f"Max 'other/unknown' rate≈{r:.3f}", "per_attr": per}

def detect_context_shift_bias(df: pd.DataFrame, sensitive_cols: List[str]) -> Optional[Dict]:
    src = _find_col(df, ["source","environment","dataset","phase","split","domain"])
    if not src: return None
    try:
        mixes = []
        heatmaps = {}
        for s in sensitive_cols:
            tbl = pd.crosstab(df[src].astype(str), df[s].astype(str), normalize="index")
            if tbl.shape[0] >= 2 and tbl.shape[1] >= 2:
                rows = tbl.values
                max_js = 0.0
                for i in range(len(rows)):
                    for j in range(i+1, len(rows)):
                        max_js = max(max_js, js_divergence(rows[i], rows[j]))
                mixes.append(max_js)
                heatmaps[s] = tbl
        if not mixes: return None
        risk = float(np.interp(max(mixes), [0.0, 0.1, 0.25, 0.5], [0, 30, 70, 100]))
        return {"bias": "Context-shift bias", "score": round(risk,1),
                "detail": f"Max JS divergence across sources≈{max(mixes):.3f}",
                "src_col": src, "heatmaps": heatmaps}
    except Exception:
        return None

# --------------------------
# NEW: Visualization helpers for new detectors
# --------------------------

def show_hist_or_bar_for_outcome(df: pd.DataFrame, target_col: str):
    y = df[target_col]
    if y.dtype.kind in "bifc":
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(pd.to_numeric(y, errors="coerce").dropna(), bins=30)
        ax.set_title(f"Outcome distribution: {target_col}")
        ax.set_xlabel(target_col); ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        vc = y.astype(str).value_counts().sort_values(ascending=False).head(25)
        fig, ax = plt.subplots(figsize=(8, 0.3*max(6, len(vc))))
        ax.barh(range(len(vc)), vc.values[::-1])
        ax.set_yticks(range(len(vc))); ax.set_yticklabels(vc.index[::-1])
        ax.set_title(f"Outcome label counts (top 25): {target_col}")
        ax.set_xlabel("Count")
        st.pyplot(fig)

def show_response_by_group(series_bin: pd.Series, df: pd.DataFrame, sensitive_cols: List[str], title_prefix="Response"):
    for s in sensitive_cols:
        try:
            rates = df.groupby(s)[series_bin.name].apply(lambda x: float(np.nanmean(x)))
            fig, ax = plt.subplots(figsize=(8, 0.3*max(6, len(rates))))
            ax.barh(range(len(rates)), rates.values[::-1])
            ax.set_yticks(range(len(rates))); ax.set_yticklabels(rates.index[::-1])
            ax.set_xlabel("Rate"); ax.set_title(f"{title_prefix} rate by {s}")
            st.pyplot(fig)
        except Exception:
            continue

def show_heatmap_from_crosstab(tbl: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(8, 0.35*max(6, tbl.shape[0])))
    im = ax.imshow(tbl.values, aspect="auto")
    ax.set_yticks(range(tbl.shape[0])); ax.set_yticklabels(tbl.index.astype(str))
    ax.set_xticks(range(tbl.shape[1])); ax.set_xticklabels(tbl.columns.astype(str), rotation=45, ha="right")
    ax.set_title(title)
    for (i,j), v in np.ndenumerate(tbl.values):
        try:
            ax.text(j, i, int(v), ha="center", va="center", fontsize=9)
        except Exception:
            pass
    st.pyplot(fig)

def show_early_late_bars(e_counts: pd.Series, l_counts: pd.Series, s_name: str):
    cats = sorted(set(e_counts.index) | set(l_counts.index))
    e_vals = [int(e_counts.get(c, 0)) for c in cats]
    l_vals = [int(l_counts.get(c, 0)) for c in cats]
    y = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(9, 0.3*max(6, len(cats))))
    ax.barh(y+0.2, e_vals, height=0.4, label="Early")
    ax.barh(y-0.2, l_vals, height=0.4, label="Late")
    ax.set_yticks(y); ax.set_yticklabels(cats)
    ax.set_title(f"Early vs Late presence by {s_name}")
    ax.set_xlabel("Count"); ax.legend()
    st.pyplot(fig)

def show_entropy_by_group(labels: List[str], values: List[float]):
    if not labels or not values: return
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    fig, ax = plt.subplots(figsize=(9, 0.3*max(6, len(values))))
    ax.barh(range(len(values)), values)
    ax.set_yticks(range(len(values))); ax.set_yticklabels(labels)
    ax.set_xlabel("Outcome entropy"); ax.set_title("Within-group outcome entropy (lower can indicate over-aggregation)")
    st.pyplot(fig)

def show_dup_hist(dup_counts: pd.Series, key_name: str):
    if dup_counts is None or dup_counts.empty: return
    fig, ax = plt.subplots(figsize=(8,3))
    ax.hist(dup_counts.values, bins=30)
    ax.set_title(f"Records per {key_name} (higher mass >1 indicates linkage risk)")
    ax.set_xlabel("Count per key"); ax.set_ylabel("Frequency")
    st.pyplot(fig)

def show_performance_bars(metric_dict: Dict[str, Tuple[List[str], List[float]]], metric_name: str):
    for sens, (labs, vals) in metric_dict.items():
        fig, ax = plt.subplots(figsize=(8, 0.3*max(6, len(vals))))
        order = np.argsort(vals)
        vals = [vals[i] for i in order]
        labs = [labs[i] for i in order]
        ax.barh(range(len(vals)), vals)
        ax.set_yticks(range(len(vals))); ax.set_yticklabels(labs)
        ax.set_xlabel(metric_name); ax.set_title(f"{metric_name} by {sens}")
        st.pyplot(fig)

def show_cardinality_bars(card_counts: Dict[str, int]):
    items = sorted(card_counts.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k,_ in items]; vals = [v for _,v in items]
    fig, ax = plt.subplots(figsize=(8, 0.3*max(6, len(vals))))
    ax.barh(range(len(vals)), vals)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels)
    ax.set_xlabel("Unique categories"); ax.set_title("High-cardinality sensitive attributes")
    st.pyplot(fig)

def show_other_unknown_bars(per_attr: Dict[str, float]):
    items = sorted(per_attr.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k,_ in items]; vals = [v for _,v in items]
    fig, ax = plt.subplots(figsize=(8, 0.3*max(6, len(vals))))
    ax.barh(range(len(vals)), vals)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels)
    ax.set_xlabel("'other/unknown' share"); ax.set_title("Interpretive label choices by sensitive attribute")
    st.pyplot(fig)

def show_context_heatmaps(heatmaps: Dict[str, pd.DataFrame], src_col: str):
    for s, tbl in heatmaps.items():
        fig, ax = plt.subplots(figsize=(8, 0.35*max(6, tbl.shape[0])))
        im = ax.imshow(tbl.values, aspect="auto")
        ax.set_yticks(range(tbl.shape[0])); ax.set_yticklabels(tbl.index.astype(str))
        ax.set_xticks(range(tbl.shape[1])); ax.set_xticklabels(tbl.columns.astype(str), rotation=45, ha="right")
        ax.set_title(f"{src_col} × {s}: group share per source (row-normalized)")
        st.pyplot(fig)

# --------------------------
# Streamlit UI
# --------------------------

st.set_page_config(page_title="Bias Audit — Monroe Nathan Pastiche", layout="wide")
st.title("CSV Bias Audit (Compat + Impute + Robust)")

st.write(
    """
Upload a CSV and select:
- **Sensitive attribute(s)** (e.g., age, gender, ZIP),
- Optional **outcome/label** column,
- Optional **prediction/score** column(s),
- Optional **timestamp** (for drift),
- Optional **baseline CSV** (columns: group, share) for representation checks.
"""
)

uploaded = st.file_uploader("Upload CSV", type=["csv", "tsv", "txt"])
baseline_file = st.file_uploader("Optional: baseline group distribution CSV (columns: group, share)", type=["csv", "tsv", "txt"])

if uploaded is not None:
    st.markdown("### Reader options")
    c1, c2, c3, c4, c5 = st.columns(5)
    delim = c1.selectbox("Delimiter", ("auto", ",", ";", "\\t", "|"), index=0)
    bad   = c2.selectbox("On bad lines", ("skip", "warn", "error"), index=0)
    encoding = c3.text_input("Encoding", "utf-8")
    quotechar = c4.text_input("Quote char", '"')
    has_header = c5.checkbox("Has header row", value=True)
    skiprows = st.number_input("Skip rows at top", min_value=0, max_value=1000, value=0, step=1)

    read_kwargs = {
        "encoding": encoding or "utf-8",
        "on_bad_lines": bad,
        "quotechar": (quotechar or '"')[0],
        "skiprows": int(skiprows) if skiprows else 0,
        "header": 0 if has_header else None
    }
    if delim == "auto":
        read_kwargs.update({"sep": None, "engine": "python"})
    elif delim == "\\t":
        read_kwargs.update({"sep": "\t"})
    else:
        read_kwargs.update({"sep": delim})

    try:
        df = pd.read_csv(uploaded, **read_kwargs)
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")
        st.stop()

    st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]:,} cols")
    st.dataframe(df.head(20), use_container_width=True)

    suggested = [c for c in df.columns if any(k in c.lower() for k in ["race","gender","age","zip","zipcode","postal","language","ethnic"])]
    sensitive_cols = st.multiselect("Sensitive attribute columns", options=list(df.columns), default=suggested)

    target_col = st.selectbox("Outcome/label column (optional)", options=[None] + list(df.columns))
    pred_col = st.selectbox("Prediction column (optional; 0/1 or class)", options=[None] + list(df.columns))
    score_col = st.selectbox("Predicted score/probability column (optional; 0-1)", options=[None] + list(df.columns))
    time_col = st.selectbox("Timestamp column (optional)", options=[None] + list(df.columns))

    st.markdown("---")
    report_rows = []

    baselines: Dict[str, Dict[str, float]] = {}
    if baseline_file is not None:
        try:
            bdf = pd.read_csv(baseline_file, sep=None, engine="python")
            if {"group", "share"}.issubset(set(bdf.columns)):
                baselines["__default__"] = dict(zip(bdf["group"].astype(str), bdf["share"].astype(float)))
                st.info("Loaded baseline distribution for representation checks.")
        except Exception as e:
            st.warning(f"Could not parse baseline file: {e}")

    # controls specific to binning
    st.markdown("### Binning options")
    cA, cB = st.columns(2)
    bin_age_on = cA.checkbox("Bin AGE into 5-year groups (if numeric)", value=True)
    bin_zip_on = cB.checkbox("Group ZIP/Postal into ZIP3 (first 3 digits)", value=True)

    for s_col in sensitive_cols:
        st.subheader(f"Sensitive attribute: {s_col}")

        # ---------------- Representation (with binning for age & zip) ----------------
        series = None
        lower = s_col.lower().strip()

        # AGE binning (5-year)
        if bin_age_on and ("age" in lower) and pd.api.types.is_numeric_dtype(df[s_col]):
            vals = pd.to_numeric(df[s_col], errors="coerce")
            vmin = int(np.nanmin(vals)) if np.isfinite(np.nanmin(vals)) else 0
            vmax = int(np.nanmax(vals)) if np.isfinite(np.nanmax(vals)) else 100
            start = (vmin // 5) * 5
            stop = ((vmax // 5) + 1) * 5
            bins = list(range(start, stop + 5, 5))
            cats = pd.cut(vals, bins=bins, right=False)
            series = cats.value_counts(dropna=False).sort_index()
            labels = []
            for iv in series.index:
                if pd.isna(iv):
                    labels.append("NaN")
                else:
                    labels.append(f"{int(iv.left)}–{int(iv.right-1)}")
            series.index = labels

        # ZIP binning (ZIP3)
        if series is None and bin_zip_on and any(k in lower for k in ["zip", "zipcode", "postal"]):
            z = df[s_col].astype(str).str.extract(r"(\d{3})", expand=False)
            series = z.value_counts(dropna=False).sort_index()
            series.index = series.index.fillna("NaN")

        # default: no binning
        if series is None:
            series = compute_group_distribution(df, s_col)

        groups = [str(i) for i in series.index.tolist()]
        counts = series.values.astype(float)
        sample_p = safe_normalize(counts)

        fig1, ax1 = plt.subplots(figsize=(9, 0.34 * max(8, len(groups))))
        ypos = np.arange(len(groups))
        ax1.barh(ypos, sample_p)
        ax1.set_yticks(ypos)
        ax1.set_yticklabels(groups)
        ax1.invert_yaxis()
        ax1.set_xlabel("Share in dataset")
        ax1.set_title(f"Representation of {s_col}")
        st.pyplot(fig1)

        # Score vs baseline/uniform
        if baselines.get("__default__"):
            base_dict = baselines["__default__"]
            base_p = np.array([base_dict.get(g, 0.0) for g in groups], dtype=float)
            base_p = safe_normalize(base_p)
            jsd = js_divergence(sample_p, base_p)
            rep_score = score_representation(jsd)
            rep_note = "vs provided baseline"
        else:
            uniform = np.ones_like(sample_p) / len(sample_p) if len(sample_p) else sample_p
            jsd = js_divergence(sample_p, uniform) if len(sample_p) else np.nan
            rep_score = score_representation(jsd) if not np.isnan(jsd) else np.nan
            rep_note = "vs uniform (baseline not provided)"

        report_rows.append({
            "bias": "Coverage / Representation",
            "sensitive_attr": s_col,
            "score_0to100": round(rep_score, 1) if not np.isnan(rep_score) else np.nan,
            "detail": f"JSD={jsd:.3f} {rep_note}" if not np.isnan(jsd) else "insufficient data"
        })

        # ---------------- Missingness ----------------
        miss_rates = missingness_by_group(df, s_col)
        max_gap = max(miss_rates.values()) if miss_rates else np.nan
        miss_score = score_missingness(max_gap)

        fig2, ax2 = plt.subplots(figsize=(9, 0.34 * max(6, len(miss_rates))))
        y2 = np.arange(len(miss_rates))
        ax2.barh(y2, list(miss_rates.values()))
        ax2.set_yticks(y2)
        ax2.set_yticklabels(list(miss_rates.keys()))
        ax2.invert_yaxis()
        ax2.set_xlabel("Avg missingness across columns")
        ax2.set_title(f"Missingness by {s_col}")
        st.pyplot(fig2)

        report_rows.append({
            "bias": "Missingness / Data quality by group",
            "sensitive_attr": s_col,
            "score_0to100": round(miss_score, 1) if not np.isnan(miss_score) else np.nan,
            "detail": f"Max group missingness={max_gap:.3f}" if not np.isnan(max_gap) else "n/a"
        })

        # ---------------- Outcome disparity ----------------
        if target_col:
            diratio, rates = disparate_impact_ratio(df[target_col], df[s_col])
            di_score = score_disparate_impact(diratio)

            if rates:
                fig3, ax3 = plt.subplots(figsize=(9, 0.34 * max(6, len(rates))))
                y3 = np.arange(len(rates))
                ax3.barh(y3, list(rates.values()))
                ax3.set_yticks(y3)
                ax3.set_yticklabels(list(rates.keys()))
                ax3.invert_yaxis()
                ax3.set_xlabel("Positive outcome rate")
                ax3.set_title(f"Outcome rates by {s_col}")
                st.pyplot(fig3)

            report_rows.append({
                "bias": "Outcome disparity (four-fifths rule)",
                "sensitive_attr": s_col,
                "score_0to100": round(di_score, 1) if not np.isnan(di_score) else np.nan,
                "detail": f"Disparate impact ratio={diratio:.3f} (min/max positive rates)" if not np.isnan(diratio) else "n/a"
            })

        # ---------------- Proxy leakage (robust) ----------------
        feature_cols = [c for c in df.columns if c != s_col]
        drop_like = ["id", "uuid", "guid", "ssn", "email", "name"]
        feature_cols = [c for c in feature_cols if not any(x in c.lower() for x in drop_like)]
        if len(feature_cols) >= 2:
            auc, top_feats = predictability_of_sensitive(df, s_col, feature_cols)
            proxy_score = score_proxy_leakage(auc)
            top_str = ", ".join([f"{k}:{v:.3f}" for k,v in top_feats[:5]]) if top_feats else ""
            report_rows.append({
                "bias": "Proxy bias (sensitive predictability)",
                "sensitive_attr": s_col,
                "score_0to100": round(proxy_score, 1) if not np.isnan(proxy_score) else np.nan,
                "detail": f"AUC≈{auc:.3f}; top MI features -> {top_str}" if not np.isnan(auc) else "Skipped: too-few samples/rare levels"
            })

        # ---------------- Drift ----------------
        if time_col:
            try:
                tseries = pd.to_datetime(df[time_col], errors="coerce", utc=True)
            except Exception:
                tseries = None
            if tseries is not None:
                df2 = df.copy()
                df2["_t"] = tseries
                df2 = df2.dropna(subset=["_t"])
                if len(df2) >= 50:
                    qs = df2["_t"].quantile([0.25, 0.75]).values
                    early = df2[df2["_t"] <= qs[0]]
                    late = df2[df2["_t"] >= qs[1]]
                    codes = df2[s_col].astype("category").cat.codes
                    e_vals = codes.loc[early.index].values
                    l_vals = codes.loc[late.index].values
                    psi = population_stability_index(e_vals, l_vals, bins=max(5, min(20, df2[s_col].nunique())))
                    drift_score = score_temporal_drift(psi)
                    report_rows.append({
                        "bias": "Historical drift",
                        "sensitive_attr": s_col,
                        "score_0to100": round(drift_score, 1) if not np.isnan(drift_score) else np.nan,
                        "detail": f"PSI (early vs late) = {psi:.3f}" if not np.isnan(psi) else "n/a"
                    })

    # ---------------- NEW: Dataset-level detectors (with visualizations) ----------------
    # Construct bias
    c_result = detect_construct_bias(df, sensitive_cols, target_col)
    if c_result:
        report_rows.append({
            "bias": c_result["bias"], "sensitive_attr": "__dataset__",
            "score_0to100": c_result["score"], "detail": c_result["detail"]
        })
        if target_col:
            st.markdown("#### Construct bias — Outcome distribution")
            show_hist_or_bar_for_outcome(df, target_col)

    # Nonresponse bias
    nr_result = detect_nonresponse_bias(df, sensitive_cols)
    if nr_result:
        report_rows.append({
            "bias": nr_result["bias"], "sensitive_attr": "__dataset__",
            "score_0to100": nr_result["score"], "detail": nr_result["detail"]
        })
        st.markdown("#### Nonresponse bias — Response rate by group")
        sbin = nr_result.get("series")
        if sbin is not None:
            sbin.name = "__response__"
            show_response_by_group(sbin, df, sensitive_cols, title_prefix="Response")

    # Enumerator bias
    en_result = detect_enumerator_bias(df, sensitive_cols, target_col)
    if en_result:
        report_rows.append({
            "bias": en_result["bias"], "sensitive_attr": "__dataset__",
            "score_0to100": en_result["score"], "detail": en_result["detail"]
        })
        st.markdown("#### Enumerator bias — Outcome by enumerator (counts)")
        show_heatmap_from_crosstab(en_result["table"], f"{en_result['enum_col']} × {en_result['target_col']}")

    # Consent bias
    cs_result = detect_consent_bias(df, sensitive_cols, target_col)
    if cs_result:
        report_rows.append({
            "bias": cs_result["bias"], "sensitive_attr": "__dataset__",
            "score_0to100": cs_result["score"], "detail": cs_result["detail"]
        })
        st.markdown("#### Consent bias — Consent rate by sensitive group")
        per = cs_result.get("per_group", {})
        for s, rates in per.items():
            fig, ax = plt.subplots(figsize=(8, 0.3*max(6, len(rates))))
            labs = list(rates.keys()); vals = list(rates.values())
            order = np.argsort(vals)
            vals = [vals[i] for i in order]; labs = [labs[i] for i in order]
            ax.barh(range(len(vals)), vals)
            ax.set_yticks(range(len(vals))); ax.set_yticklabels(labs)
            ax.set_xlabel("Consent rate"); ax.set_title(f"Consent rate by {s}")
            st.pyplot(fig)

    # Labeling bias
    lb_result = detect_labeling_bias(df, target_col)
    if lb_result:
        report_rows.append({
            "bias": lb_result["bias"], "sensitive_attr": "__dataset__",
            "score_0to100": lb_result["score"], "detail": lb_result["detail"]
        })
        st.markdown("#### Labeling bias — Labels by labeler (counts)")
        show_heatmap_from_crosstab(lb_result["table"], f"{lb_result['labeler_col']} × {lb_result['target_col']}")

    # Archival / Erasure bias (per sensitive attr, if time present)
    if time_col:
        ae_rows = detect_archival_erasure_bias(df, time_col, sensitive_cols)
        for row in ae_rows:
            report_rows.append(row)
            st.markdown(f"#### Archival/Erasure bias — Early vs Late for **{row['sensitive_attr']}**")
            show_early_late_bars(row["early_counts"], row["late_counts"], row["sensitive_attr"])

    # Aggregation bias
    ag_result = detect_aggregation_bias(df, target_col, sensitive_cols)
    if ag_result:
        report_rows.append({
            "bias": ag_result["bias"], "sensitive_attr": "__dataset__",
            "score_0to100": ag_result["score"], "detail": ag_result["detail"]
        })
        st.markdown("#### Aggregation bias — Within-group outcome entropy")
        show_entropy_by_group(ag_result.get("entropy_labels", []), ag_result.get("entropy_values", []))

    # Linkage bias
    lk_result = detect_linkage_bias(df)
    if lk_result:
        report_rows.append({
            "bias": lk_result["bias"], "sensitive_attr": "__dataset__",
            "score_0to100": lk_result["score"], "detail": lk_result["detail"]
        })
        st.markdown("#### Linkage bias — Records per key")
        show_dup_hist(lk_result.get("dup_counts"), lk_result.get("key", "key"))

    # Evaluation bias
    ev_result = detect_evaluation_bias(df, target_col, score_col, pred_col, sensitive_cols)
    if ev_result:
        report_rows.append({
            "bias": ev_result["bias"], "sensitive_attr": "__dataset__",
            "score_0to100": ev_result["score"], "detail": ev_result["detail"]
        })
        if "per_group_auc" in ev_result:
            st.markdown("#### Evaluation bias — AUC by group")
            show_performance_bars(ev_result["per_group_auc"], "AUC")
        if "per_group_acc" in ev_result:
            st.markdown("#### Evaluation bias — Accuracy by group")
            show_performance_bars(ev_result["per_group_acc"], "Accuracy")

    # Visualization / Presentation bias
    vz_result = detect_visualization_bias(df, sensitive_cols)
    if vz_result:
        report_rows.append({
            "bias": vz_result["bias"], "sensitive_attr": "__dataset__",
            "score_0to100": vz_result["score"], "detail": vz_result["detail"]
        })
        st.markdown("#### Visualization/Presentation bias — Cardinality of sensitive attributes")
        show_cardinality_bars(vz_result.get("card_counts", {}))

    # Interpretive bias
    ip_result = detect_interpretive_bias(df, sensitive_cols)
    if ip_result:
        report_rows.append({
            "bias": ip_result["bias"], "sensitive_attr": "__dataset__",
            "score_0to100": ip_result["score"], "detail": ip_result["detail"]
        })
        st.markdown("#### Interpretive/Label choice bias — 'other/unknown' share by attribute")
        show_other_unknown_bars(ip_result.get("per_attr", {}))

    # Context-shift bias
    cx_result = detect_context_shift_bias(df, sensitive_cols)
    if cx_result:
        report_rows.append({
            "bias": cx_result["bias"], "sensitive_attr": "__dataset__",
            "score_0to100": cx_result["score"], "detail": cx_result["detail"]
        })
        st.markdown("#### Context-shift bias — Group mix per source")
        show_context_heatmaps(cx_result.get("heatmaps", {}), cx_result.get("src_col", "source"))

    if report_rows:
        scorecard = pd.DataFrame(report_rows)
        st.markdown("## Bias Scorecard")
        st.dataframe(scorecard, use_container_width=True)

        csv_bytes = scorecard.to_csv(index=False).encode("utf-8")
        st.download_button("Download scorecard CSV", data=csv_bytes, file_name="bias_scorecard.csv", mime="text/csv")

        json_bytes = json.dumps(report_rows, indent=2).encode("utf-8")
        st.download_button("Download scorecard JSON", data=json_bytes, file_name="bias_scorecard.json", mime="application/json")

    st.markdown("---")
    st.markdown(
        """
**Interpreting scores**  
- 0 = no observed risk; 100 = high observed risk based on heuristics.  
- Some biases require *process* or *external* data to diagnose (enumerator, policy/admin, archival/erasure, benchmark, deployment shift).  
Provide a baseline file for stronger representation checks.

**Notes on new detectors & visuals**
- *Construct*: outcome histogram/bar gives a quick sense of near-constant or dominant-class issues.  
- *Nonresponse*: bar charts show response/complete rates per sensitive group (if a response indicator exists).  
- *Enumerator*: heatmap of outcome by enumerator highlights systematic differences.  
- *Consent*: bars show consent-rate disparities per sensitive attribute.  
- *Labeling*: heatmap of labels by labeler highlights annotator effects.  
- *Archival/Erasure*: early vs late stacked bars display disappearing groups over time.  
- *Aggregation*: entropy bars reveal groups with little to no outcome variation.  
- *Linkage*: histogram of records per ID shows potential over-linkage/duplicate keys.  
- *Evaluation*: bars of AUC/accuracy by group visualize performance gaps.  
- *Visualization/Presentation*: cardinality bars flag attributes likely to force over-aggregation in dashboards.  
- *Interpretive*: bars show reliance on "other/unknown" categories.  
- *Context-shift*: heatmaps display group distributions across sources/environments.  
"""
    )

else:
    st.info("Upload any CSV to begin.")
