
# bias_audit_app_compat_impute_robust.py
# -----------------------------------------------------------
# Streamlit app to scan a CSV for common data biases.
# Compatible with sklearn's OneHotEncoder API changes, imputes NaNs,
# and robust to rare classes in the proxy-leakage model.
# Run:  streamlit run bias_audit_app_compat_impute_robust.py
# -----------------------------------------------------------

import io
import json
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import inspect

from typing import Dict, List, Optional, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import chi2_contingency, entropy

# Build OHE kwargs compatible with sklearn>=1.2 (sparse_output) and <1.2 (sparse)
_OHE_KW_BASE = {"handle_unknown": "ignore"}
if 'sparse_output' in inspect.signature(OneHotEncoder).parameters:
    _OHE_KW = dict(_OHE_KW_BASE, sparse_output=False)
else:
    _OHE_KW = dict(_OHE_KW_BASE, sparse=False)

# --------------------------
# Utility functions
# --------------------------

def safe_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr, dtype=float)
    s = arr.sum()
    if s == 0:
        return np.ones_like(arr) / len(arr) if len(arr) else arr
    return arr / s

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon divergence between two distributions."""
    p = safe_normalize(p)
    q = safe_normalize(q)
    m = 0.5 * (p + q)
    kl_pm = entropy(p, m)
    kl_qm = entropy(q, m)
    return float(0.5 * (kl_pm + kl_qm))

def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.array(expected, dtype=float)
    actual = np.array(actual, dtype=float)
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    e_hist = np.histogram(expected, bins=qs)[0]
    a_hist = np.histogram(actual, bins=qs)[0]
    e_pct = safe_normalize(e_hist)
    a_pct = safe_normalize(a_hist)
    psi = np.sum((a_pct - e_pct) * np.log((a_pct + 1e-12) / (e_pct + 1e-12)))
    return float(psi)

def compute_group_distribution(df: pd.DataFrame, group_col: str) -> pd.Series:
    return df[group_col].astype("category").value_counts(dropna=False).sort_index()

def other_or_unknown_rate(df: pd.DataFrame, col: str, keywords=("other","unknown","unspecified","n/a","na","none")) -> float:
    s = df[col].astype(str).str.strip().str.lower()
    mask = s.isin(keywords)
    return float(mask.mean())

def missingness_by_group(df: pd.DataFrame, group_col: str) -> Dict[str, float]:
    rates = {}
    for g, sub in df.groupby(group_col, dropna=False):
        cols = [c for c in df.columns if c != group_col]
        miss = sub[cols].isna().mean().mean()
        rates[str(g)] = float(miss)
    return rates

# --------------------------
# Fairness helpers
# --------------------------

def disparate_impact_ratio(y: pd.Series, group: pd.Series, positive_label=None) -> Tuple[float, Dict[str, float]]:
    if positive_label is None:
        vals = y.dropna().unique().tolist()
        if len(vals) == 2:
            positive_label = vals[1]
        elif len(vals) > 0:
            positive_label = vals[0]
        else:
            return (np.nan, {})
    rates = {}
    for g, sub in y.groupby(group):
        if len(sub) == 0:
            continue
        rates[str(g)] = float((sub == positive_label).mean())
    if not rates:
        return (np.nan, {})
    min_rate = min(rates.values())
    max_rate = max(rates.values())
    if max_rate == 0:
        return (np.nan, rates)
    return (min_rate / max_rate, rates)

def collapse_rare_labels(y: pd.Series, min_count: int = 5, rare_label: str = "__RARE__") -> pd.Series:
    """Collapse categories with frequency < min_count into a single rare bucket."""
    vc = y.value_counts()
    rare = vc[vc < min_count].index
    if len(rare) == 0:
        return y
    return y.where(~y.isin(rare), rare_label)

def predictability_of_sensitive(df: pd.DataFrame, sensitive_col: str, feature_cols: List[str]) -> Tuple[float, List[Tuple[str, float]]]:
    """Predict the sensitive attribute from other features to test proxy leakage.
    - Imputes missing values
    - Collapses rare categories to avoid stratification errors
    - Avoids stratify when classes are too small
    """
    data = df.copy()
    y = data[sensitive_col].astype(str)
    X = data[feature_cols]

    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False)),
            ]), num_cols),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(**_OHE_KW)),
            ]), cat_cols),
        ],
        remainder="drop"
    )
    clf = LogisticRegression(max_iter=1000, multi_class="auto" )
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    mask = ~y.isna()
    X2, y2 = X[mask], y[mask]

    # Need enough examples
    if len(X2) < 50:
        return (np.nan, [])

    # Collapse rare classes then optionally drop still-rare singletons
    y2c = collapse_rare_labels(y2, min_count=5, rare_label="__RARE__")
    vc = y2c.value_counts()
    if len(vc) < 2:
        return (np.nan, [])
    if vc.min() < 2:
        # drop classes that still have <2 samples to avoid stratify issues
        keep = y2c.isin(vc[vc >= 2].index)
        X2, y2c = X2[keep], y2c[keep]
        if len(y2c.value_counts()) < 2:
            return (np.nan, [])

    strat_arg = y2c if y2c.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X2, y2c, test_size=0.3, random_state=42, stratify=strat_arg
    )

    pipe.fit(X_train, y_train)
    try:
        proba = pipe.predict_proba(X_test)
        classes = pipe.named_steps["clf"].classes_
        y_bin = pd.get_dummies(y_test).reindex(columns=classes, fill_value=0).values
        auc = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
    except Exception:
        pred = pipe.predict(X_test)
        auc = accuracy_score(y_test, pred)

    # MI proxy features
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
    if np.isnan(jsd):
        return np.nan
    return float(min(100.0, (jsd / 0.5) * 100.0))

def score_missingness(max_gap: float) -> float:
    if np.isnan(max_gap):
        return np.nan
    return float(np.interp(max_gap, [0.01, 0.4], [0, 100]))

def score_disparate_impact(diratio: float) -> float:
    if np.isnan(diratio):
        return np.nan
    diratio = max(min(diratio, 1.0), 0.0)
    return float((1.0 - diratio) * 100)

def score_proxy_leakage(auc: float) -> float:
    if np.isnan(auc):
        return np.nan
    if auc <= 0.5:
        return 0.0
    return float(np.interp(auc, [0.5, 0.8, 0.9, 1.0], [0, 80, 95, 100]))

def score_other_bucket(rate: float) -> float:
    if np.isnan(rate):
        return np.nan
    return float(np.interp(rate, [0.01, 0.3], [0, 100]))

def score_temporal_drift(psi: float) -> float:
    if np.isnan(psi):
        return np.nan
    return float(np.interp(psi, [0.0, 0.1, 0.25, 0.5], [0, 30, 70, 100]))

# --------------------------
# Streamlit UI
# --------------------------

st.set_page_config(page_title="Bias Audit", layout="wide")
st.title("CSV Bias Audit (Compat + Impute + Robust)")

st.write(
    """
Upload a CSV and select:
- **Sensitive attribute(s)** (e.g., race, gender, age),
- Optional **outcome/label** column (to check outcome disparity),
- Optional **prediction** or **score** column(s) (to check model performance disparity),
- Optional **timestamp** (to assess historical drift),
- Optional **baseline CSV** mapping groups to population shares for representation checks.
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

    suggested = [c for c in df.columns if any(k in c.lower() for k in ["race","gender","age","zip","language","ethnic"])]
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

    for s_col in sensitive_cols:
        st.subheader(f"Sensitive attribute: {s_col}")

        # Representation
        dist = compute_group_distribution(df, s_col)
        groups = [str(i) for i in dist.index.tolist()]
        counts = dist.values.astype(float)
        sample_p = safe_normalize(counts)
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

        fig1, ax1 = plt.subplots()
        ax1.bar(range(len(groups)), sample_p)
        ax1.set_xticks(range(len(groups)))
        ax1.set_xticklabels(groups, rotation=45, ha="right")
        ax1.set_ylabel("Share in dataset")
        ax1.set_title(f"Representation of {s_col}")
        st.pyplot(fig1)

        report_rows.append({
            "bias": "Coverage / Representation",
            "sensitive_attr": s_col,
            "score_0to100": round(rep_score, 1) if not np.isnan(rep_score) else np.nan,
            "detail": f"JSD={jsd:.3f} {rep_note}" if not np.isnan(jsd) else "insufficient data"
        })

        # Missingness
        miss_rates = missingness_by_group(df, s_col)
        max_gap = max(miss_rates.values()) if miss_rates else np.nan
        miss_score = score_missingness(max_gap)

        fig2, ax2 = plt.subplots()
        ax2.bar(range(len(miss_rates)), list(miss_rates.values()))
        ax2.set_xticks(range(len(miss_rates)))
        ax2.set_xticklabels(list(miss_rates.keys()), rotation=45, ha="right")
        ax2.set_ylabel("Avg missingness across columns")
        ax2.set_title(f"Missingness by {s_col}")
        st.pyplot(fig2)

        report_rows.append({
            "bias": "Missingness / Data quality by group",
            "sensitive_attr": s_col,
            "score_0to100": round(miss_score, 1) if not np.isnan(miss_score) else np.nan,
            "detail": f"Max group missingness={max_gap:.3f}" if not np.isnan(max_gap) else "n/a"
        })

        # Labeling
        if df[s_col].dtype == object or str(df[s_col].dtype).startswith("category"):
            other_rate = other_or_unknown_rate(df, s_col)
            other_score = score_other_bucket(other_rate)
            report_rows.append({
                "bias": "Labeling / 'Other-Unknown' bucket",
                "sensitive_attr": s_col,
                "score_0to100": round(other_score, 1) if not np.isnan(other_score) else np.nan,
                "detail": f"Rate of ['other','unknown',...] in {s_col} = {other_rate:.3f}"
            })

        # Outcome disparity
        if target_col:
            diratio, rates = disparate_impact_ratio(df[target_col], df[s_col])
            di_score = score_disparate_impact(diratio)

            if rates:
                fig3, ax3 = plt.subplots()
                ax3.bar(range(len(rates)), list(rates.values()))
                ax3.set_xticks(range(len(rates)))
                ax3.set_xticklabels(list(rates.keys()), rotation=45, ha="right")
                ax3.set_ylabel("Positive outcome rate")
                ax3.set_title(f"Outcome rates by {s_col}")
                st.pyplot(fig3)

            report_rows.append({
                "bias": "Outcome disparity (four-fifths rule)",
                "sensitive_attr": s_col,
                "score_0to100": round(di_score, 1) if not np.isnan(di_score) else np.nan,
                "detail": f"Disparate impact ratio={diratio:.3f} (min/max positive rates)" if not np.isnan(diratio) else "n/a"
            })

        # Proxy leakage (robust)
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

        # Drift
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
"""
    )

else:
    st.info("Upload any CSV to begin.")
