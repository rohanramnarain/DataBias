# Bias Audit Toolkit

*A Streamlit app to scan CSV datasets for common data biases.*  
It combines classic representation/quality checks with heuristic detectors for process‑level biases (enumerator, consent, labeling, etc.), and renders side‑by‑side visualizations plus a downloadable **Bias Scorecard**.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Inputs & Column Conventions](#inputs--column-conventions)
- [What’s Detected (Definitions & Signals)](#whats-detected-definitions--signals)
- [Scorecard](#scorecard)
- [Visualizations](#visualizations)
- [Configuration & Binning](#configuration--binning)
- [Design Choices & Limits](#design-choices--limits)
- [Privacy](#privacy)
- [Extending the Toolkit](#extending-the-toolkit)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Features

### Upload & Configure
- Drag‑and‑drop **CSV/TSV/TXT**, auto‑delimiter detection
- Optional **baseline CSV** for representation checks (`group,share`)
- Select **sensitive attributes**, **outcome/label**, **predictions/scores**, and **timestamp**

### Representation & Quality
- Group distribution, **AGE 5‑year binning**, **ZIP3 grouping**
- **Missingness by group**
- **Outcome disparity** (four‑fifths rule)
- **Proxy leakage** (predictability of sensitive attributes)
- **Temporal drift** (PSI early vs. late)

### Heuristic Detectors *(New)*
Construct • Nonresponse • Enumerator • Consent • Labeling • Archival/Erasure • Proxy • Aggregation • Linkage • Historical • Evaluation • Visualization/Presentation • Interpretive (label choice) • Context‑shift

### Visualizations
- Distributions, response/consent rate bars
- Heatmaps (labeler × label, enumerator × outcome, source × group)
- Early vs. late comparison bars
- Entropy bars (aggregation risk), AUC/accuracy by group
- Cardinality bars & “other/unknown” share

### Artifacts
- **Bias Scorecard** displayed in‑app
- **CSV and JSON** downloads of findings

---

## Quick Start

### 1) Environment
- Python ≥ **3.9**
- Recommended: create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scriptsctivate
```

### 2) Install
```bash
pip install -U streamlit pandas numpy matplotlib scikit-learn scipy
```

### 3) Run the App
```bash
streamlit run bias_audit_app_compat_impute_robust.py
```
Open the provided local URL in your browser.

---

## Inputs & Column Conventions

The app auto‑detects columns by **name fragments**. You can always select columns manually in the UI.

| Purpose | Examples of Auto‑Detected Names |
|---|---|
| Sensitive attributes | `race`, `gender`, `age`, `zip`, `zipcode`, `postal`, `language`, `ethnic` |
| Outcome / Label | Any selected column |
| Prediction (class) | Any selected column |
| Score / Probability (0–1) | Any selected column |
| Timestamp | Any selected column *(parsed to datetime)* |
| Enumerator / Interviewer | `enumerator`, `interviewer`, `collector`, `agent`, `staff` |
| Consent / Opt‑in | `consent`, `assent`, `opt_in`, `optout`, `opt-out`, `opt out` |
| Response Indicator | `respond`, `response`, `completed`, `finished` |
| Labeler / Annotator | `labeler`, `annotator`, `rater`, `coder`, `source` |
| Record Key | `id`, `subject_id`, `patient_id`, `record_id`, `uuid`, `guid` |
| Source / Domain | `source`, `environment`, `dataset`, `phase`, `split`, `domain` |

> **Tip:** If your schema uses different headers, rename columns (or open an issue with the header names and we can add them).

---

## What’s Detected (Definitions & Signals)

> These are **heuristics** meant to flag risk, not prove causation.

- **Coverage / Representation** — Group share vs baseline or uniform. *(Jensen–Shannon divergence; AGE/ZIP binning available)*
- **Missingness by Group** — Average missingness across non‑sensitive columns for each group.
- **Outcome Disparity** — Four‑fifths rule on positive outcome rates.
- **Proxy Leakage** — How well features predict a sensitive attribute *(LogReg + preprocessing; AUC/accuracy + top mutual‑information features)*.
- **Historical Drift** — Population Stability Index *(PSI early vs late)*.
- **Construct Bias** — Near‑constant numeric outcomes or dominant class in categorical outcome.
- **Nonresponse Bias** — Response/complete rate disparity across sensitive groups.
- **Enumerator Bias** — Association between enumerator/interviewer and outcomes. *(χ² test on crosstab)*
- **Consent Bias** — Consent rate disparity across groups.
- **Labeling Bias** — Label distribution varies by labeler/annotator. *(χ² on crosstab)*
- **Archival/Erasure Bias** — Groups present early but missing late *(requires timestamp)*.
- **Aggregation Bias** — Groups with zero variation in outcomes; low within‑group entropy.
- **Linkage Bias** — Duplicate rates in putative unique keys.
- **Evaluation Bias** — Performance spread *(AUC or accuracy)* across groups.
- **Visualization/Presentation Bias** — Very high‑cardinality sensitive fields prone to misleading aggregation.
- **Interpretive (Label Choice) Bias** — Heavy use of “other/unknown”.
- **Context‑Shift Bias** — Group‑mix divergence across sources/domains. *(Pairwise JS divergence on row‑normalized crosstabs)*

---

## Scorecard

All detectors append rows to a unified **Bias Scorecard** with fields:

- `bias` — detector name  
- `sensitive_attr` — the attribute examined *(or `__dataset__`)*  
- `score_0to100` — higher = higher observed risk  
- `detail` — short rationale/metric *(e.g., `JSD=0.163`, `AUC≈0.74`, `p≈2.4e-5`)*

You can download the full **CSV** and **JSON** from the app.

---

## Visualizations

Representative plots shown inline (automatically if the needed columns exist):

- Outcome histograms / label counts
- Response & consent rate bars by group
- Heatmaps: enumerator × outcome, labeler × label, source × sensitive
- Early vs. late group counts
- Within‑group outcome entropy bars
- Performance bars (AUC/accuracy) by group
- Cardinality bars; “other/unknown” share bars

---

## Configuration & Binning

- **AGE** → 5‑year bins when numeric *(toggle in UI)*  
- **ZIP/Postal** → ZIP3 grouping *(first 3 digits; toggle in UI)*

**Baseline CSV format**:
```csv
group,share
A,0.40
B,0.35
C,0.25
```

---

## Design Choices & Limits

- Detectors are **data‑only heuristics**; some bias types require process audits or external metadata.
- Statistical tests (e.g., χ²) require adequate counts; **small groups are auto‑skipped** when unstable.
- Proxy leakage uses a simple logistic model with basic preprocessing; results are **directional**.
- **Score scales are interpretable heuristics**, not regulatory thresholds.

---

## Privacy

- All analysis runs **locally** where you execute Streamlit.
- **Do not upload PHI/PII** unless your environment complies with your org’s policies.
- Consider hashing/aliasing keys and masking free‑text before analysis.

---

## Extending the Toolkit

Add a new detector by following the pattern:

```
def detect_<name>_bias(df, ...) -> Optional[Dict]:
    """Return keys: bias, score_0to100, detail, and any plotting extras (e.g., crosstabs)."""
    ...
```
1. Return keys: **`bias`**, **`score_0to100`**, **`detail`**, and any plotting extras.  
2. Add optional plotting helper(s).  
3. Append results to `report_rows` and render plots if present.  
4. All new logic should be **additive** (no changes to existing behavior).

---

## Troubleshooting

- **“Could not parse CSV”** → Check delimiter/encoding/quote char in **Reader options**.
- **Charts missing** → The underlying detector likely didn’t find the required columns or there weren’t enough rows.
- **All scores NaN/blank** → Ensure you selected **sensitive attributes**, and where applicable, the **outcome/score/time** columns.
- **Performance plots empty** → Evaluation bias needs either a **score/probability** column *or* a **predicted class** column plus an outcome.

---

## Contributing

- Open an issue describing the bias type or visualization.
- Include a small synthetic CSV demonstrating the failure/signal.
- Submit a PR with:
  - A self‑contained detector + tests (if feasible)
  - Clear notes on assumptions and thresholds

---

## License

**MIT** (or your organization’s preferred license). Add the license file to the repo root.

---

## Acknowledgments

Inspired by fairness auditing practices across industry and academia. JS divergence, PSI, MI, and χ² are standard building blocks adapted here for pragmatic dataset screening.

---

## Contact

Questions, ideas, or integration requests? Open an issue or reach out to the maintainers.
