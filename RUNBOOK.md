# PromoLift AI Runbook

This runbook explains how to regenerate the PromoLift AI project outputs from scratch.

## 1. Environment Setup

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If the environment is already created, activate it before running any project commands:

```powershell
.venv\Scripts\activate
```

## 2. Run the Full Pipeline

To regenerate all non-dashboard outputs in order:

```powershell
python scripts/run_all.py
```

This runs:

1. Data loading and preprocessing
2. EDA generation
3. Causal EDA generation
4. Baseline conversion model
5. Uplift model
6. Causal validation

The script stops immediately if a step fails.

## 3. Data Loading

```powershell
python -m src.data.load_hillstrom
```

This downloads the Hillstrom Email Marketing dataset if needed and writes the processed binary treatment dataset to:

```text
data/processed/hillstrom_mens_email.csv
```

## 4. EDA Generation

```powershell
python -m src.analysis.eda_hillstrom
```

This creates treatment/control summary tables, segment uplift tables, charts, and the EDA markdown report.

Key outputs:

```text
reports/eda/
reports/figures/
```

## 5. Causal EDA Generation

```powershell
python -m src.analysis.causal_eda
```

This creates causal-focused EDA artifacts: treatment balance plots, propensity overlap diagnostics, naive versus stratified-adjusted effect comparisons, subgroup heterogeneity visuals, a DAG sketch, and a causal EDA markdown report.

Key outputs:

```text
reports/causal_eda/
reports/figures/
```

## 6. Baseline Model

```powershell
python -m src.models.baseline_model
```

This trains the standard conversion prediction model and saves baseline metrics, model comparison tables, and charts.

Key outputs:

```text
reports/modeling/
reports/figures/
```

## 7. Uplift Model

```powershell
python -m src.models.uplift_model
```

This compares T-Learner and S-Learner uplift models and saves model comparison, policy value tables, and uplift charts.

Key outputs:

```text
reports/uplift/
reports/figures/
```

## 8. Causal Validation

```powershell
python -m src.causal.causal_validation
```

This estimates the observed ATE, checks covariate balance, calculates propensity diagnostics, and runs the optional DoWhy validation layer when available.

Key outputs:

```text
reports/causal/
reports/figures/
```

## 9. Streamlit Dashboard

After the pipeline outputs exist, run:

```powershell
streamlit run app/streamlit_app.py
```

The dashboard reads saved CSV, JSON, Markdown, and PNG outputs. It does not retrain models.

## Common Troubleshooting

### Missing processed dataset

Run:

```powershell
python -m src.data.load_hillstrom
```

### Missing EDA charts or report files

Run:

```powershell
python -m src.analysis.eda_hillstrom
python -m src.analysis.causal_eda
```

### Missing baseline, uplift, or causal outputs

Run the relevant module:

```powershell
python -m src.models.baseline_model
python -m src.models.uplift_model
python -m src.causal.causal_validation
```

### Streamlit cannot find dashboard artifacts

Regenerate the full pipeline:

```powershell
python scripts/run_all.py
```

### Package import errors

Confirm the virtual environment is activated and dependencies are installed:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

### Data files not visible on GitHub

This is expected. Raw and processed data files are ignored by Git and should be regenerated locally.
