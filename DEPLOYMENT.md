# Deployment Guide

PromoLift AI is ready to deploy as a Streamlit app.

## Local Run Command

From the project root:

```powershell
streamlit run app/streamlit_app.py
```

## Streamlit Community Cloud Setup

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from the GitHub repository.
4. Set the main app file to:

```text
app/streamlit_app.py
```

5. Deploy the app.

## Deployment Settings

- Main app file: `app/streamlit_app.py`
- Python runtime: Python 3.11
- Runtime file: `runtime.txt`
- Streamlit config: `.streamlit/config.toml`
- Dependencies: `requirements.txt`

## Important Note

Generated reports and figures are committed so the dashboard can load them without retraining models during app startup. The dashboard reads saved CSV, JSON, Markdown, and PNG artifacts from `reports/`.

Raw and processed data files are intentionally ignored by Git. They can be regenerated locally with:

```powershell
python -m src.data.load_hillstrom
```
