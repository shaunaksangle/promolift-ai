"""Run the full PromoLift AI pipeline.

This script regenerates project outputs in the correct order. It does not start
the Streamlit dashboard.
"""

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_STEPS = [
    ("Data loading and preprocessing", "src.data.load_hillstrom"),
    ("EDA and treatment/control analysis", "src.analysis.eda_hillstrom"),
    ("Causal EDA and overlap diagnostics", "src.analysis.causal_eda"),
    ("Baseline conversion model", "src.models.baseline_model"),
    ("Uplift modeling", "src.models.uplift_model"),
    ("Causal validation", "src.causal.causal_validation"),
]


def run_step(step_number, step_name, module_name):
    """Run one pipeline module and stop if it fails."""
    print(f"\n[{step_number}/{len(PIPELINE_STEPS)}] {step_name}")
    print(f"Running: python -m {module_name}")

    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=PROJECT_ROOT,
        check=False,
    )

    if result.returncode != 0:
        print(f"\nPipeline stopped. Failed step: {step_name}")
        print(f"Module: {module_name}")
        sys.exit(result.returncode)

    print(f"Completed: {step_name}")


def main():
    """Run all pipeline steps in order."""
    print("Starting PromoLift AI pipeline...")

    for index, (step_name, module_name) in enumerate(PIPELINE_STEPS, start=1):
        run_step(index, step_name, module_name)

    print("\nPipeline completed successfully.")
    print("To open the dashboard, run: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
