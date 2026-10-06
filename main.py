"""
main.py — Full end-to-end pipeline runner for
Multivariate Weather Forecasting with Attention.

Run steps individually or all at once:
    python main.py            # runs the full pipeline
    python main.py --step 1   # run only one step (1-5)
"""

import argparse
import sys
from pathlib import Path

# ── ensure project root is importable ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def step1_clean_data():
    print("\n" + "=" * 60)
    print("STEP 1: Data Loading, Cleaning & Hourly Resampling")
    print("=" * 60)
    from src.data_loader import prepare_and_save_data
    from src.config import CLEANED_DATA_PATH
    if CLEANED_DATA_PATH.exists():
        print(f"[+] Cleaned dataset already exists at {CLEANED_DATA_PATH}. Skipping.")
    else:
        prepare_and_save_data()


def step2_eda():
    print("\n" + "=" * 60)
    print("STEP 2: Exploratory Data Analysis — 5 Figures")
    print("=" * 60)
    from src.visualization import run_eda_pipeline
    run_eda_pipeline()


def step3_train():
    print("\n" + "=" * 60)
    print("STEP 3: Train Baseline LSTM, Stacked GRU & GRU+Attention")
    print("=" * 60)
    from src.train import run_all_training
    run_all_training()


def step4_evaluate():
    print("\n" + "=" * 60)
    print("STEP 4: Evaluate Models & Generate Comparison Table + Plots")
    print("=" * 60)
    from src.evaluate import run_evaluation_pipeline
    run_evaluation_pipeline()


def step5_notebooks():
    print("\n" + "=" * 60)
    print("STEP 5: Generate Jupyter Notebooks")
    print("=" * 60)
    import subprocess
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "create_notebooks.py")],
        cwd=str(PROJECT_ROOT)
    )
    if result.returncode == 0:
        print("[+] All notebooks created successfully.")
    else:
        print("[!] Notebook creation failed.")


STEPS = {
    1: ("Data Cleaning & Resampling", step1_clean_data),
    2: ("EDA Visualisations",         step2_eda),
    3: ("Model Training",             step3_train),
    4: ("Model Evaluation",           step4_evaluate),
    5: ("Notebook Generation",        step5_notebooks),
}


def main():
    parser = argparse.ArgumentParser(
        description="Multivariate Weather Forecasting with Attention — Pipeline Runner"
    )
    parser.add_argument(
        "--step", type=int, choices=range(1, 6),
        help="Run only a specific step (1–5). Omit to run all steps."
    )
    args = parser.parse_args()

    print("\n  Multivariate Weather Forecasting with Attention")
    print("    Full Pipeline Runner")
    print("=" * 60)

    if args.step:
        name, fn = STEPS[args.step]
        print(f"Running step {args.step}: {name}")
        fn()
    else:
        for step_num, (name, fn) in STEPS.items():
            fn()

    print("\n" + "=" * 60)
    print("[DONE] Pipeline finished successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
