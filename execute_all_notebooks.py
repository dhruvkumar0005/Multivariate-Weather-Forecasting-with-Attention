"""Notebook Execution Script.
Executes all 4 Jupyter notebooks end-to-end and saves the rendered cell outputs.
"""

import sys
import time
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

notebook_files = [
    "01_data_cleaning_and_eda.ipynb",
    "02_feature_eng_and_sequences.ipynb",
    "03_model_training_and_eval.ipynb",
    "04_attention_interpretability.ipynb"
]

ep = ExecutePreprocessor(timeout=1200, kernel_name="python3")

for nb_file in notebook_files:
    nb_path = NOTEBOOKS_DIR / nb_file
    print(f"\n{'='*60}")
    print(f"[*] Executing Notebook: {nb_file}")
    print(f"{'='*60}")
    t0 = time.time()
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    
    # Execute with working directory set to notebooks directory
    ep.preprocess(nb, {"metadata": {"path": str(NOTEBOOKS_DIR)}})
    
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
        
    elapsed = time.time() - t0
    print(f"[+] Successfully executed {nb_file} in {elapsed:.1f}s with all cell outputs saved!")

print("\n[DONE] All 4 notebooks executed and verified with complete outputs!")
