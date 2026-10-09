#!/usr/bin/env bash
# Exécute tous les notebooks dans l'ordre du pipeline, avec l'environnement uv.
# Les notebooks exécutés sont écrits dans executed/ : ceux du dépôt ne sont pas modifiés
# (01_cleaning et 02_analyse réécrivent toutefois data/silver.csv et data/gold.csv).
# N_TRIALS fixe le nombre d'essais Optuna par modèle (100 par défaut).
set -euo pipefail
cd "$(dirname "$0")/.."

NOTEBOOKS=(01_cleaning 02_analyse 03_featureimportance 03_pycaret 04_et 04_rf 04_gbr 04_catboost)

for nb in "${NOTEBOOKS[@]}"; do
    echo "=== $nb (N_TRIALS=${N_TRIALS:-100})"
    uv run --locked jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.kernel_name=python3 \
        --ExecutePreprocessor.timeout=3600 \
        --output-dir executed "notebook/$nb.ipynb"
done
