"""Préparation des données du dataset Boston Housing."""

import pandas as pd

TARGET = "prix_median"

# Noms d'origine du dataset -> noms explicites utilisés dans les notebooks
COLUMN_NAMES = {
    "CRIM": "tx_crim",
    "ZN": "tx_residence",
    "INDUS": "tx_commerce",
    "CHAS": "riviere",
    "NOX": "tx_nitriq",
    "RM": "nb_piece",
    "AGE": "tx_ancienneté_parc_immo",
    "DIS": "distance_centre_emploi",
    "RAD": "proximité_autoroute",
    "TAX": "indice_impot_foncier",
    "PTRATIO": "ratio_eleve_enseignant",
    "B": "tx_person_couleur",
    "LSTAT": "tx_status_sociaux_eco_inf",
    "MEDV": TARGET,
}


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes d'origine ; échoue si l'une d'elles manque."""
    missing = set(COLUMN_NAMES) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes absentes du dataset : {sorted(missing)}")
    return df.rename(columns=COLUMN_NAMES)
