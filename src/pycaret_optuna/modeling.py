"""Préprocesseur, pipeline et évaluation communs aux notebooks 04_*."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pycaret_optuna.data import TARGET

RANDOM_STATE = 42

CATEGORICAL_FEATURES = ["proximité_autoroute"]

NUMERICAL_FEATURES = [
    "tx_crim",
    "tx_residence",
    "tx_commerce",
    "tx_nitriq",
    "nb_piece",
    "tx_ancienneté_parc_immo",
    "distance_centre_emploi",
    "indice_impot_foncier",
    "ratio_eleve_enseignant",
    "tx_status_sociaux_eco_inf",
]


def split_train_test(df: pd.DataFrame, train_size: float = 0.8):
    """Sépare variables explicatives et cible, puis jeu d'entraînement et de test."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, train_size=train_size, shuffle=True, random_state=RANDOM_STATE)


def build_preprocessor() -> ColumnTransformer:
    """One-hot sur les variables catégorielles, standardisation des numériques."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(sparse_output=True, handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERICAL_FEATURES),
        ],
        remainder="passthrough",
    )


def build_pipeline(model) -> Pipeline:
    """Pipeline préprocesseur + modèle utilisé pour l'entraînement et l'évaluation."""
    return Pipeline([("preprocessor", build_preprocessor()), ("model", model)])


def cross_val_r2(model, X, y, n_splits: int = 5) -> float:
    """R² moyen en validation croisée du pipeline complet (à utiliser sur X_train uniquement)."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(build_pipeline(model), X, y, cv=cv, scoring="r2", n_jobs=-1).mean()


def regression_metrics(y_true, y_pred) -> dict:
    """MSE, MAE et R²."""
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
