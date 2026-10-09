import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from pycaret_optuna.data import TARGET
from pycaret_optuna.modeling import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_pipeline,
    build_preprocessor,
    cross_val_r2,
    regression_metrics,
    split_train_test,
)


def test_les_variables_couvrent_gold_sans_recouvrement(gold):
    features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    assert len(features) == len(set(features))
    assert set(features) == set(gold.columns) - {TARGET}


def test_split_train_test_taille_et_separation(gold):
    X_train, X_test, y_train, y_test = split_train_test(gold)
    assert (len(X_train), len(X_test)) == (404, 102)
    assert TARGET not in X_train.columns
    assert X_train.index.intersection(X_test.index).empty
    assert y_train.index.equals(X_train.index)


def test_split_train_test_reproductible(gold):
    first = split_train_test(gold)[0].index
    second = split_train_test(gold)[0].index
    assert first.equals(second)


def test_split_train_test_decoupage_de_reference(gold):
    # Les scores des notebooks dépendent de ce découpage précis (mélange, random_state=42)
    _, X_test, *_ = split_train_test(gold)
    assert list(X_test.index[:5]) == [173, 274, 491, 72, 452]


def test_preprocesseur_forme_de_sortie(gold):
    X_train, *_ = split_train_test(gold)
    out = build_preprocessor().fit_transform(X_train)
    n_categories = X_train["proximité_autoroute"].nunique()
    assert out.shape == (len(X_train), n_categories + len(NUMERICAL_FEATURES))


def test_preprocesseur_standardise_les_variables_numeriques(gold):
    X_train, *_ = split_train_test(gold)
    pre = build_preprocessor().fit(X_train)
    num = pre.named_transformers_["num"].transform(X_train[NUMERICAL_FEATURES])
    np.testing.assert_allclose(num.mean(axis=0), 0, atol=1e-9)
    np.testing.assert_allclose(num.std(axis=0), 1, atol=1e-9)


def test_preprocesseur_ignore_une_categorie_inconnue(gold):
    X_train, X_test, *_ = split_train_test(gold)
    pre = build_preprocessor().fit(X_train)
    unknown = X_test.head(1).copy()
    unknown["proximité_autoroute"] = 999
    out = pre.transform(unknown)
    out = out.toarray() if hasattr(out, "toarray") else out
    n_categories = X_train["proximité_autoroute"].nunique()
    # Catégorie inconnue : toutes les colonnes one-hot à zéro, sans erreur
    assert out[0, :n_categories].sum() == 0


def test_build_pipeline_etapes_et_prediction(gold):
    X_train, X_test, y_train, _ = split_train_test(gold)
    pipe = build_pipeline(LinearRegression())
    assert list(pipe.named_steps) == ["preprocessor", "model"]
    pred = pipe.fit(X_train, y_train).predict(X_test)
    assert pred.shape == (len(X_test),)


def test_build_pipeline_cree_un_preprocesseur_neuf_a_chaque_appel():
    assert build_pipeline(LinearRegression())[0] is not build_pipeline(LinearRegression())[0]


def test_cross_val_r2_modele_constant_proche_de_zero(gold):
    X_train, _, y_train, _ = split_train_test(gold)
    # Prédire la moyenne donne un R² nul (légèrement négatif hors échantillon)
    assert cross_val_r2(DummyRegressor(), X_train, y_train) == pytest.approx(0, abs=0.05)


def test_cross_val_r2_regression_lineaire(gold):
    X_train, _, y_train, _ = split_train_test(gold)
    r2 = cross_val_r2(LinearRegression(), X_train, y_train)
    assert 0.6 < r2 < 0.9
    assert r2 == cross_val_r2(LinearRegression(), X_train, y_train)


def test_cross_val_r2_n_utilise_que_les_donnees_fournies():
    # Cible parfaitement linéaire : R² de 1 quel que soit le découpage
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, len(NUMERICAL_FEATURES))), columns=NUMERICAL_FEATURES)
    X["proximité_autoroute"] = rng.integers(1, 4, size=50)
    y = 3 * X["nb_piece"] - 2 * X["tx_crim"]
    assert cross_val_r2(LinearRegression(), X, y) == pytest.approx(1)


class ExpectsPreprocessedInput(RegressorMixin, BaseEstimator):
    """Prédit la moyenne, mais refuse des données qui n'ont pas été préparées."""

    def fit(self, X, y):
        X = np.asarray(X)
        # Après one-hot, il y a plus de colonnes que de variables d'origine
        if X.shape[1] <= len(CATEGORICAL_FEATURES) + len(NUMERICAL_FEATURES):
            raise ValueError("données non préparées")
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_cross_val_r2_applique_le_preprocesseur(gold):
    X_train, _, y_train, _ = split_train_test(gold)
    # Un échec de fit donnerait un score NaN
    assert np.isfinite(cross_val_r2(ExpectsPreprocessedInput(), X_train, y_train))


def test_regression_metrics_valeurs_connues():
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    assert metrics["mse"] == pytest.approx(4 / 3)
    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["r2"] == pytest.approx(1 - 4 / 2)


def test_regression_metrics_prediction_parfaite():
    assert regression_metrics([1.0, 2.0], [1.0, 2.0]) == {"mse": 0.0, "mae": 0.0, "r2": 1.0}
