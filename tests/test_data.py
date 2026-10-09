import pandas as pd
import pytest

from pycaret_optuna.data import COLUMN_NAMES, TARGET, rename_columns


def test_rename_columns_renomme_toutes_les_colonnes(bronze):
    renamed = rename_columns(bronze)
    assert list(renamed.columns) == [COLUMN_NAMES[c] for c in bronze.columns]


def test_rename_columns_ne_modifie_pas_les_valeurs(bronze):
    renamed = rename_columns(bronze)
    assert (renamed.to_numpy() == bronze.to_numpy()).all()


def test_rename_columns_ne_modifie_pas_l_entree(bronze):
    before = list(bronze.columns)
    rename_columns(bronze)
    assert list(bronze.columns) == before


def test_rename_columns_signale_une_colonne_manquante(bronze):
    with pytest.raises(ValueError, match="MEDV"):
        rename_columns(bronze.drop(columns=["MEDV"]))


def test_noms_de_colonnes_uniques():
    assert len(set(COLUMN_NAMES.values())) == len(COLUMN_NAMES)


# Cohérence des fichiers produits par 01_cleaning et 02_analyse


def test_silver_correspond_a_bronze_renomme_sans_b(bronze, silver):
    expected = rename_columns(bronze).drop(columns=["tx_person_couleur"])
    pd.testing.assert_frame_equal(silver, expected)


def test_gold_correspond_a_silver_sans_riviere(silver, gold):
    pd.testing.assert_frame_equal(gold, silver.drop(columns=["riviere"]))


@pytest.mark.parametrize("name", ["bronze", "silver", "gold"])
def test_aucune_valeur_manquante(name, request):
    df = request.getfixturevalue(name)
    assert len(df) == 506
    assert not df.isna().any().any()


def test_cible_dans_l_intervalle_du_dataset(gold):
    # MEDV est exprimé en milliers de dollars et plafonné à 50 dans le dataset d'origine
    assert gold[TARGET].between(5, 50).all()
