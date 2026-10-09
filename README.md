# Pycaret_Optuna

[![Notebooks](https://github.com/Olaffson/Pycaret_Optuna/actions/workflows/notebooks.yml/badge.svg?branch=main)](https://github.com/Olaffson/Pycaret_Optuna/actions/workflows/notebooks.yml)

recuperation du dataset :
https://faculty.tuck.dartmouth.edu/images/uploads/faculty/business-analytics/Boston_Housing.xlsx

autre lien util :
https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html

context du projet :
Depuis plusieurs années, les prix de vente des maisons explosent à Boston en raison de la spéculation. La municipalité soucieuse de rendre les logements accessibles au plus grand nombre décide de plafonner les prix de ventes dans une fourchette proches des prix actuels (qui sont déja très hauts). Cependant pour que ce prix soit juste, il faut qu'il intègre les caractéristiques propres aux logements et au quartier. Elle vous partage à cette fin des données liées à l'immobilier dans la ville (voir dataset en lien).
La municipalité désire donc que vous créez un modèle permettant de lier l’adresse et les caractéristiques du logement afin de pouvoir déterminer pour chaque logement le prix maximal auquel il peut être vendu.

Vous devez donc:
Créer un modèle d'IA.
Présenter le modèle obtenu en accord avec les bonnes pratiques du métier et sous format slides.


Le but du projet est de réaliser un modèle d IA rapidement a l'aide de Pycaret et Optuna, et de réaliser une veille sur les features importances.

Pycaret permet de determiner les algo (ici de regression) les plus pertinants.
Optuna permet d optimiser les hyperparametres (plus rapide que la methode GridSearch)

## Démarche

- la PCA n a pas été utilisée ici car il y a peu de colonnes
- Pycaret permet de determiner les algo de regression les plus pertinants
- les differents algo de regression sont ensuite testés individuellement
- on utilise egalement optuna afin d optimiser les hyperparametres

## Notebooks

1. `01_cleaning` : renommage des colonnes, suppression de `B` → `data/silver.csv`
2. `02_analyse` : profiling, suppression de `riviere` → `data/gold.csv`
3. `03_pycaret` : comparaison des algorithmes de regression avec Pycaret
4. `04_et`, `04_rf`, `04_gbr`, `04_catboost` : chaque modele teste individuellement, hyperparametres optimises avec Optuna (validation croisee sur le jeu d entrainement, evaluation finale sur le jeu de test)

## Installation

Les dépendances sont gérées avec [uv](https://docs.astral.sh/uv/) : `pyproject.toml` liste les dépendances, `uv.lock` fige toutes les versions. uv installe lui-même Python 3.11 (Pycaret ne supporte pas Python 3.12 et plus).

```bash
# installer uv (une seule fois)
curl -LsSf https://astral.sh/uv/install.sh | sh

# créer l'environnement à partir du fichier de verrouillage
uv sync

# lancer Jupyter
uv run jupyter lab
```

Ajouter une dépendance : `uv add <paquet>` (met à jour `pyproject.toml` et `uv.lock`).

Sans uv, avec pip : `uv export --no-hashes --no-dev > requirements.txt` puis `pip install -r requirements.txt` dans un environnement Python 3.11.

## Exécuter tous les notebooks

```bash
./scripts/run_notebooks.sh              # 100 essais Optuna par modèle
N_TRIALS=5 ./scripts/run_notebooks.sh   # rapide, pour vérifier que tout s'exécute
```

Les notebooks exécutés sont écrits dans `executed/` (ignoré par git).

## Intégration continue

Le workflow GitHub Actions `Notebooks` (`.github/workflows/notebooks.yml`) installe l'environnement depuis `uv.lock`, exécute tous les notebooks avec 5 essais Optuna et vérifie que `silver.csv` et `gold.csv` sont régénérés à l'identique. Il tourne à chaque pull request et à chaque push sur `main` ; il peut aussi être lancé à la main (onglet Actions) avec un autre nombre d'essais. Les notebooks exécutés sont disponibles dans l'artefact `notebooks-executes`.
