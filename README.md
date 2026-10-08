# Pycaret_Optuna

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

Python 3.9 à 3.11 (contrainte de Pycaret) :

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
