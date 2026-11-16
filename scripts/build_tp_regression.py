"""Génère les notebooks du TP de régression linéaire à partir de son README.

Usage : python scripts/build_tp_regression.py [--executer]

Le README (tp/regression_lineaire/README.md) est la seule source du code : les
cellules des deux notebooks en sont extraites telles quelles, ce qui garantit des
noms de variables identiques partout.
- TP_regression_lineaire_ELEVE.ipynb : sans sorties, avec les trous `___` et des
  cellules « Je prédis » / « Je constate » à remplir.
- TP_regression_lineaire_CORRIGE.ipynb : trous complétés, une cellule « Réponse »
  par étape. Avec --executer, il est exécuté de bout en bout (nbconvert, ExecutePreprocessor) et ses
  sorties sont enregistrées. Pour cela, le CSV est lu en local (le fichier peut ne
  pas être encore publié sur GitHub), puis l'URL GitHub est remise dans le code.
"""
import re
import sys
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

RACINE = Path(__file__).resolve().parent.parent
TP = RACINE / "tp" / "regression_lineaire"
URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/clean/rayonnement_pf.csv"

TROUS = {
    "X = df[[ ___ ]]": 'X = df[["heures_soleil", "humidite", "pression", "latitude", "ile", "saison"]]',
    "y = df[ ___ ]": 'y = df["rayonnement"]',
}

REPONSES = {
    "1.0": "Le tableau contient **3 000 lignes** (une par journée et par station) et **11 colonnes**. "
           "La cible est `rayonnement`, le rayonnement solaire reçu au sol en J/cm² : c'est la dernière colonne.",
    "1.1": "Avec `[[ ]]`, on donne une *liste* de colonnes et on obtient un tableau à 6 colonnes ; avec `[ ]`, "
           "on prend une seule colonne, qui donne une série. On ne met surtout pas `rayonnement` dans X : "
           "ce serait donner la réponse au modèle. Il « tricherait » et ne servirait à rien pour une journée "
           "dont on ne connaît pas encore le rayonnement.",
    "1.2": "**2 400 lignes** pour l'entraînement (80 %) et **600** pour le test (20 %). Si on évaluait le modèle "
           "sur les journées qu'il a déjà vues, on mesurerait sa mémoire, pas sa capacité à prédire de nouvelles "
           "journées : le score serait trop optimiste.",
    "1.3": "`ile` et `saison` sont du texte. Le modèle ne fait que des calculs (multiplier chaque feature par un "
           "coefficient, puis additionner) : il ne sait pas multiplier « Tahiti » ou « été austral » par un nombre. "
           "Il faut d'abord les transformer en nombres, c'est l'étape 5.",
    "1.4": "Avant normalisation, la pression tourne autour de **1 014 hPa** et les heures de soleil autour de "
           "**6,4 h** : les nombres de la pression sont environ 150 fois plus grands, alors qu'elle varie peu "
           "(écart-type 3,4 hPa). Sans normalisation, la pression pèserait bien plus lourd dans les calculs et "
           "les coefficients ne seraient pas comparables entre eux. Après normalisation, les deux colonnes ont "
           "une moyenne de 0 et un écart-type de 1 : elles partent à égalité.",
    "1.5": "`ile` devient **9 colonnes**, une par île présente dans l'entraînement, et `saison` en devient **2**. "
           "Sur chaque ligne, une seule colonne `ile_…` vaut 1 : celle de l'île où la journée a été mesurée.",
    "1.6": "**15 colonnes** : les 4 colonnes numériques normalisées, plus 9 colonnes d'îles et 2 colonnes de "
           "saison. Il y en a plus que 6 parce que l'encodage remplace chaque colonne de texte par une colonne "
           "par catégorie. Le tableau ne contient plus que des nombres : le modèle peut le lire.",
    "2.1": "Non : le modèle est encore une « boîte vide ». Il n'a vu aucune donnée et n'a pas de coefficients ; "
           "il les apprendra avec `fit`, à l'étape suivante.",
    "2.2": "Le plus grand coefficient est celui de `heures_soleil` : **+146 J/cm² par heure de soleil en plus**, "
           "toutes les autres features restant identiques. C'est très proche des 145 du cours. Le coefficient "
           "de l'humidité est **négatif** (−56,5) : à soleil égal, un air plus humide (brume, nuages fins) laisse "
           "passer moins de rayonnement, c'est logique. On voit aussi que l'été austral apporte environ "
           "390 J/cm² de plus que l'hiver (194,3 − (−194,3)).",
    "2.3": "Le R² vaut au maximum **1** (prédiction parfaite) ; **0** veut dire « pas mieux que la moyenne », et il "
           "peut même être négatif si le modèle fait pire. Un R² de 0,65 voudrait dire que le modèle explique 65 % "
           "des variations du rayonnement ; ici on obtient **0,74**, soit 74 %. Le modèle se trompe en moyenne de "
           "**255 J/cm²**, contre **538** pour la baseline : l'erreur est divisée par deux, le modèle a donc bien "
           "appris quelque chose.",
    "2.4": "Si le modèle était parfait, tous les points seraient **exactement sur la diagonale** orange "
           "(prédit = réel). Ici, ils s'alignent bien le long de la diagonale, avec une certaine épaisseur : "
           "c'est l'erreur moyenne de 255 J/cm². Aux extrémités, le modèle « tire vers la moyenne » : "
           "il sous-estime les journées les plus fortes (aucune prédiction au-delà de 2 900 J/cm², pour des "
           "journées réelles jusqu'à 3 500) et surestime les plus faibles.",
    "2.5": "**Tahiti en décembre** reçoit le plus (environ 2 656 J/cm², soit 7,4 kWh/m²) : c'est la journée la plus "
           "ensoleillée (11 h), en été austral. **Rapa en juillet** reçoit le moins (environ 1 358 J/cm², soit 3,8 kWh/m²) : peu de "
           "soleil (4 h), en hiver austral, sur l'île la plus au sud. Bora Bora est inconnue du modèle : grâce à "
           "`handle_unknown=\"ignore\"`, toutes ses colonnes `ile_…` valent 0, et le modèle s'appuie sur les autres "
           "features (soleil, latitude, saison).",
    "2.6": "Le R² baisse à peine (**0,739 → 0,733**) : l'humidité et la pression n'apportent presque rien de plus "
           "que la durée d'ensoleillement, la saison et l'île. Plus de colonnes n'est donc pas toujours mieux : "
           "un modèle plus simple, qui demande moins de mesures, peut être presque aussi bon. Des colonnes "
           "inutiles peuvent même faire baisser le score sur le test.",
}


def decouper(texte):
    """README -> liste de cellules (type, contenu, id d'étape ou None)."""
    debut_tp1 = texte.index("## TP 1")
    debut_fin = texte.index("## Ce qu'il faut rendre")
    cellules = [("md", texte[:debut_tp1].strip(), None)]
    for num, bloc in enumerate(re.split(r"(?m)^(?=## TP 2)", texte[debut_tp1:debut_fin]), 1):
        parties = re.split(r"(?m)^(?=### )", bloc)
        cellules.append(("md", parties[0].strip().strip("-").strip(), None))
        for etape in parties[1:]:
            numero = re.match(r"### Étape (\d+)", etape).group(1)
            ide = f"{num}.{numero}"
            m = re.search(r"```python\n(.*?)```", etape, re.S)
            avant, code, apres = etape[:m.start()], m.group(1).rstrip(), etape[m.end():]
            cellules.append(("md", avant.strip(), None))
            cellules.append(("code", code, ide))
            cellules.append(("md", apres.strip().strip("-").strip(), ide))
    cellules.append(("md", texte[debut_fin:].strip(), None))
    return [c for c in cellules if c[1]]


def notebook(cellules, corrige):
    nb = nbf.v4.new_notebook()
    titre = "CORRIGÉ" if corrige else "Notebook élève"
    nb.cells.append(nbf.v4.new_markdown_cell(f"**{titre}**"))
    for i, (genre, contenu, ide) in enumerate(cellules):
        a_questions = ide in REPONSES
        if genre == "code":
            if not corrige and a_questions:
                nb.cells.append(nbf.v4.new_markdown_cell("**Je prédis :** …"))
            if corrige:
                for trou, rempli in TROUS.items():
                    contenu = contenu.replace(trou, rempli)
                contenu = contenu.replace("# Complétez : les noms de colonnes entre guillemets, "
                                          "séparés par des virgules\n", "")
            nb.cells.append(nbf.v4.new_code_cell(contenu))
        else:
            nb.cells.append(nbf.v4.new_markdown_cell(contenu))
            if ide and a_questions:  # cellule « Ce que vous devez obtenir » d'une étape
                nb.cells.append(nbf.v4.new_markdown_cell(
                    f"**Réponse.** {REPONSES[ide]}" if corrige else "**Je constate :** …"))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def executer(chemin):
    """Exécute le corrigé avec le CSV local, puis remet l'URL GitHub dans le code."""
    local = str(RACINE / "data" / "clean" / "rayonnement_pf.csv")
    nb = nbf.read(chemin, as_version=4)
    for c in nb.cells:
        c.source = c.source.replace(URL, local)
    # API de nbconvert (ExecutePreprocessor) plutôt que la ligne de commande : elle
    # n'est pas perturbée par les fichiers de configuration Jupyter de la machine
    ExecutePreprocessor(timeout=300, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(TP)}})
    for c in nb.cells:
        c.source = c.source.replace(local, URL)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, chemin)


if __name__ == "__main__":
    cellules = decouper((TP / "README.md").read_text())
    assert set(REPONSES) == {c[2] for c in cellules if c[0] == "code"} - {"2.0"}, "réponses manquantes"
    corrige = TP / "TP_regression_lineaire_CORRIGE.ipynb"
    nbf.write(notebook(cellules, corrige=True), corrige)
    nbf.write(notebook(cellules, corrige=False), TP / "TP_regression_lineaire_ELEVE.ipynb")
    if "--executer" in sys.argv:
        executer(corrige)
    print("Notebooks écrits dans", TP)
