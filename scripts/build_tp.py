"""Génère les notebooks ÉLÈVE et CORRIGÉ d'un TP à partir de son README.

Usage : python scripts/build_tp.py classification|clustering [--executer] [--brouillon]

Le README (tp/<nom>/README.md) est la seule source du code : les cellules des deux
notebooks en sont extraites telles quelles, ce qui garantit des noms de variables
identiques partout. Le contenu propre au corrigé (cellules à trous complétées,
réponses rédigées) est dans scripts/corriges/<nom>.py.

- TP_<nom>_ELEVE.ipynb : sans sorties ; les cellules à trous gardent leur
  commentaire « # À compléter » ; une cellule « Votre réponse : » sous chaque étape
  ou question qui comporte des **Questions**.
- TP_<nom>_CORRIGE.ipynb : trous complétés, une cellule « Réponse » à la place de
  chaque « Votre réponse ». Avec --executer, il est exécuté de bout en bout
  (nbconvert, ExecutePreprocessor) et ses sorties sont enregistrées. Le CSV est
  alors lu en local (il peut ne pas être encore publié sur GitHub), puis l'URL
  GitHub est remise dans le code.
- --brouillon : autorise des réponses manquantes (pendant la rédaction).

Le TP de régression linéaire a son propre script (build_tp_regression.py).
"""
import importlib.util
import re
import sys
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

RACINE = Path(__file__).resolve().parent.parent
DEPOT = "Joz84/cours_ml_ia"
URL_BRUTE = f"https://raw.githubusercontent.com/{DEPOT}/main/data/clean/"


def charger_corrige(nom):
    chemin = RACINE / "scripts" / "corriges" / f"{nom}.py"
    spec = importlib.util.spec_from_file_location(f"corrige_{nom}", chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decouper(texte):
    """README -> liste de cellules (type, contenu, id d'étape ou None, a_questions)."""
    debut_tp1 = texte.index("## TP 1")
    fins = [texte.index(t) for t in ("## Réutiliser", "## Ce qu'il faut rendre") if t in texte]
    debut_fin = min(fins)
    cellules = [("md", texte[:debut_tp1].strip(), None, False)]
    for num, bloc in enumerate(re.split(r"(?m)^(?=## TP 2)", texte[debut_tp1:debut_fin]), 1):
        parties = re.split(r"(?m)^(?=### )", bloc)
        cellules.append(("md", parties[0].strip().strip("-").strip(), None, False))
        for etape in parties[1:]:
            m_id = re.match(r"### (?:Étape (\d+)|Q(\d+))", etape)
            ide = f"{num}.{m_id.group(1)}" if m_id.group(1) else f"{num}.Q{m_id.group(2)}"
            questions = "**Questions**" in etape
            m = re.search(r"```python\n(.*?)```", etape, re.S)
            if m is None:  # question sans code : texte, puis réponse
                cellules.append(("md", etape.strip().strip("-").strip(), ide, questions))
                continue
            cellules.append(("md", etape[:m.start()].strip(), None, False))
            cellules.append(("code", m.group(1).rstrip(), ide, False))
            cellules.append(("md", etape[m.end():].strip().strip("-").strip(), ide, questions))
    cellules.append(("md", texte[debut_fin:].strip(), None, False))
    return [c for c in cellules if c[1]]


def notebook(nom, cellules, corrige, cfg):
    nb = nbf.v4.new_notebook()
    fichier = f"TP_{nom}_{'CORRIGE' if corrige else 'ELEVE'}.ipynb"
    badge = (f"[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
             f"(https://colab.research.google.com/github/{DEPOT}/blob/main/tp/{nom}/{fichier})")
    nb.cells.append(nbf.v4.new_markdown_cell(f"{badge}\n\n**{'CORRIGÉ' if corrige else 'Notebook élève'}**"))
    for genre, contenu, ide, questions in cellules:
        if genre == "code":
            if corrige and ide in cfg.TROUS:
                contenu = cfg.TROUS[ide]
            nb.cells.append(nbf.v4.new_code_cell(contenu))
            continue
        if ide is None and contenu.startswith("# "):  # en-tête : le badge est déjà en haut
            contenu = re.sub(r"\n\[!\[Ouvrir dans Colab\][^\n]*\n", "\n", contenu, count=1)
        nb.cells.append(nbf.v4.new_markdown_cell(contenu))
        if questions:
            reponse = cfg.REPONSES.get(ide, "*(réponse à rédiger)*")
            nb.cells.append(nbf.v4.new_markdown_cell(
                f"**Réponse.** {reponse}" if corrige else "**Votre réponse :** …"))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def executer(chemin, csv):
    """Exécute le corrigé avec le CSV local, puis remet l'URL GitHub dans le code."""
    url, local = URL_BRUTE + csv, str(RACINE / "data" / "clean" / csv)
    nb = nbf.read(chemin, as_version=4)
    for c in nb.cells:
        c.source = c.source.replace(url, local)
    # API de nbconvert plutôt que la ligne de commande : elle n'est pas perturbée
    # par les fichiers de configuration Jupyter installés sur la machine
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(chemin.parent)}})
    for c in nb.cells:
        c.source = c.source.replace(local, url)
    nbf.write(nb, chemin)


if __name__ == "__main__":
    nom = sys.argv[1]
    cfg = charger_corrige(nom)
    dossier = RACINE / "tp" / nom
    cellules = decouper((dossier / "README.md").read_text())

    attendues = {c[2] for c in cellules if c[3]}
    manquantes = attendues - set(cfg.REPONSES)
    if manquantes and "--brouillon" not in sys.argv:
        sys.exit(f"Réponses manquantes : {sorted(manquantes)}")
    assert set(cfg.TROUS) <= {c[2] for c in cellules if c[0] == "code"}, "trou inconnu"

    corrige = dossier / f"TP_{nom}_CORRIGE.ipynb"
    nbf.write(notebook(nom, cellules, True, cfg), corrige)
    nbf.write(notebook(nom, cellules, False, cfg), dossier / f"TP_{nom}_ELEVE.ipynb")
    if "--executer" in sys.argv:
        executer(corrige, cfg.CSV)
    print("Notebooks écrits dans", dossier)
