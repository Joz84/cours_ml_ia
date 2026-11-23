"""Génère les notebooks ÉLÈVE et CORRIGÉ des TP « données propres, données lues ».

Usage : python scripts/build_tp_donnees.py nettoyage|analyse_exploratoire|visualisation|classification_images [--executer]

Le README (tp/<nom>/README.md) est la seule source du code : chaque bloc
```python``` devient une cellule de code, le texte qui le précède devient une
cellule Markdown. Une étape (titre « ### … ») peut contenir plusieurs cellules de
code, par exemple les options A, B et C d'une décision.

- Un bloc précédé de `<!-- id: nom -->` peut être remplacé dans le CORRIGÉ par le
  code de scripts/corriges/<nom_du_tp>.py (TROUS) : décisions prises, valeurs
  tapées par l'étudiant, graphiques à produire.
- Une étape qui contient « **Questions** » reçoit à la fin une cellule
  « **Votre réponse :** » (ÉLÈVE) ou « **Réponse.** … » (CORRIGÉ, REPONSES).
- Avec --executer, le CORRIGÉ est exécuté de bout en bout (nbconvert,
  ExecutePreprocessor) dans le noyau « colab_like » (pandas 2.2, comme Colab), les
  URL GitHub étant remplacées par les fichiers locaux le temps de l'exécution.
  La fonction APRES du corrigé est ensuite appelée (par ex. pour ranger le CSV
  propre produit par le TP 1 dans data/clean/).
"""
import importlib.util
import os
import re
import sys
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

RACINE = Path(__file__).resolve().parent.parent
DEPOT = "Joz84/cours_ml_ia"
BRUT = f"https://raw.githubusercontent.com/{DEPOT}/main/"
NOYAU = "colab_like"


def charger_corrige(nom):
    chemin = RACINE / "scripts" / "corriges" / f"{nom}.py"
    spec = importlib.util.spec_from_file_location(f"corrige_{nom}", chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decouper(texte):
    """README -> liste de cellules (type, contenu, id du bloc, id de l'étape à questions)."""
    debut = texte.index("## Le carnet")
    fin = texte.index("## Ce qu'il faut rendre")
    entete = re.sub(r"\n\[!\[Ouvrir dans Colab\][^\n]*\n", "\n", texte[:debut], count=1)
    cellules = [("md", entete.strip().strip("-").strip(), None, None)]
    parties = re.split(r"(?m)^(?=### |## )", texte[debut:fin])
    for partie in parties:
        if not partie.startswith("### "):
            cellules.append(("md", partie.strip().strip("-").strip(), None, None))
            continue
        etape = re.match(r"### ([^·\n]+?)\s*·", partie).group(1).strip()
        position = 0
        for m in re.finditer(r"(?:<!-- id: (\w+) -->\s*\n)?```python\n(.*?)```", partie, re.S):
            avant = partie[position:m.start()]
            cellules.append(("md", avant.strip(), None, None))
            cellules.append(("code", m.group(2).rstrip(), m.group(1), None))
            position = m.end()
        questions = etape if "**Questions**" in partie else None
        cellules.append(("md", partie[position:].strip().strip("-").strip(), None, questions))
    cellules.append(("md", texte[fin:].strip(), None, None))
    return [c for c in cellules if c[1] or c[3]]


def notebook(nom, cellules, corrige, cfg):
    nb = nbf.v4.new_notebook()
    fichier = f"{cfg.NOTEBOOK}_{'CORRIGE' if corrige else 'ELEVE'}.ipynb"
    badge = (f"[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
             f"(https://colab.research.google.com/github/{DEPOT}/blob/main/tp/{nom}/{fichier})")
    nb.cells.append(nbf.v4.new_markdown_cell(f"{badge}\n\n**{'CORRIGÉ' if corrige else 'Notebook élève'}**"))
    for genre, contenu, ide, questions in cellules:
        if genre == "code":
            if corrige and ide in cfg.TROUS:
                contenu = cfg.TROUS[ide]
            nb.cells.append(nbf.v4.new_code_cell(contenu))
            continue
        if contenu:
            nb.cells.append(nbf.v4.new_markdown_cell(contenu))
        if questions:
            reponse = cfg.REPONSES.get(questions, "*(réponse à rédiger)*")
            nb.cells.append(nbf.v4.new_markdown_cell(
                f"**Réponse.** {reponse}" if corrige else "**Votre réponse :** …"))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def remplacer(nb, correspondances):
    for c in nb.cells:
        for avant, apres in correspondances.items():
            c.source = c.source.replace(avant, apres)


def executer(chemin, cfg):
    """Exécute le corrigé avec les fichiers locaux, puis remet les URL GitHub dans le code."""
    locales = {BRUT + rel: str(RACINE / rel) for rel in cfg.FICHIERS}
    nb = nbf.read(chemin, as_version=4)
    remplacer(nb, locales)
    # comme dans Colab : les graphiques s'affichent dans le carnet
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    ExecutePreprocessor(timeout=getattr(cfg, "TIMEOUT", 900), kernel_name=NOYAU).preprocess(
        nb, {"metadata": {"path": str(chemin.parent)}})
    remplacer(nb, {v: k for k, v in locales.items()})
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, chemin)


if __name__ == "__main__":
    nom = sys.argv[1]
    cfg = charger_corrige(nom)
    dossier = RACINE / "tp" / nom
    cellules = decouper((dossier / "README.md").read_text())

    attendues = {c[3] for c in cellules if c[3]}
    manquantes = attendues - set(cfg.REPONSES)
    if manquantes and "--brouillon" not in sys.argv:
        sys.exit(f"Réponses manquantes : {sorted(manquantes)}")
    inconnus = set(cfg.TROUS) - {c[2] for c in cellules if c[0] == "code"}
    assert not inconnus, f"trous inconnus : {inconnus}"

    corrige = dossier / f"{cfg.NOTEBOOK}_CORRIGE.ipynb"
    nbf.write(notebook(nom, cellules, True, cfg), corrige)
    nbf.write(notebook(nom, cellules, False, cfg), dossier / f"{cfg.NOTEBOOK}_ELEVE.ipynb")
    if "--executer" in sys.argv:
        executer(corrige, cfg)
        if hasattr(cfg, "APRES"):
            cfg.APRES(RACINE)
    print("Notebooks écrits dans", dossier)
