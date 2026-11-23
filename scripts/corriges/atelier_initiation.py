"""Corrigé de l'atelier d'initiation « L'électricité de Tahiti » (utilisé par scripts/build_tp_donnees.py)."""
import re
import shutil
from pathlib import Path

NOTEBOOK = "Atelier_initiation"
FICHIERS = ["data/clean/electricite_tahiti.csv", "tp/atelier_initiation/quiz.py"]

README = (Path(__file__).resolve().parent.parent.parent / "tp" / "atelier_initiation" / "README.md").read_text()
BONNES = {1: "B", 2: "B", 3: "C", 4: "C", 5: "C", 6: "A", 7: "C", 8: "C", 9: "A"}


def bloc(ide, *remplacements):
    code = re.search(rf"<!-- id: {ide} -->\s*\n```python\n(.*?)```", README, re.S).group(1).rstrip()
    for avant, apres in remplacements:
        assert avant in code, (ide, avant)
        code = code.replace(avant, apres)
    return code


TROUS = {f"q{n}": bloc(f"q{n}", (f'verifier({n}, "")', f'verifier({n}, "{lettre}")')) for n, lettre in BONNES.items()}
TROUS["a_vous_colonne"] = bloc("a_vous_colonne", ('colonne = "total_MW"', 'colonne = "hydraulique_MW"'))
TROUS["libre"] = '''# Question : le solaire produit-il plus en décembre ou en juin ?
solaire_par_mois = df[df["annee"] == 2025].groupby("mois")["solaire_MW"].mean()
plt.figure(figsize=(9, 4))
plt.bar(solaire_par_mois.index, solaire_par_mois.values, color="#f2b701")
plt.title("En 2025, le solaire produit moins en juin (8,7 MW) qu'en décembre (12,4 MW)")
plt.xlabel("mois"); plt.ylabel("puissance solaire moyenne (MW)")
plt.xticks(range(1, 13))
plt.show()
print(solaire_par_mois.round(1))'''

REPONSES = {
    "Mission 10": "Exemple de réponse : « En 2025, le solaire de Tahiti a produit en moyenne 8,7 MW en juin, contre 12,4 MW en décembre. » En juin, c'est l'hiver austral : les journées sont plus courtes et le soleil plus bas. Attention à un piège : de nouvelles centrales solaires ont été raccordées en cours d'année, ce qui gonfle aussi les derniers mois. Pour comparer les saisons sans ce piège, il faudrait diviser par la puissance installée, ou comparer plusieurs années.",
}


def APRES(racine):
    shutil.rmtree(racine / "tp" / "atelier_initiation" / "__pycache__", ignore_errors=True)
