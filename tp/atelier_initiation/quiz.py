"""Les QCM de l'atelier d'initiation « L'électricité de Tahiti » (UE 5.7, UPF).

    verifier(1, "B")   corrige la question 1 et compte les points
    score()            affiche le score

Les bonnes réponses sont codées pour ne pas sauter aux yeux : le jeu, c'est de parier
avant de regarder les données !
"""
import base64
import json

_R = json.loads(base64.b64decode(
    "eyIxIjogIkIiLCAiMiI6ICJCIiwgIjMiOiAiQyIsICI0IjogIkMiLCAiNSI6ICJDIiwgIjYiOiAiQSIsICI3IjogIkMiLCAiOCI6ICJDIiwgIjkiOiAiQSJ9"))

_EXPLICATIONS = {
    "1": "11 ans × 365 jours × 24 heures, plus les 29 février : 96 432 heures. Une ligne = une heure.",
    "2": "Tahiti consomme le plus vers 11 h (74,5 MW en moyenne), sans doute parce que bureaux, commerces, "
         "hôtels et leur climatisation tournent à plein. Le creux est vers 3 h du matin (46 MW).",
    "3": "Le thermique (des moteurs qui brûlent du fioul) fournit environ 60 % de l'électricité de 2015 à 2025, "
         "l'hydraulique 32 %, le solaire 8 %.",
    "4": "Le 8 octobre 2025 à 11 h, le soleil a fourni 55,4 MW sur 84 : 66 % de l'électricité de Tahiti, "
         "le record de la période.",
    "5": "26,7 GWh en 2015, 99,2 GWh en 2025 : 3,7 fois plus, dont un bond de 54 à 99 GWh entre 2024 et 2025.",
    "6": "Les barrages produisent le plus en saison des pluies (décembre : 25 MW en moyenne) et le moins en "
         "saison sèche (août : 15 MW). Pas de pluie, pas d'eau dans les rivières.",
    "7": "Le dimanche, Tahiti consomme 53,8 MW en moyenne, contre 64,6 MW en semaine : environ 17 % de moins. "
         "Bureaux et commerces sont fermés.",
    "8": "En avril 2020, en plein confinement, la consommation moyenne est tombée à 60,4 MW, contre 66,2 MW "
         "en avril 2019 : environ 9 % de moins.",
    "9": "Vers 2 h du matin, le réseau tombe à 3 MW, puis remonte en deux heures : 23 MW, puis 54 MW. Toutes "
         "les sources chutent ensemble et repartent l'une après l'autre : c'est la signature d'une coupure "
         "générale suivie d'un rétablissement progressif, pas d'une erreur de saisie.",
}

_trouvees = set()


def verifier(numero, reponse):
    """Corrige une question du QCM : verifier(1, "B")."""
    cle = str(numero)
    if cle not in _R:
        print(f"Il n'y a pas de question {numero} : les questions vont de 1 à {len(_R)}.")
        return
    reponse = str(reponse).strip().upper()
    if reponse not in ("A", "B", "C", "D"):
        print("Écrivez votre lettre (A, B, C ou D) entre les guillemets, puis réexécutez la cellule.")
        return
    if reponse == _R[cle]:
        nouveau = cle not in _trouvees
        _trouvees.add(cle)
        print(f"✅ Bravo, c'est {reponse} !" + (" +1 point." if nouveau else ""))
        print("   " + _EXPLICATIONS[cle])
    else:
        print(f"❌ Pas {reponse}… Regardez bien le graphique, changez votre lettre et réessayez.")
    print(f"   Score : {len(_trouvees)} / {len(_R)}")


def score():
    """Affiche le score."""
    n = len(_trouvees)
    print(f"Score final : {n} / {len(_R)}")
    if n == len(_R):
        print("🏆 Sans faute ! Vous savez déjà faire parler un tableau de 96 432 lignes.")
    elif n >= 6:
        print("👏 Très bien ! Revenez sur les questions manquées : les graphiques ont la réponse.")
    else:
        print("💪 C'est un début ! Relancez les cellules et regardez les graphiques de près.")
