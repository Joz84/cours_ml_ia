"""Prépare le jeu « électricité de Tahiti » de l'atelier d'initiation.

Usage : python scripts/prepare_energie_tahiti.py

Source : Service des énergies de la Polynésie française (Observatoire polynésien de
l'énergie), « Enregistrements des productions d'électricité sur Tahiti, pas de temps
10 minutes, par source d'énergie », données EDT, 2015 à 2025 :
https://www.service-public.pf/sde/production-delectricite-sur-tahiti-2/
Données utilisées à des fins pédagogiques dans le cadre du partenariat entre
l'Université de la Polynésie française et le Service des énergies.

Le script télécharge les 11 fichiers Excel dans data/raw/energie_tahiti/ (s'ils n'y
sont pas), harmonise leurs colonnes et leurs unités, puis écrit :
- data/clean/electricite_tahiti.csv : une ligne par heure, de 2015 à 2025, avec la
  puissance moyenne de chaque source sur l'heure (en MW, donc aussi l'énergie en MWh
  produite pendant l'heure).

Ce que fait l'harmonisation (aucune valeur n'est inventée) :
- chaque fichier a ses propres noms de colonnes ; on les ramène à trois sources :
  thermique, hydraulique, solaire ;
- les fichiers 2018-2019, 2024 et 2025 sont en kW, les autres en MW : tout passe en MW ;
- solaire = solaire « fatal » (centrales sans stockage) + à partir de 2024 les petites
  centrales Fare Gouwe et Fare Meri + en 2025 les centrales solaires avec stockage
  (« P stocké », qui suit exactement la courbe du soleil) ;
- la colonne « Putu Uira » (2023 à 2025, 0,03 MW en moyenne, moins de 0,05 % du total)
  est laissée de côté ;
- pour 2018, on prend le fichier 2018-2019 (en kW, avec la puissance totale fournie)
  plutôt que le fichier 2018 seul ; les deux concordent ;
- un pas de 10 minutes où la puissance totale est nulle (panne d'enregistrement, comme
  le 20/01/2022 de 00 h 50 à 06 h 40, signalé dans le fichier) devient une case vide ;
  de même quand thermique et hydraulique valent zéro pile pendant que le solaire produit
  (le 29/06/2025 vers midi) : en coupure générale, le solaire décroche aussi, c'est donc
  un trou d'enregistrement, pas une coupure ;
- les pas de 10 minutes sont moyennés par heure ; une heure sans aucun pas valide reste
  vide.
"""
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
BRUT = RACINE / "data" / "raw" / "energie_tahiti"
SORTIE = RACINE / "data" / "clean" / "electricite_tahiti.csv"
BASE = "https://www.service-public.pf/sde/wp-content/uploads/sites/15/"
JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

# fichier, feuille, ligne d'en-tête, unité, colonnes : date, thermique, hydraulique, [colonnes solaires]
FICHIERS = [
    ("2019/05/03-EDT-données-production-2015-TAHITI.xlsx", 0, 0, "MW",
     "date", "puissance_therm", "puissance_hydro", ["puissance_sol"]),
    ("2019/05/04-EDT-données-production-2016-TAHITI.xlsx", 0, 0, "MW",
     "date", "puissance_therm", "puissance_hydro", ["puissance_sol"]),
    ("2019/05/05-EDT-données-production-2017-TAHITI.xlsx", 0, 0, "MW",
     "date", "puissance_therm", "puissance_hydro", ["puissance_sol"]),
    ("2020/05/EDT-données-production-2018-19-TAHITI.xlsx", "2018", 0, "kW",
     "Date-Heure", "Puissance Active Brute Totale Thermique moyennée 10 min",
     "Puissance active hydro totale moyennée sur 10 min", ["Puissance active solaire totale moyennée 10 min"]),
    ("2020/05/EDT-données-production-2018-19-TAHITI.xlsx", "2019", 0, "kW",
     "Date", "Puissance Active Brute Totale Thermique moyennée 10 min",
     "Puissance active hydro totale moyennée sur 10 min", ["Puissance active solaire totale moyennée 10 min"]),
    ("2021/04/donnees-production-2020.xlsx", 0, 0, "MW",
     "Dates", "Total thermique", "Puissance Hydro", ["Puissance PV"]),
    ("2022/05/08-EDT-donnees-production-2021-TAHITI.xlsx", 0, 0, "MW",
     "Dates", "Total thermique", "Puissance Hydro", ["Solaire"]),
    ("2023/01/10-EDT-donnees_Productions_-2022-TAHITI.xlsx", 0, 0, "MW",
     "Dates", "Total thermique MW", "Puissance Hydro MW", ["Solaire MW"]),
    ("2024/08/11-Donnees_Productions_2023.xlsx", 0, 1, "MW",
     "Unnamed: 0", "Pth", "Hydro", ["Solaire"]),
    ("2026/02/Donnees-Production-2024-TAHITI.xlsx", 0, 0, "kW",
     "Date", "Puissance thermique (kW)", "Puissance hydraulique (kW)",
     ["Puissance solaire Fatale (kW)", "P Fare Gouwe (kW)", "Puissance FARE-MERI kW"]),
    ("2026/06/Donnees-Production_2025_Tahiti.xlsx", "Consolidation", 0, "kW",
     "Date-Time", "P thermique (kW)", "P hydro (kW)", ["P fatal (kW)", "P stocke (kW)"]),
]


def telecharger(chemin):
    local = BRUT / Path(chemin).name
    if not local.exists():
        BRUT.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(BASE + urllib.parse.quote(chemin), headers={"User-Agent": "Mozilla/5.0"})
        local.write_bytes(urllib.request.urlopen(req).read())
    return local


def lire(chemin, feuille, entete, unite, c_date, c_therm, c_hydro, c_sol):
    d = pd.read_excel(telecharger(chemin), sheet_name=feuille, header=entete)
    nombre = lambda c: pd.to_numeric(d[c], errors="coerce")
    t = pd.DataFrame({"date": pd.to_datetime(d[c_date], errors="coerce"),
                      "thermique_MW": nombre(c_therm),
                      "hydraulique_MW": nombre(c_hydro),
                      "solaire_MW": sum(nombre(c).fillna(0) for c in c_sol)})
    t.loc[nombre(c_sol[0]).isna(), "solaire_MW"] = np.nan       # pas de mesure solaire : vide, pas zéro
    if unite == "kW":
        t[["thermique_MW", "hydraulique_MW", "solaire_MW"]] /= 1000
    return t.dropna(subset=["date"])


def main():
    morceaux = []
    for f in FICHIERS:
        t = lire(*f)
        annee = t["date"].dt.year.mode()[0]
        t = t[t["date"].dt.year == annee]                     # une ligne de l'an suivant traîne parfois
        morceaux.append(t)
        print(f"{annee} : {len(t):>6} pas de 10 min · {Path(f[0]).name}")
    d = pd.concat(morceaux).drop_duplicates("date").sort_values("date")

    sources = ["thermique_MW", "hydraulique_MW", "solaire_MW"]
    d["total"] = d[sources].sum(axis=1, min_count=3)
    panne = d["total"].fillna(0) <= 1                           # enregistrement manquant : tout à zéro
    # thermique ET hydraulique à zéro pile pendant que le solaire produit : impossible physiquement
    # (en coupure générale, les onduleurs solaires décrochent aussi) -> trou d'enregistrement
    panne |= (d["thermique_MW"] == 0) & (d["hydraulique_MW"] == 0)
    print("Pas de 10 min sans enregistrement (mis en vide) :", int(panne.sum()))
    d.loc[panne, sources] = np.nan
    d[sources] = d[sources].clip(lower=0)                       # quelques -0,00x MW de bruit de mesure

    h = d.set_index("date")[sources].resample("h").mean()
    h["total_MW"] = h[sources].sum(axis=1, min_count=3)
    h = h.round(2).reset_index().rename(columns={"date": "date_heure"})
    h.insert(1, "annee", h["date_heure"].dt.year)
    h.insert(2, "mois", h["date_heure"].dt.month)
    h.insert(3, "jour", h["date_heure"].dt.day)
    h.insert(4, "heure", h["date_heure"].dt.hour)
    h.insert(5, "jour_semaine", h["date_heure"].dt.dayofweek.map(dict(enumerate(JOURS))))
    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    h.to_csv(SORTIE, index=False)

    print(f"\n{len(h)} heures, de {h['date_heure'].min()} à {h['date_heure'].max()} -> {SORTIE.relative_to(RACINE)}")
    print("Heures vides :", int(h["total_MW"].isna().sum()))
    bilan = h.groupby("annee")[sources + ["total_MW"]].sum() / 1000       # MWh -> GWh
    bilan["part_renouvelable_%"] = (100 * (bilan["hydraulique_MW"] + bilan["solaire_MW"]) / bilan["total_MW"]).round(1)
    print("Énergie produite par an (GWh) :")
    print(bilan.round(0).to_string())


if __name__ == "__main__":
    main()
