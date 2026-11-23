"""Fabrique le jeu SALE du TP 1 (nettoyage) et le tableau LARGE du geste 4.

Usage : python scripts/make_releves_brut.py

Données SIMULÉES (seed 42, résultat identique à chaque exécution) : relevés
journaliers de l'année 2024 pour 5 stations FICTIVES de Tahiti. Le climat imite
celui de Faa'a (Météo-France) : pluie mensuelle moyenne 1991-2020 de Faa'a pour la
station `Port`, multipliée pour les autres stations ; températures qui baissent
avec l'altitude (environ -0,65 °C par 100 m).

| station | altitude | pluie (x Port) | relevés                         |
|---------|----------|----------------|---------------------------------|
| Port    |    5 m   | x 1,0          | toute l'année                   |
| Lagon   |   10 m   | x 0,9          | toute l'année                   |
| Vallée  |  150 m   | x 2,7          | toute l'année                   |
| Plateau |  450 m   | x 1,8          | toute l'année                   |
| Col     |  900 m   | x 2,2          | novembre à avril SEULEMENT (*)  |

(*) le piège des n du TP 2 : Col n'a que des relevés de saison des pluies.

Fichiers produits :
- data/raw/releves_meteo_brut.csv : séparateur « ; », colonnes `date`, `station`,
  `altitude_m`, `pluie_mm`, `temp_min_C`, `temp_max_C`, `code_agent`, `remarque` ;
- data/raw/pluie_mensuelle_large.csv : pluie totale par station et par mois, au
  format LARGE (une colonne par mois), calculée sur les valeurs vraies.

Défauts injectés, un par geste du cours (les nombres exacts sont affichés à
l'exécution et vérifiés par des assert) :
- geste 2 (typer) : les dates sont écrites sous trois formats (`03/01/2024`,
  `2024-01-03`, `3 janv. 24`) ; environ 30 % des nombres ont une virgule
  décimale (`12,5`) ; 6 pluies sont écrites avec leur unité (`48 mm`) ; une
  température maximale est écrite en lettres (`vingt-deux`) ;
- geste 3 (trier) : deux colonnes inutiles, `code_agent` et `remarque` ;
- geste 5 (trous) : 4 % de cases vides dans `pluie_mm` et dans `temp_min_C`,
  dont un quart écrites `NA` et un quart codées `-999` ;
- geste 6 (doublons) : 40 lignes recopiées à l'identique ;
- geste 7 (harmoniser) : `Port` écrit aussi `port`, `PORT ` et ` Port` ;
  `Vallée` écrit aussi `Vallee` (sans accent) ;
- geste 8 (impossibles et incohérentes) : 3 températures impossibles (270 et 99
  en maximale, -5 en minimale), 2 pluies très fortes mais plausibles (180 et
  210 mm, à Port), 5 lignes où `temp_min_C` est plus grande que `temp_max_C`.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
RAW = RACINE / "data" / "raw"
SEED = 42

# Pluie mensuelle moyenne à Faa'a, 1991-2020 (mm), calculée à partir des données
# quotidiennes Météo-France (département 987, station FAAA)
PLUIE_FAAA = {1: 268, 2: 244, 3: 167, 4: 99, 5: 104, 6: 75, 7: 44, 8: 41, 9: 50, 10: 103, 11: 105, 12: 279}
STATIONS = {"Port": (5, 1.0), "Lagon": (10, 0.9), "Vallée": (150, 2.7), "Plateau": (450, 1.8), "Col": (900, 2.2)}
MOIS_FR = {1: "janv.", 2: "févr.", 3: "mars", 4: "avr.", 5: "mai", 6: "juin", 7: "juil.", 8: "août",
           9: "sept.", 10: "oct.", 11: "nov.", 12: "déc."}
MOIS_LARGE = ["janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout", "septembre",
              "octobre", "novembre", "decembre"]


def releves_vrais(rng):
    """Les valeurs « vraies », sans aucun défaut."""
    lignes = []
    for station, (altitude, facteur) in STATIONS.items():
        for jour in pd.date_range("2024-01-01", "2024-12-31"):
            if station == "Col" and jour.month in (5, 6, 7, 8, 9, 10):
                continue
            saison_pluies = jour.month in (11, 12, 1, 2, 3, 4)
            p_pluie = 0.50 if saison_pluies else 0.28
            moyenne_jour_pluie = PLUIE_FAAA[jour.month] * facteur / (jour.days_in_month * p_pluie)
            pluie = rng.gamma(1.1, moyenne_jour_pluie / 1.1) if rng.uniform() < p_pluie else 0.0
            saison_t = 1.2 * np.cos(2 * np.pi * (jour.dayofyear - 45) / 366)  # plus chaud en février
            tmin = 23.5 - 0.0065 * altitude + saison_t + rng.normal(0, 0.8)
            tmax = tmin + 6.8 + saison_t * 0.5 - 0.02 * min(pluie, 60) + rng.normal(0, 1.0)
            lignes.append({"date": jour, "station": station, "altitude_m": altitude,
                           "pluie_mm": round(pluie, 1), "temp_min_C": round(tmin, 1),
                           "temp_max_C": round(tmax, 1)})
    return pd.DataFrame(lignes)


def tableau_large(vrai):
    m = vrai.assign(mois=vrai["date"].dt.month).pivot_table(
        index="station", columns="mois", values="pluie_mm", aggfunc="sum")
    m = m.round(1)
    # mois sans aucun relevé (Col de mai à octobre) : case vide, pas zéro
    n = vrai.assign(mois=vrai["date"].dt.month).pivot_table(
        index="station", columns="mois", values="pluie_mm", aggfunc="size")
    m = m.where(n > 0)
    m.columns = MOIS_LARGE
    return m.reindex(list(STATIONS)).reset_index()


def salir(vrai, rng):
    """Injecte un défaut par geste ; renvoie le tableau sale (texte) et le décompte."""
    n = len(vrai)
    libres = list(rng.permutation(n))
    tirer = lambda k: [libres.pop() for _ in range(k)]  # lignes jamais touchées deux fois
    d = vrai.copy()
    compte = {}

    # Geste 8 : impossibles, très fortes mais plausibles, incohérentes
    i = tirer(3)
    d.loc[i[0], "temp_max_C"], d.loc[i[1], "temp_max_C"], d.loc[i[2], "temp_min_C"] = 270.0, 99.0, -5.0
    port_sec = [k for k in libres if d.loc[k, "station"] == "Port" and d.loc[k, "date"].month in (6, 7, 8)][:2]
    for k, v in zip(port_sec, [180.0, 210.0]):
        libres.remove(k)
        d.loc[k, "pluie_mm"] = v
    i = tirer(5)
    d.loc[i, ["temp_min_C", "temp_max_C"]] = d.loc[i, ["temp_max_C", "temp_min_C"]].values
    compte["impossibles"], compte["fortes_plausibles"], compte["min_sup_max"] = 3, 2, 5

    # Tout passe en texte, comme dans un fichier mal saisi
    t = pd.DataFrame({"station": d["station"].astype(str), "altitude_m": d["altitude_m"].astype(str)})
    for col in ["pluie_mm", "temp_min_C", "temp_max_C"]:
        t[col] = d[col].map(lambda v: f"{v:.1f}")

    # Geste 5 : trous (vides, « NA », « -999 »)
    for col in ["pluie_mm", "temp_min_C"]:
        i = tirer(round(0.04 * n))
        q = len(i) // 4
        t.loc[i[:2 * q], col] = ""
        t.loc[i[2 * q:3 * q], col] = "NA"
        t.loc[i[3 * q:], col] = "-999"
        compte[f"trous_{col}"] = (len(i), 2 * q, q, len(i) - 3 * q)

    # Geste 2 : virgules, unités, nombre en lettres
    for col in ["pluie_mm", "temp_min_C", "temp_max_C"]:
        ok = t[col].str.match(r"^\d+\.\d$")
        virgule = ok & (rng.uniform(size=n) < 0.3)
        t.loc[virgule, col] = t.loc[virgule, col].str.replace(".", ",", regex=False)
        compte[f"virgules_{col}"] = int(virgule.sum())
    i = [k for k in tirer(60) if t.loc[k, "pluie_mm"] not in ("", "NA", "-999") and float(t.loc[k, "pluie_mm"].replace(",", ".")) > 5][:6]
    t.loc[i, "pluie_mm"] = t.loc[i, "pluie_mm"].str.replace(r"[.,]\d$", "", regex=True) + " mm"
    compte["unites_mm"] = len(i)
    i = tirer(1)
    t.loc[i, "temp_max_C"] = "vingt-deux"
    compte["en_lettres"] = 1

    # Geste 2 : trois formats de date
    fmt = rng.integers(0, 3, n)
    t.insert(0, "date", [
        jour.strftime("%d/%m/%Y") if f == 0 else jour.strftime("%Y-%m-%d") if f == 1
        else f"{jour.day} {MOIS_FR[jour.month]} {jour.strftime('%y')}"
        for jour, f in zip(d["date"], fmt)])
    compte["formats_date"] = np.bincount(fmt).tolist()

    # Geste 7 : écritures différentes d'une même station
    port = [k for k in libres if t.loc[k, "station"] == "Port"]
    variantes = rng.choice(["port", "PORT ", " Port"], size=60)
    for k, v in zip(port[:60], variantes):
        t.loc[k, "station"] = v
    vallee = [k for k in libres if t.loc[k, "station"] == "Vallée"][:12]
    t.loc[vallee, "station"] = "Vallee"
    compte["variantes_port"], compte["vallee_sans_accent"] = 60, 12

    # Geste 3 : deux colonnes inutiles
    t["code_agent"] = [f"AG{c:02d}" for c in rng.integers(1, 9, n)]
    t["remarque"] = rng.choice(["", "", "", "", "", "", "RAS", "capteur nettoyé", "relevé tardif"], n)

    # Geste 6 : 40 doublons exacts, glissés au hasard dans le tableau
    doublons = t.iloc[rng.choice(n, 40, replace=False)]
    t = pd.concat([t, doublons]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    compte["doublons"] = 40
    return t, compte


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    vrai = releves_vrais(rng)
    sale, compte = salir(vrai, rng)
    RAW.mkdir(parents=True, exist_ok=True)
    sale.to_csv(RAW / "releves_meteo_brut.csv", sep=";", index=False)
    tableau_large(vrai).to_csv(RAW / "pluie_mensuelle_large.csv", index=False)

    assert len(sale) == len(vrai) + 40
    assert (sale["temp_max_C"] == "vingt-deux").sum() == 1
    print(f"{len(vrai)} relevés vrais + 40 doublons = {len(sale)} lignes -> data/raw/releves_meteo_brut.csv")
    print("Relevés par station :", vrai["station"].value_counts().to_dict())
    print("Pluie annuelle vraie par station (mm) :",
          vrai.groupby("station")["pluie_mm"].sum().round(0).to_dict())
    print("Part de jours sans pluie :", round((vrai["pluie_mm"] == 0).mean() * 100, 1), "%")
    for k, v in compte.items():
        print(f"  {k} : {v}")
