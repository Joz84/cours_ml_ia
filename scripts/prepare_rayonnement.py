"""Prépare data/clean/rayonnement_pf.csv pour le TP de régression linéaire.

Usage : python scripts/prepare_rayonnement.py

Source : Météo-France, « Données climatologiques de base - quotidiennes »
(data.gouv.fr, Licence Ouverte v2.0), département 987 (Polynésie française),
fichier « autres paramètres » 1950-2024. Le fichier brut est téléchargé dans
data/raw/meteofrance/ s'il n'y est pas déjà.

Le fichier produit contient environ 3 000 journées réelles (2000-2024), réparties
à parts égales entre les stations, sans aucune valeur manquante.
"""
from pathlib import Path

import pandas as pd
import requests

RACINE = Path(__file__).resolve().parent.parent
RAW = RACINE / "data" / "raw" / "meteofrance"
CLEAN = RACINE / "data" / "clean"
NOM = "Q_987_previous-1950-2024_autres-parametres.csv.gz"
URL = "https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/QUOT/" + NOM

# Station Météo-France -> île (déduit du nom de la station)
ILE = {
    "FAAA": "Tahiti", "AFAAHITI 2": "Tahiti", "MOOREA-AERO": "Moorea",
    "HIVA-OA": "Hiva Oa", "TAKAROA": "Takaroa", "ANAA1": "Anaa", "HAO AERO": "Hao",
    "TUBUAI AERO": "Tubuai", "MATAURA 1": "Tubuai",  # Mataura est un village de Tubuai
    "MANGAREVA": "Mangareva", "RAPA": "Rapa",
}
PAR_STATION = 300  # journées tirées au hasard dans chaque station


def telecharger():
    RAW.mkdir(parents=True, exist_ok=True)
    if not (RAW / NOM).exists():
        r = requests.get(URL, timeout=300)
        r.raise_for_status()  # en cas d'échec : on s'arrête, on n'invente rien
        (RAW / NOM).write_bytes(r.content)
    return RAW / NOM


def preparer(chemin):
    d = pd.read_csv(chemin, sep=";", low_memory=False,
                    usecols=["NOM_USUEL", "LAT", "LON", "AAAAMMJJ", "GLOT", "INST", "UM", "PMERM"])
    d["date"] = pd.to_datetime(d["AAAAMMJJ"].astype(str), format="%Y%m%d")
    d = d[d["date"].dt.year.between(2000, 2024)]

    # Lignes exploitables : rayonnement et insolation renseignés, rayonnement plausible,
    # et pas de case vide dans les autres colonnes gardées
    d = d.dropna(subset=["GLOT", "INST", "UM", "PMERM"])
    d = d[(d["GLOT"] > 0) & (d["GLOT"] <= 4000)]
    inconnues = set(d["NOM_USUEL"]) - set(ILE)
    assert not inconnues, f"Stations sans île : {inconnues}"

    # Stations ayant assez de journées pour l'échantillon (AFAAHITI 2 n'en a aucune complète)
    n = d["NOM_USUEL"].value_counts()
    d = d[d["NOM_USUEL"].isin(n[n >= PAR_STATION].index)]
    d = d.groupby("NOM_USUEL").sample(PAR_STATION, random_state=42)

    # Une latitude / longitude par île (moyenne de ses stations). Tubuai a deux stations
    # distantes de 3 km : avec deux latitudes presque égales pour une même île, la
    # régression devient instable (latitude et île disent presque la même chose).
    lat = d.groupby(d["NOM_USUEL"].map(ILE))["LAT"].transform("mean")
    lon = d.groupby(d["NOM_USUEL"].map(ILE))["LON"].transform("mean")

    out = pd.DataFrame({
        "date": d["date"].dt.date,
        "station": d["NOM_USUEL"],
        "ile": d["NOM_USUEL"].map(ILE),
        "latitude": lat.round(2),
        "longitude": lon.round(2),
        "heures_soleil": (d["INST"] / 60).round(1),  # INST est en minutes
        "humidite": d["UM"].round(0).astype(int),     # humidité moyenne, %
        "pression": d["PMERM"],                       # pression moyenne au niveau de la mer, hPa
        "mois": d["date"].dt.month,
        "saison": d["date"].dt.month.map(lambda m: "été austral" if m in (11, 12, 1, 2, 3, 4)
                                         else "hiver austral"),
        "rayonnement": d["GLOT"].astype(int),         # rayonnement global, J/cm² : la cible
    })
    out = out.sort_values(["date", "station"]).reset_index(drop=True)
    assert out.notna().all().all()
    return out


if __name__ == "__main__":
    df = preparer(telecharger())
    CLEAN.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN / "rayonnement_pf.csv", index=False)
    print(f"{len(df)} lignes x {df.shape[1]} colonnes -> data/clean/rayonnement_pf.csv")
    print(df.groupby(["ile", "station"]).agg(jours=("date", "size"), debut=("date", "min"),
                                             fin=("date", "max")).to_string())
    print(df.describe().round(1).to_string())
