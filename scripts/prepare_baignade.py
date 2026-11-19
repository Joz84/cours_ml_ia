"""Prépare data/clean/baignade_taharuu.csv pour le TP de classification.

Usage : python scripts/prepare_baignade.py

Une ligne = un jour et une plage, pour 3 plages de Tahiti, du 1er janvier 2023 au
31 décembre 2024 (731 jours x 3 plages = 2 193 lignes).

Ce qui est RÉEL (Météo-France, « Données climatologiques de base - quotidiennes »,
département 987, fichier RR-T-Vent 1950-2024, data.gouv.fr, Licence Ouverte 2.0) :
- `pluie_24h_mm` : pluie du jour (RR) au pluviomètre le plus proche de la plage :
  PAPARA 7 pour `taharuu`, FAAA pour `vaiava`, PAEA 3 pour `pk18` ;
- `pluie_72h_mm` : somme de la pluie du jour et des deux jours précédents ;
- `temperature_C` : température moyenne de l'air (TM), à PAPARA 7 pour `taharuu`,
  à FAAA pour les deux autres plages.

Ce qui est SIMULÉ (seed fixée, résultat identique à chaque exécution) :
- `debit_m3s` : débit de la rivière qui débouche près de la plage. Débit de base
  propre à chaque rivière + réponse à la pluie des jours précédents (effet qui
  décroît de moitié environ chaque jour) + un bruit de ±15 % ;
- `conforme` (`oui` / `non`) : qualité bactériologique de l'eau. La probabilité
  d'être NON conforme suit une courbe logistique qui monte surtout avec
  `pluie_72h_mm` et `debit_m3s` (le ruissellement et la rivière apportent les
  bactéries), un peu avec `pluie_24h_mm`. `taharuu`, à l'embouchure d'une grande
  rivière, est plus fragile que les deux autres. Le résultat est ensuite tiré au
  sort selon cette probabilité : deux jours identiques peuvent donc avoir des
  résultats différents (c'est le « bruit »). La constante est réglée pour que la
  classe `non` soit rare, entre 12 et 18 % des lignes.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RACINE = Path(__file__).resolve().parent.parent
RAW = RACINE / "data" / "raw" / "meteofrance"
CLEAN = RACINE / "data" / "clean"
NOM = "Q_987_previous-1950-2024_RR-T-Vent.csv.gz"
URL = "https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/QUOT/" + NOM

# plage : (station de pluie, station de température, débit de base m3/s, réponse à la pluie, fragilité)
PLAGES = {
    "taharuu": ("PAPARA 7", "PAPARA 7", 3.0, 0.060, 0.9),  # embouchure de la Taharu'u
    "vaiava":  ("FAAA",     "FAAA",     0.6, 0.020, 0.0),
    "pk18":    ("PAEA 3",   "FAAA",     0.8, 0.025, 0.0),
}
SEED = 42


def telecharger():
    RAW.mkdir(parents=True, exist_ok=True)
    if not (RAW / NOM).exists():
        r = requests.get(URL, timeout=300)
        r.raise_for_status()  # en cas d'échec : on s'arrête avec l'erreur
        (RAW / NOM).write_bytes(r.content)
    return RAW / NOM


def preparer(chemin):
    rng = np.random.default_rng(SEED)
    m = pd.read_csv(chemin, sep=";", usecols=["NOM_USUEL", "AAAAMMJJ", "RR", "TM"], low_memory=False)
    m = m[m["AAAAMMJJ"] >= 20221201]  # décembre 2022 : pour calculer la pluie sur 72 h dès le 1er janvier
    m["date"] = pd.to_datetime(m["AAAAMMJJ"].astype(str), format="%Y%m%d")

    lignes = []
    for plage, (st_pluie, st_temp, debit_base, reponse, _) in PLAGES.items():
        pluie = m[m["NOM_USUEL"] == st_pluie].set_index("date")["RR"].asfreq("D")
        temp = m[m["NOM_USUEL"] == st_temp].set_index("date")["TM"].asfreq("D")
        d = pd.DataFrame({"pluie_24h_mm": pluie, "temperature_C": temp})
        d["pluie_72h_mm"] = d["pluie_24h_mm"].rolling(3).sum()
        # débit : réponse à la pluie qui s'atténue de moitié environ chaque jour
        memoire = d["pluie_24h_mm"].ewm(halflife=1.0, adjust=False).mean()
        d["debit_m3s"] = (debit_base + reponse * memoire * 10) * rng.uniform(0.85, 1.15, len(d))
        d["plage"] = plage
        lignes.append(d.loc["2023-01-01":"2024-12-31"].reset_index())
    d = pd.concat(lignes, ignore_index=True)
    assert d[["pluie_24h_mm", "pluie_72h_mm", "temperature_C"]].notna().all().all(), "mesure manquante"

    # Probabilité d'être NON conforme (courbe logistique), puis tirage au sort
    z = lambda s: (s - s.mean()) / s.std()
    fragilite = d["plage"].map({p: v[4] for p, v in PLAGES.items()})
    logit = (-2.95 + 1.3 * z(d["pluie_72h_mm"]) + 0.9 * z(np.log1p(d["debit_m3s"]))
             + 0.4 * z(d["pluie_24h_mm"]) + fragilite)
    proba_non = 1 / (1 + np.exp(-logit))
    non = rng.uniform(size=len(d)) < proba_non

    out = pd.DataFrame({
        "date": d["date"].dt.date,
        "plage": d["plage"],
        "pluie_24h_mm": d["pluie_24h_mm"].round(1),
        "pluie_72h_mm": d["pluie_72h_mm"].round(1),
        "temperature_C": d["temperature_C"].round(1),
        "debit_m3s": d["debit_m3s"].round(2),
        "saison": d["date"].dt.month.map(lambda mo: "ete_austral" if mo in (11, 12, 1, 2, 3, 4)
                                         else "hiver_austral"),
        "conforme": np.where(non, "non", "oui"),
    })
    return out.sort_values(["date", "plage"]).reset_index(drop=True)


if __name__ == "__main__":
    df = preparer(telecharger())
    part_non = (df["conforme"] == "non").mean()
    assert 0.12 <= part_non <= 0.18, f"part de « non » hors de la cible : {part_non:.1%}"
    assert df.notna().all().all()
    CLEAN.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN / "baignade_taharuu.csv", index=False)
    print(f"{len(df)} lignes -> data/clean/baignade_taharuu.csv ; non conformes : {part_non:.1%}")
    print(df.groupby("plage")["conforme"].apply(lambda s: f"{(s == 'non').mean():.1%} de non").to_string())
    print(df.describe().round(1).to_string())
