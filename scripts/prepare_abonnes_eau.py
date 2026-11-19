"""Prépare data/clean/abonnes_eau.csv pour le TP de clustering.

Usage : python scripts/prepare_abonnes_eau.py

Données entièrement SIMULÉES (seed fixée, résultat identique à chaque exécution).
Une ligne = un abonné au service de l'eau, dans 3 communes de Tahiti.

Trois profils cachés, bien séparés :
- petits foyers (45 %) : 1 à 3 personnes, petite consommation, presque pas de jardin ;
- familles avec jardin (35 %) : 3 à 7 personnes, forte hausse en été (arrosage),
  une bonne part d'eau consommée la nuit (arrosage automatique), grand jardin ;
- pensions et commerces (20 %) : `nb_personnes = 0` (aucun résident déclaré),
  très forte consommation, peu d'eau la nuit, petit jardin. Ils sont plus
  nombreux à Punaauia (zone touristique).
Environ 5 % d'abonnés « ambigus » ont des valeurs à mi-chemin entre deux profils.

Le profil d'origine N'EST PAS écrit dans le CSV : le TP consiste à le retrouver.
La surface de jardin (en centaines de m²) a une échelle bien plus grande que les
autres colonnes : sans normalisation, c'est elle qui décide seule des groupes.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
CLEAN = RACINE / "data" / "clean"
SEED = 42
N = 600


def profil(rng, nom, n):
    """Tire n abonnés d'un profil."""
    if nom == "petit_foyer":
        return pd.DataFrame({
            "nb_personnes": rng.choice([1, 2, 3], n, p=[0.4, 0.45, 0.15]),
            "conso_moy_m3_mois": rng.normal(9, 2.5, n),
            "pic_ete_pct": rng.normal(10, 5, n),
            "part_nuit_pct": rng.normal(12, 4, n),
            "surface_jardin_m2": rng.exponential(25, n),
            "commune": rng.choice(["Papara", "Paea", "Punaauia"], n, p=[0.3, 0.35, 0.35]),
        })
    if nom == "famille_jardin":
        return pd.DataFrame({
            "nb_personnes": rng.choice([3, 4, 5, 6, 7], n, p=[0.15, 0.35, 0.3, 0.15, 0.05]),
            "conso_moy_m3_mois": rng.normal(26, 5, n),
            "pic_ete_pct": rng.normal(45, 9, n),
            "part_nuit_pct": rng.normal(32, 6, n),
            "surface_jardin_m2": rng.normal(550, 180, n),
            "commune": rng.choice(["Papara", "Paea", "Punaauia"], n, p=[0.5, 0.3, 0.2]),
        })
    return pd.DataFrame({  # pension ou commerce
        "nb_personnes": np.zeros(n, dtype=int),
        "conso_moy_m3_mois": rng.normal(58, 14, n),
        "pic_ete_pct": rng.normal(22, 7, n),
        "part_nuit_pct": rng.normal(5, 2, n),
        "surface_jardin_m2": rng.exponential(60, n),
        "commune": rng.choice(["Papara", "Paea", "Punaauia"], n, p=[0.15, 0.2, 0.65]),
    })


def preparer():
    rng = np.random.default_rng(SEED)
    effectifs = {"petit_foyer": 256, "famille_jardin": 200, "pension_commerce": 114}
    df = pd.concat([profil(rng, nom, n) for nom, n in effectifs.items()], ignore_index=True)

    # 30 abonnés ambigus : à mi-chemin entre deux profils tirés au hasard
    a = profil(rng, "petit_foyer", 30)
    b = profil(rng, "famille_jardin", 30)
    c = profil(rng, "pension_commerce", 30)
    autre = np.where(rng.uniform(size=30) < 0.5, 1, 2)
    ambigus = a.copy()
    for col in ["conso_moy_m3_mois", "pic_ete_pct", "part_nuit_pct", "surface_jardin_m2"]:
        ambigus[col] = np.where(autre == 1, (a[col] + b[col]) / 2, (a[col] + c[col]) / 2)
    ambigus["nb_personnes"] = np.where(autre == 1, (a["nb_personnes"] + b["nb_personnes"]) // 2,
                                       a["nb_personnes"])
    df = pd.concat([df, ambigus], ignore_index=True)

    # Mise en forme : valeurs plausibles, arrondies ; ordre des abonnés mélangé
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    out = pd.DataFrame({
        "abonne": [f"AB{i:04d}" for i in range(1, len(df) + 1)],
        "commune": df["commune"],
        "conso_moy_m3_mois": df["conso_moy_m3_mois"].clip(lower=2).round(1),
        "pic_ete_pct": df["pic_ete_pct"].clip(lower=0).round(0).astype(int),
        "part_nuit_pct": df["part_nuit_pct"].clip(lower=1, upper=60).round(0).astype(int),
        "nb_personnes": df["nb_personnes"].astype(int),
        "surface_jardin_m2": df["surface_jardin_m2"].clip(lower=0).round(0).astype(int),
    })
    assert len(out) == N and out.notna().all().all()
    return out


if __name__ == "__main__":
    df = preparer()
    CLEAN.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN / "abonnes_eau.csv", index=False)
    print(f"{len(df)} abonnés -> data/clean/abonnes_eau.csv")
    print(df.describe().round(1).to_string())
