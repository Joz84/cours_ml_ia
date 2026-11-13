"""Nettoie chaque fichier de data/sources/ pour la visualisation -> data/sources_propres/.

Usage : python clean_sources.py   (après make_sources.py)

Une ligne reste une observation (un modèle, un répondant, une cible…). Le nettoyage :
libellés traduits et harmonisés, cases à valeurs multiples dédoublonnées, code pays
ISO-3 ajouté pour les cartes, colonnes oui/non en vrai/faux. Rien n'est inventé :
une valeur inconnue reste vide.
"""
from pathlib import Path

import pandas as pd

from prepare_data import (ALGO_FR, DOMAINE_FR, EPOCH_ISO3, EUROSTAT_ISO3, FREINS_FR,
                          NACE_FR, ODD_FR, PAYS_FR, SECTEUR_FR, TECH_FR)

SRC = Path(__file__).parent / "data" / "sources"
OUT = Path(__file__).parent / "data" / "sources_propres"

PAYS_FR = {**PAYS_FR,
           "DZA": "Algérie", "BGD": "Bangladesh", "BLR": "Biélorussie", "BRA": "Brésil",
           "CMR": "Cameroun", "CHL": "Chili", "COL": "Colombie", "ECU": "Équateur",
           "EGY": "Égypte", "ETH": "Éthiopie", "GHA": "Ghana", "IRQ": "Irak",
           "KAZ": "Kazakhstan", "KEN": "Kenya", "MAR": "Maroc", "NPL": "Népal",
           "NGA": "Nigeria", "PAK": "Pakistan", "PER": "Pérou", "PHL": "Philippines",
           "ZAF": "Afrique du Sud", "LKA": "Sri Lanka", "TUN": "Tunisie", "UGA": "Ouganda",
           "ZWE": "Zimbabwe"}


def liste(s):
    """« A,B,A » -> ["A", "B"] (ordre conservé, doublons retirés)."""
    if pd.isna(s):
        return []
    return list(dict.fromkeys(x.strip() for x in str(s).split(",") if x.strip()))


# ---------------------------------------------------------------- Epoch AI
TYPE_ORGA_FR = {"Industry": "Entreprise", "Academia": "Université",
                "Government": "Organisme public", "Research collective": "Collectif de recherche"}
ACCES_FR = {
    "Open weights (unrestricted)": "Ouvert (poids téléchargeables)",
    "Open weights (restricted use)": "Ouvert (poids téléchargeables)",
    "Open weights (non-commercial)": "Ouvert (poids téléchargeables)",
    "API access": "Accessible en ligne (API ou service)",
    "Hosted access (no API)": "Accessible en ligne (API ou service)",
    "Unreleased": "Non publié ou accès limité",
    "Limited access": "Non publié ou accès limité",
}


def epoch():
    d = pd.read_csv(SRC / "epoch_modeles.csv")
    pays = d["pays_organisation"].map(liste)
    dom = d["domaine"].map(liste)
    types = d["type_organisation"].map(liste)
    for valeurs, connus, nom in [(pays, EPOCH_ISO3, "pays"), (dom, DOMAINE_FR, "domaines"),
                                 (types, TYPE_ORGA_FR, "types d'organisation"),
                                 (d["acces"].dropna().map(lambda x: [x]), ACCES_FR, "accès")]:
        inconnus = set(valeurs.explode().dropna()) - set(connus)
        assert not inconnus, f"Epoch, {nom} sans traduction : {inconnus}"

    def pays_fr(p):
        return "Multinationale" if p == "Multinational" else PAYS_FR[EPOCH_ISO3[p]]

    date = pd.to_datetime(d["date_publication"])
    out = pd.DataFrame({
        "modele": d["modele"],
        "organisation": d["organisation"],
        # 1er pays cité ; les autres sont dans pays_tous
        "pays": pays.map(lambda l: pays_fr(l[0]) if l else None),
        "code_pays": pays.map(lambda l: EPOCH_ISO3[l[0]] if l else None),
        "nb_pays": pays.str.len(),
        "pays_tous": pays.map(lambda l: " ; ".join(map(pays_fr, l)) or None),
        "type_organisation": types.map(
            lambda l: None if not l else TYPE_ORGA_FR[l[0]] if len(l) == 1 else "Plusieurs types"),
        "date_publication": date.dt.date,
        "annee": date.dt.year,
        "mois": date.dt.month,
        "domaine": dom.map(lambda l: DOMAINE_FR[l[0]] if l else None),
        "nb_domaines": dom.str.len(),
        "domaines_tous": dom.map(lambda l: " ; ".join(DOMAINE_FR[x] for x in l) or None),
        "tache": d["tache"],
        "nb_parametres": d["nb_parametres"],
        "calcul_entrainement_flop": d["calcul_entrainement_flop"],
        "acces": d["acces"].map(ACCES_FR),
        "notable": d["notable"],
        "annee_partielle": date.dt.year == 2026,  # données arrêtées au 12/11/2026
    })
    for c in ["nb_pays", "nb_domaines"]:  # 0 = case vide à l'origine
        out[c] = out[c].where(out[c] > 0).astype("Int64")
    return out.sort_values("date_publication")


# ---------------------------------------------------------------- Eurostat
def eurostat():
    e = pd.read_csv(SRC / "eurostat_entreprises_ia.csv", keep_default_na=False, na_values=[""])
    indic = {"E_AI_TANY": ("Usage", "Utilise au moins une technologie d'IA"),
             **{c: ("Technologie", l) for c, l in TECH_FR.items()},
             **{c: ("Frein", l) for c, l in FREINS_FR.items()}}
    e = e[e["code_indicateur"].isin(indic) & e["code_secteur"].isin(NACE_FR)
          & e["code_pays"].isin(list(EUROSTAT_ISO3) + ["EU27_2020"])]
    iso3 = {**EUROSTAT_ISO3, "EU27_2020": None}
    out = pd.DataFrame({
        "code_pays": e["code_pays"].map(iso3),
        "pays": e["code_pays"].map(lambda c: "UE à 27" if c == "EU27_2020" else PAYS_FR[iso3[c]]),
        "est_pays": e["code_pays"] != "EU27_2020",
        "secteur": e["code_secteur"].map(NACE_FR),
        "theme": e["code_indicateur"].map(lambda c: indic[c][0]),
        "indicateur": e["code_indicateur"].map(lambda c: indic[c][1]),
        "annee": e["annee"],
        "pct_entreprises": e["pct_entreprises"],
    })
    return out.sort_values(["theme", "indicateur", "secteur", "pays", "annee"])


# ---------------------------------------------------------------- Kaggle
KAGGLE_ISO3 = {
    "Algeria": "DZA", "Argentina": "ARG", "Australia": "AUS", "Austria": "AUT",
    "Bangladesh": "BGD", "Belarus": "BLR", "Belgium": "BEL", "Brazil": "BRA",
    "Cameroon": "CMR", "Canada": "CAN", "Chile": "CHL", "China": "CHN",
    "People 's Republic of China": "CHN", "Colombia": "COL", "Czech Republic": "CZE",
    "Denmark": "DNK", "Ecuador": "ECU", "Egypt": "EGY", "Ethiopia": "ETH", "Finland": "FIN",
    "France": "FRA", "Germany": "DEU", "Ghana": "GHA", "Greece": "GRC", "Hong Kong": "HKG",
    "Hong Kong (S.A.R.)": "HKG", "Hungary": "HUN", "India": "IND", "Indonesia": "IDN",
    "Iran": "IRN", "Iran, Islamic Republic of...": "IRN", "Iraq": "IRQ", "Ireland": "IRL",
    "Israel": "ISR", "Italy": "ITA", "Japan": "JPN", "Kazakhstan": "KAZ", "Kenya": "KEN",
    "Malaysia": "MYS", "Mexico": "MEX", "Morocco": "MAR", "Nepal": "NPL",
    "Netherlands": "NLD", "New Zealand": "NZL", "Nigeria": "NGA", "Norway": "NOR",
    "Pakistan": "PAK", "Peru": "PER", "Philippines": "PHL", "Poland": "POL",
    "Portugal": "PRT", "Republic of Korea": "KOR", "South Korea": "KOR", "Romania": "ROU",
    "Russia": "RUS", "Saudi Arabia": "SAU", "Singapore": "SGP", "South Africa": "ZAF",
    "Spain": "ESP", "Sri Lanka": "LKA", "Sweden": "SWE", "Switzerland": "CHE",
    "Taiwan": "TWN", "Thailand": "THA", "Tunisia": "TUN", "Turkey": "TUR", "Uganda": "UGA",
    "Ukraine": "UKR", "United Arab Emirates": "ARE", "United Kingdom": "GBR",
    "United Kingdom of Great Britain and Northern Ireland": "GBR", "United States": "USA",
    "United States of America": "USA", "Viet Nam": "VNM", "Vietnam": "VNM",
    "Zimbabwe": "ZWE",
}
PAYS_SANS_CODE = {
    "Other": "Autre pays",
    "I do not wish to disclose my location": "Non communiqué",
    # 2017 : proposé à côté de « Taiwan » et de « People's Republic of China » -> ambigu
    "Republic of China": "Chine ou Taïwan (ambigu)",
}
METIER_FR = {
    "Data Scientist": "Data scientist",
    "Software Developer/Software Engineer": "Développement logiciel",
    "Software Engineer": "Développement logiciel", "Programmer": "Développement logiciel",
    "Data Analyst": "Analyse de données", "Business Analyst": "Analyse de données",
    "Marketing Analyst": "Analyse de données",
    "Data Analyst (Business, Marketing, Financial, Quantitative, etc)": "Analyse de données",
    "Scientist/Researcher": "Recherche", "Researcher": "Recherche",
    "Research Scientist": "Recherche", "Research Assistant": "Recherche",
    "Principal Investigator": "Recherche", "Computer Scientist": "Recherche",
    "Machine Learning Engineer": "Ingénierie machine learning",
    "Machine Learning/ MLops Engineer": "Ingénierie machine learning",
    "Data Engineer": "Ingénierie des données", "DBA/Database Engineer": "Ingénierie des données",
    "Data Architect": "Ingénierie des données", "Data Administrator": "Ingénierie des données",
    "Data Miner": "Ingénierie des données",
    "Statistician": "Statistique", "Predictive Modeler": "Statistique",
    "Operations Research Practitioner": "Statistique",
    "Student": "Études", "Not employed": "Sans emploi", "Currently not employed": "Sans emploi",
    "Manager": "Management, gestion de projet", "Chief Officer": "Management, gestion de projet",
    "Product/Project Manager": "Management, gestion de projet",
    "Program/Project Manager": "Management, gestion de projet",
    "Product Manager": "Management, gestion de projet",
    "Manager (Program, Project, Operations, Executive-level, etc)": "Management, gestion de projet",
    "Teacher / professor": "Enseignement",
    "Engineer": "Autre ingénierie", "Engineer (non-software)": "Autre ingénierie",
    "Consultant": "Autre", "Developer Advocate": "Autre",
    "Developer Relations/Advocacy": "Autre", "Salesperson": "Autre",
    "Data Journalist": "Autre", "Other": "Autre",
}
# une colonne vrai/faux par famille d'algorithmes (libellés harmonisés de prepare_data)
ALGO_COL = {
    "Régression linéaire ou logistique": "algo_regression",
    "Arbres de décision / forêts aléatoires": "algo_arbres_forets",
    "Gradient boosting (XGBoost…)": "algo_gradient_boosting",
    "Méthodes bayésiennes": "algo_bayesien",
    "Algorithmes évolutionnaires": "algo_evolutionnaire",
    "Réseaux de neurones « classiques »": "algo_reseaux_neurones",
    "Réseaux convolutifs (images)": "algo_cnn_images",
    "Réseaux génératifs adverses (GAN)": "algo_gan",
    "Réseaux récurrents (séquences)": "algo_rnn_sequences",
    "Transformers (BERT, GPT…)": "algo_transformers",
    "Auto-encodeurs": "algo_autoencodeurs",
    "Réseaux de neurones sur graphes": "algo_graphes",
}
AGES = [18, 22, 25, 30, 35, 40, 45, 50, 55, 60, 70, 200]
AGES_LIB = ["18-21", "22-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
            "55-59", "60-69", "70+"]


def kaggle():
    k = pd.read_csv(SRC / "kaggle_repondants.csv", low_memory=False)
    for col, connus in [("pays", {**KAGGLE_ISO3, **PAYS_SANS_CODE}), ("metier", METIER_FR)]:
        inconnus = set(k[col].dropna()) - set(connus)
        assert not inconnus, f"Kaggle, {col} sans correspondance : {inconnus}"

    # âge : 2017 = âge exact -> tranches des années suivantes ; 70-79 et 80+ (2018) -> 70+
    age = k["age"].replace({"70-79": "70+", "80+": "70+"})
    exact = pd.to_numeric(k["age"], errors="coerce").where(k["annee"] == 2017)
    exact = exact.where(exact.between(18, 99))  # âges < 18 ou > 99 en 2017 : jugés non plausibles
    age = age.where(k["annee"] != 2017, pd.cut(exact, AGES, right=False, labels=AGES_LIB).astype(str))
    age = age.replace("nan", pd.NA)

    algos = k["algorithmes"].map(
        lambda s: None if pd.isna(s) else list(dict.fromkeys(
            ALGO_FR[x.strip()] for x in s.split(";") if x.strip() in ALGO_FR)))
    out = pd.DataFrame({
        "annee": k["annee"], "id_repondant": k["id_repondant"],
        "pays": k["pays"].map(lambda p: PAYS_FR.get(KAGGLE_ISO3.get(p)) or PAYS_SANS_CODE.get(p)),
        "code_pays": k["pays"].map(KAGGLE_ISO3),
        "tranche_age": age,
        "metier": k["metier"].map(METIER_FR),
        # « I am a student » (2018) n'est pas un secteur -> vide
        "secteur": k["secteur"].map(SECTEUR_FR),
        "algorithmes": algos.map(lambda l: None if l is None else " ; ".join(l) or None),
        "nb_algorithmes": algos.map(lambda l: None if l is None else len(l)).astype("Int64"),
    })
    for lib, col in ALGO_COL.items():
        # vide si le répondant n'a pas répondu à la question (ou question absente cette année)
        out[col] = algos.map(lambda l: None if l is None else lib in l).astype("boolean")
    return out


# ---------------------------------------------------------------- OWID
def owid():
    o = pd.read_csv(SRC / "owid_incidents.csv")
    return pd.DataFrame({"annee": o["annee"], "nb_incidents": o["nb_incidents_controverses"],
                         "nb_incidents_cumules": o["nb_incidents_controverses"].cumsum()})


# ---------------------------------------------------------------- Vinuesa
def vinuesa():
    v = pd.read_csv(SRC / "vinuesa_cibles_odd.csv", dtype={"cible": str})
    lev, fre = v["levier"].astype(bool), v["frein"].astype(bool)
    effet = pd.Series("Aucun effet documenté", index=v.index)
    effet[lev & ~fre] = "Levier seulement"
    effet[~lev & fre] = "Frein seulement"
    effet[lev & fre] = "Levier et frein"
    return pd.DataFrame({"odd": v["odd"], "objectif": v["odd"].map(ODD_FR),
                         "cible": v["cible"], "texte": v["texte"].str.strip(),
                         "levier": lev, "frein": fre, "effet": effet})


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for nom, f in {"epoch_modeles": epoch, "eurostat_entreprises_ia": eurostat,
                   "kaggle_repondants": kaggle, "owid_incidents": owid,
                   "vinuesa_cibles_odd": vinuesa}.items():
        df = f()
        chemin = OUT / f"{nom}_propre.csv"
        df.to_csv(chemin, index=False)
        print(f"{chemin.name} : {df.shape[0]} lignes x {df.shape[1]} colonnes, "
              f"{chemin.stat().st_size / 1e6:.1f} Mo")
