"""Produit un CSV simple par source dans data/sources/, à partir de data/raw/.

Usage : python make_sources.py

Contrairement à data/clean/ (chiffres agrégés), ces fichiers gardent les données
brutes, une ligne par observation (un modèle, un répondant, une cible ODD…),
avec seulement les colonnes utiles au cours et des noms de colonnes en français.
Les valeurs ne sont ni traduites ni recodées (ex. réponses Kaggle en anglais).

    epoch_modeles.csv            un modèle d'IA par ligne (Epoch AI)
    eurostat_entreprises_ia.csv  un chiffre par pays × secteur × indicateur × année (Eurostat),
                                 en % des entreprises de 10 salariés ou plus
    eurostat_indicateurs.csv     intitulé de chaque code_indicateur
    kaggle_repondants.csv        un répondant par ligne, 2017-2022 (Kaggle Survey)
    owid_incidents.csv           un an par ligne (OWID / AI Incident Database)
    vinuesa_cibles_odd.csv       une cible ODD par ligne (Vinuesa et al. 2020)
"""
from pathlib import Path

import pandas as pd

RAW = Path(__file__).parent / "data" / "raw"
OUT = Path(__file__).parent / "data" / "sources"


def epoch():
    d = pd.read_csv(RAW / "epoch_all_ai_models.csv", low_memory=False)
    notables = set(pd.read_csv(RAW / "epoch_notable_ai_models.csv", low_memory=False)["Model"])
    cols = {
        "Model": "modele", "Organization": "organisation",
        "Country (of organization)": "pays_organisation",
        "Organization categorization": "type_organisation",
        "Publication date": "date_publication", "Domain": "domaine", "Task": "tache",
        "Parameters": "nb_parametres", "Training compute (FLOP)": "calcul_entrainement_flop",
        "Model accessibility": "acces",
    }
    d = d[list(cols)].rename(columns=cols)
    d.insert(5, "annee", pd.to_datetime(d["date_publication"], errors="coerce").dt.year.astype("Int64"))
    d["notable"] = d["modele"].isin(notables)
    return d.sort_values("date_publication")


def eurostat():
    e = pd.read_csv(RAW / "eurostat_isoc_eb_ain2.csv").rename(columns={"geo\\TIME_PERIOD": "geo"})
    e = e.melt(id_vars=["freq", "size_emp", "nace_r2", "indic_is", "unit", "geo"],
               var_name="annee", value_name="valeur").dropna(subset=["valeur"])
    # on garde l'unité principale : % de l'ensemble des entreprises (10 salariés et plus)
    e = e[e["unit"] == "PC_ENT"]
    lib = {p: pd.read_csv(RAW / f"eurostat_dic_{p}.csv").set_index("val")["descr"]
           for p in ["geo", "nace_r2", "indic_is"]}
    out = pd.DataFrame({
        "code_pays": e["geo"], "pays": e["geo"].map(lib["geo"]),
        "code_secteur": e["nace_r2"], "secteur": e["nace_r2"].map(lib["nace_r2"]),
        "code_indicateur": e["indic_is"], "annee": e["annee"].astype(int),
        "pct_entreprises": e["valeur"],
    })
    # intitulés des indicateurs (longs) dans un fichier à part, pour alléger le CSV
    codes = sorted(out["code_indicateur"].unique())
    pd.DataFrame({"code_indicateur": codes, "indicateur": [lib["indic_is"].get(c) for c in codes]}
                 ).to_csv(OUT / "eurostat_indicateurs.csv", index=False)
    return out.sort_values(["code_indicateur", "code_secteur", "code_pays", "annee"])


# Kaggle : nom de chaque question selon l'année (None = question absente)
KAGGLE = {
    2017: dict(fichier="kaggle_2017/multipleChoiceResponses.csv", pays="Country", age="Age",
               metier="CurrentJobTitleSelect", secteur="EmployerIndustry", algo="WorkAlgorithmsSelect"),
    2018: dict(fichier="kaggle_2018/multipleChoiceResponses.csv", pays="Q3", age="Q2",
               metier="Q6", secteur="Q7", algo=None),
    2019: dict(fichier="kaggle_2019/multiple_choice_responses.csv", pays="Q3", age="Q1",
               metier="Q5", secteur=None, algo=("Q24_Part_",)),
    2020: dict(fichier="kaggle_2020/kaggle_survey_2020_responses.csv", pays="Q3", age="Q1",
               metier="Q5", secteur=None, algo=("Q17_Part_", "Q17_OTHER")),
    2021: dict(fichier="kaggle_2021/kaggle_survey_2021_responses.csv", pays="Q3", age="Q1",
               metier="Q5", secteur="Q20", algo=("Q17_Part_", "Q17_OTHER")),
    2022: dict(fichier="kaggle_2022/kaggle_survey_2022_responses.csv", pays="Q4", age="Q2",
               metier="Q23", secteur="Q24", algo=("Q18_",)),
}


def kaggle():
    out = []
    for annee, q in KAGGLE.items():
        if annee == 2017:
            d = pd.read_csv(RAW / q["fichier"], encoding="latin-1", low_memory=False)
        else:  # 1re ligne sous l'en-tête = texte des questions
            d = pd.read_csv(RAW / q["fichier"], low_memory=False, skiprows=[1])
        r = pd.DataFrame({"annee": annee, "id_repondant": range(1, len(d) + 1)})
        for c in ["pays", "age", "metier", "secteur"]:
            r[c] = d[q[c]].values if q[c] else pd.NA
        if q["algo"] is None:
            r["algorithmes"] = pd.NA
        elif annee == 2017:  # choix séparés par des virgules -> « ; »
            r["algorithmes"] = d[q["algo"]].str.replace(",", " ; ").values
        else:  # une colonne par choix coché -> une seule colonne « a ; b ; c »
            cols = [c for c in d.columns if c.startswith(q["algo"])]
            r["algorithmes"] = d[cols].apply(
                lambda s: " ; ".join(x.strip() for x in s.dropna()) or pd.NA, axis=1).values
        out.append(r)
    return pd.concat(out, ignore_index=True)


def owid():
    o = pd.read_csv(RAW / "owid_ai_incidents.csv")
    return pd.DataFrame({"zone": o["Entity"], "annee": o["Year"],
                         "nb_incidents_controverses": o[o.columns[-1]]})


def vinuesa():
    v = pd.read_excel(RAW / "vinuesa2020_supplementary_data_1.xlsx", header=None, skiprows=3,
                      usecols=range(5), names=["cible", "texte", "levier", "frein", "justification"])
    v["cible"] = v["cible"].astype(str).str.strip()
    # lignes « Goal X: … » = titre de l'objectif, reporté sur ses cibles
    titre = v["cible"].where(v["cible"].str.startswith("Goal")).ffill()
    v["odd"] = titre.str.extract(r"Goal (\d+)")[0].astype(int)
    v["objectif"] = titre.str.replace(r"\s*Main contributors.*", "", regex=True)
    v = v[v["cible"].str.match(r"^\d+\.")].copy()
    # Excel a lu « 8.10 » comme le nombre 8.1 : la 2e occurrence (après 8.9) est la cible X.10
    v.loc[v["cible"].duplicated(), "cible"] = v["cible"].str.replace(r"\.1$", ".10", regex=True)
    v["levier"] = (v["levier"] == 1).astype(int)
    v["frein"] = (v["frein"] == 1).astype(int)
    return v[["odd", "objectif", "cible", "texte", "levier", "frein", "justification"]]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for nom, f in {"epoch_modeles": epoch, "eurostat_entreprises_ia": eurostat,
                   "kaggle_repondants": kaggle, "owid_incidents": owid,
                   "vinuesa_cibles_odd": vinuesa}.items():
        df = f()
        df.to_csv(OUT / f"{nom}.csv", index=False)
        print(f"{nom}.csv : {df.shape[0]} lignes x {df.shape[1]} colonnes, "
              f"{(OUT / f'{nom}.csv').stat().st_size / 1e6:.1f} Mo")
