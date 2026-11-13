"""Télécharge toutes les données brutes du cours dans data/raw/.

Usage : python download_data.py

- Les sources publiques (Epoch AI, Eurostat, Our World in Data, Nature) sont
  téléchargées directement.
- Les Kaggle Surveys nécessitent une clé API (~/.kaggle/kaggle.json) et
  l'acceptation des règles de chaque compétition sur kaggle.com.
  En cas d'échec, le script s'arrête avec un message explicite : aucune donnée
  de remplacement n'est générée.

Un journal (data/raw/download_log.csv) garde la date de téléchargement de chaque
fichier, pour pouvoir citer les sources.
"""
import csv
import datetime as dt
import json
import os
import sys
import zipfile
from pathlib import Path

import requests

RAW = Path(__file__).parent / "data" / "raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (cours-data-science)"}

SOURCES_HTTP = {
    # Epoch AI – base complète et sous-ensemble « notable » (CC BY 4.0)
    "epoch_all_ai_models.csv": "https://epoch.ai/data/all_ai_models.csv",
    "epoch_notable_ai_models.csv": "https://epoch.ai/data/notable_ai_models.csv",
    # Vinuesa et al. 2020, Nature Communications – Supplementary Data 1 (CC BY 4.0)
    "vinuesa2020_supplementary_data_1.xlsx": (
        "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-14108-y/"
        "MediaObjects/41467_2019_14108_MOESM2_ESM.xlsx"
    ),
}

# Our World in Data – incidents et controverses IA (AI Incident Database via AI Index)
OWID_CSV = ("https://ourworldindata.org/grapher/annual-reported-ai-incidents-controversies.csv"
            "?v=1&csvType=full&useColumnShortNames=false")
# Repli : le graphique a été retiré du site OWID (URL en 404, novembre 2026), mais
# l'indicateur sous-jacent (id 1025675, identifié dans la page archivée du graphique)
# reste servi par l'API officielle d'OWID. Ce sont les mêmes chiffres, non modifiés.
OWID_API = "https://api.ourworldindata.org/v1/indicators/1025675"


def download_owid():
    name = "owid_ai_incidents.csv"
    try:
        r = requests.get(OWID_CSV, headers=HEADERS, timeout=60)
        r.raise_for_status()
        (RAW / name).write_bytes(r.content)
        return log(name, OWID_CSV, "OK")
    except Exception as e:
        log(name, OWID_CSV, "ECHEC", f"{e!r} -> repli sur l'API OWID")
    try:
        data = requests.get(OWID_API + ".data.json", timeout=60).json()
        meta = requests.get(OWID_API + ".metadata.json", timeout=60).json()
        ent = {e["id"]: e for e in meta["dimensions"]["entities"]["values"]}
        with open(RAW / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Entity", "Code", "Year", meta["name"]])
            for y, v, e in zip(data["years"], data["values"], data["entities"]):
                w.writerow([ent[e]["name"], ent[e]["code"], y, v])
        (RAW / "owid_ai_incidents_metadata.json").write_text(json.dumps(meta, indent=1))
        log(name, OWID_API + ".data.json", "OK", "via API (repli)")
    except Exception as e:
        log(name, OWID_API, "ECHEC", repr(e))


def download_eurostat():
    """Eurostat isoc_eb_ain2 via le paquet `eurostat` (fichier TSV compressé, léger)."""
    name = "eurostat_isoc_eb_ain2.csv"
    try:
        import eurostat
        df = eurostat.get_data_df("isoc_eb_ain2")
        df.to_csv(RAW / name, index=False)
        # libellés des codes (secteurs NACE, indicateurs) pour les rendre lisibles
        for par in ["nace_r2", "indic_is", "size_emp", "unit", "geo"]:
            labels = eurostat.get_dic("isoc_eb_ain2", par, frmt="df")
            labels.to_csv(RAW / f"eurostat_dic_{par}.csv", index=False)
        log(name, "eurostat.get_data_df('isoc_eb_ain2')", "OK", f"{df.shape}")
    except Exception as e:
        log(name, "eurostat isoc_eb_ain2", "ECHEC", repr(e))

KAGGLE_DATASETS = {2017: "kaggle/kaggle-survey-2017", 2018: "kaggle/kaggle-survey-2018"}
KAGGLE_COMPETITIONS = {y: f"kaggle-survey-{y}" for y in (2019, 2020, 2021, 2022)}

log_rows = []


def log(name, source, status, detail=""):
    log_rows.append({
        "fichier": name, "source": source, "statut": status, "detail": detail,
        "date_telechargement": dt.date.today().isoformat(),
    })
    print(f"[{status}] {name} {detail}")


def download_http():
    for name, url in SOURCES_HTTP.items():
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
            r.raise_for_status()
            (RAW / name).write_bytes(r.content)
            log(name, url, "OK", f"{len(r.content) / 1e6:.2f} Mo")
        except Exception as e:  # on signale, on n'invente rien
            log(name, url, "ECHEC", repr(e))


def download_kaggle():
    """Télécharge les 6 éditions du Kaggle Survey. Retourne la liste des erreurs."""
    errors = []
    kdir = Path.home() / ".kaggle"
    has_key = (
        (kdir / "kaggle.json").exists() or (kdir / "access_token").exists()
        or "KAGGLE_API_TOKEN" in os.environ or "KAGGLE_KEY" in os.environ
    )
    if not has_key:
        return [f"Clé API absente : ni {kdir}/kaggle.json, ni {kdir}/access_token, "
                "ni variable KAGGLE_API_TOKEN."]
    try:
        # NB : le paquet kaggle (v2+) s'authentifie dès l'import
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except (Exception, SystemExit) as e:
        return [f"Authentification Kaggle refusée : {e!r}"]
    for year, ref in {**KAGGLE_DATASETS, **KAGGLE_COMPETITIONS}.items():
        dest = RAW / f"kaggle_{year}"
        dest.mkdir(exist_ok=True)
        # Cas d'un zip téléchargé à la main depuis l'onglet « Data » de Kaggle
        manual = list(dest.glob("*.zip"))
        if manual:
            for z in manual:
                with zipfile.ZipFile(z) as f:
                    f.extractall(dest)
            log(f"kaggle_{year}/", f"{ref} (zip manuel : {manual[0].name})", "OK")
            continue
        if any(dest.glob("*.csv")):
            log(f"kaggle_{year}/", ref, "OK", "déjà présent")
            continue
        try:
            if year in KAGGLE_DATASETS:
                api.dataset_download_files(ref, path=dest, unzip=True, quiet=True)
            else:
                api.competition_download_files(ref, path=dest, quiet=True)
                for z in dest.glob("*.zip"):
                    with zipfile.ZipFile(z) as f:
                        f.extractall(dest)
                    z.unlink()
            log(f"kaggle_{year}/", ref, "OK")
        except Exception as e:
            log(f"kaggle_{year}/", ref, "ECHEC", repr(e))
            errors.append(f"{year} ({ref}) : {e}")
    return errors


def write_log():
    path = RAW / "download_log.csv"
    # on conserve les entrées précédentes des fichiers non retéléchargés
    old = []
    if path.exists():
        current = {r["fichier"] for r in log_rows}
        with open(path) as f:
            old = [r for r in csv.DictReader(f) if r["fichier"] not in current]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        w.writeheader()
        w.writerows(old + log_rows)


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    only_kaggle = "--kaggle-only" in sys.argv
    if not only_kaggle:
        download_http()
        download_owid()
        download_eurostat()
    kaggle_errors = download_kaggle()
    write_log()
    if kaggle_errors:
        print("\n=== Kaggle : téléchargement impossible ===")
        for e in kaggle_errors:
            print(" -", e)
        print(
            "\nÀ faire :\n"
            " 1. https://www.kaggle.com/settings/api > « Generate New Token », puis copier le jeton :\n"
            "    echo LE_JETON > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token\n"
            "    (ou, ancien format : ~/.kaggle/kaggle.json, ou bien la commande « kaggle auth login »)\n"
            " 2. Accepter les règles (bouton « I Understand and Accept » onglet Rules) :\n"
            + "".join(f"    https://www.kaggle.com/competitions/{c}/rules\n"
                      for c in KAGGLE_COMPETITIONS.values())
            + " 3. Relancer : python download_data.py --kaggle-only"
        )
        sys.exit(1)
