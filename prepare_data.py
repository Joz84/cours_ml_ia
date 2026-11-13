"""Prépare un fichier unique data/clean/panorama_ia.csv à partir de data/raw/.

Usage : python prepare_data.py

Format « long » (une ligne = un chiffre), colonnes :
    source, indicateur, annee, categorie, sous_categorie, code_pays, valeur, unite, base, note
- `code_pays` : code ISO 3166-1 alpha-3 (ex. FRA), rempli pour les indicateurs « par pays »
- `base` : effectif sur lequel un pourcentage est calculé (répondants, cibles…)
- `note` : précision utile (question absente une année, année partielle…)

Produit aussi data/clean/panorama_ia_large.csv : le même contenu, une ligne par
année et une colonne par « indicateur | catégorie | sous-catégorie ».

Indicateurs produits :
    epoch_modeles_par_an        nb de modèles d'IA publiés par an (tous / notables)
    epoch_domaines              nb de modèles par domaine technique et par an
    eurostat_ia_secteur         % d'entreprises de l'UE utilisant l'IA, par secteur
    eurostat_ia_technologie     % d'entreprises de l'UE utilisant chaque technologie d'IA
    eurostat_ia_freins          raisons de ne pas utiliser l'IA (% des entreprises y ayant pensé)
    eurostat_ia_pays            % d'entreprises utilisant l'IA, par pays européen (carte)
    epoch_pays                  nb de modèles d'IA par pays de l'organisation créatrice (carte)
    kaggle_algorithmes          % des répondants utilisant chaque famille d'algorithmes
    kaggle_secteurs             % des répondants travaillant dans chaque secteur
    owid_incidents              nb d'incidents et controverses IA signalés par an (monde)
    vinuesa_odd                 nb de cibles de chaque ODD où l'IA est un levier / un frein
"""
from pathlib import Path

import pandas as pd

RAW = Path(__file__).parent / "data" / "raw"
CLEAN = Path(__file__).parent / "data" / "clean"
COLS = ["source", "indicateur", "annee", "categorie", "sous_categorie", "code_pays",
        "valeur", "unite", "base", "note"]

# Noms français des pays, indexés par code ISO-3 (pour les cartes)
PAYS_FR = {
    "ALB": "Albanie", "ARE": "Émirats arabes unis", "ARG": "Argentine", "AUS": "Australie",
    "AUT": "Autriche", "BEL": "Belgique", "BGR": "Bulgarie", "BIH": "Bosnie-Herzégovine",
    "CAN": "Canada", "CHE": "Suisse", "CHN": "Chine", "CYP": "Chypre", "CZE": "Tchéquie",
    "DEU": "Allemagne", "DNK": "Danemark", "ESP": "Espagne", "EST": "Estonie",
    "FIN": "Finlande", "FRA": "France", "GBR": "Royaume-Uni", "GRC": "Grèce",
    "HKG": "Hong Kong", "HRV": "Croatie", "HUN": "Hongrie", "IDN": "Indonésie",
    "IND": "Inde", "IRL": "Irlande", "IRN": "Iran", "ISL": "Islande", "ISR": "Israël",
    "ITA": "Italie", "JPN": "Japon", "KOR": "Corée du Sud", "LBN": "Liban",
    "LTU": "Lituanie", "LUX": "Luxembourg", "LVA": "Lettonie", "MAC": "Macao",
    "MEX": "Mexique", "MKD": "Macédoine du Nord", "MLT": "Malte", "MNE": "Monténégro",
    "MYS": "Malaisie", "NLD": "Pays-Bas", "NOR": "Norvège", "NZL": "Nouvelle-Zélande",
    "POL": "Pologne", "PRT": "Portugal", "ROU": "Roumanie", "RUS": "Russie",
    "SAU": "Arabie saoudite", "SGP": "Singapour", "SRB": "Serbie", "SVK": "Slovaquie",
    "SVN": "Slovénie", "SWE": "Suède", "THA": "Thaïlande", "TUR": "Turquie",
    "TWN": "Taïwan", "UKR": "Ukraine", "USA": "États-Unis", "VNM": "Viêt Nam",
}


def rows(df, **const):
    """Ajoute les colonnes constantes et remet les colonnes dans l'ordre standard."""
    df = df.assign(**const)
    for c in COLS:
        if c not in df:
            df[c] = None
    return df[COLS]


# ---------------------------------------------------------------- Epoch AI
DOMAINE_FR = {
    "Language": "Langage (texte)", "Vision": "Vision (images)", "Multimodal": "Multimodal",
    "Biology": "Biologie", "Image generation": "Génération d'images", "Video": "Vidéo",
    "Speech": "Parole", "Robotics": "Robotique", "Audio": "Audio", "Games": "Jeux",
    "Medicine": "Médecine", "Recommendation": "Recommandation", "Other": "Autre",
    "3D modeling": "Modélisation 3D", "Mathematics": "Mathématiques",
    "Materials science": "Science des matériaux", "Earth science": "Sciences de la Terre",
    "Driving": "Conduite autonome", "Cybersecurity": "Cybersécurité",
    "Search": "Moteurs de recherche", "Psychology": "Psychologie", "Astronomy": "Astronomie",
}


def epoch():
    out = []
    for fichier, label in [("epoch_all_ai_models.csv", "Tous les modèles"),
                           ("epoch_notable_ai_models.csv", "Modèles notables")]:
        d = pd.read_csv(RAW / fichier, low_memory=False)
        d["annee"] = pd.to_datetime(d["Publication date"], errors="coerce").dt.year
        n_sans_date = d["annee"].isna().sum()
        d = d.dropna(subset=["annee"])
        par_an = d.groupby("annee").size().rename("valeur").reset_index()
        out.append(rows(par_an, source="Epoch AI", indicateur="epoch_modeles_par_an",
                        categorie=label, unite="modèles",
                        note=f"{n_sans_date} modèles sans date exclus"))
        if label == "Tous les modèles":
            # un modèle peut relever de plusieurs domaines (« Language,Vision ») :
            # il est alors compté dans chacun d'eux
            dom = d.assign(categorie=d["Domain"].str.split(",")).explode("categorie")
            dom["categorie"] = dom["categorie"].str.strip()
            inconnus = set(dom["categorie"].dropna()) - set(DOMAINE_FR)
            assert not inconnus, f"Domaines Epoch à traduire : {inconnus}"
            dom["categorie"] = dom["categorie"].map(DOMAINE_FR)
            dom = dom.dropna(subset=["categorie"])
            dom = dom.groupby(["annee", "categorie"]).size().rename("valeur").reset_index()
            out.append(rows(dom, source="Epoch AI", indicateur="epoch_domaines",
                            unite="modèles",
                            note="un modèle multi-domaines est compté dans chaque domaine"))
    df = pd.concat(out)
    df.loc[df["annee"] == 2026, "note"] = "année partielle (données jusqu'à fin octobre 2026)"
    return df


# Noms de pays tels qu'écrits par Epoch -> code ISO-3
EPOCH_ISO3 = {
    "United States of America": "USA", "China": "CHN", "United Kingdom": "GBR",
    "Canada": "CAN", "France": "FRA", "Germany": "DEU", "South Korea": "KOR",
    "Hong Kong": "HKG", "Switzerland": "CHE", "Japan": "JPN", "Singapore": "SGP",
    "Israel": "ISR", "Russia": "RUS", "Netherlands": "NLD", "Australia": "AUS",
    "India": "IND", "United Arab Emirates": "ARE", "Sweden": "SWE", "Taiwan": "TWN",
    "Spain": "ESP", "Thailand": "THA", "Saudi Arabia": "SAU", "Italy": "ITA",
    "Austria": "AUT", "Poland": "POL", "Denmark": "DNK", "Finland": "FIN",
    "Czechia": "CZE", "Belgium": "BEL", "Vietnam": "VNM", "Lebanon": "LBN",
    "Ukraine": "UKR", "Iran (Islamic Republic of)": "IRN", "Greece": "GRC",
    "Iceland": "ISL", "Indonesia": "IDN", "Croatia": "HRV", "New Zealand": "NZL",
    "Argentina": "ARG", "Ireland": "IRL", "Bulgaria": "BGR", "Luxembourg": "LUX",
    "Macao": "MAC", "Hungary": "HUN", "Portugal": "PRT", "Lithuania": "LTU",
    "Norway": "NOR", "Malaysia": "MYS", "Mexico": "MEX",
    "Multinational": None,  # organisation multinationale : impossible à placer sur la carte
}


def epoch_pays():
    """Nombre de modèles par pays de l'organisation créatrice et par année."""
    d = pd.read_csv(RAW / "epoch_all_ai_models.csv", low_memory=False)
    d["annee"] = pd.to_datetime(d["Publication date"], errors="coerce").dt.year
    d = d.dropna(subset=["annee"])
    # un modèle peut associer plusieurs pays (« United States of America,China ») :
    # il est alors compté dans chacun d'eux
    p = d.assign(pays=d["Country (of organization)"].str.split(",")).explode("pays")
    p["pays"] = p["pays"].str.strip()
    p = p.dropna(subset=["pays"])
    inconnus = set(p["pays"]) - set(EPOCH_ISO3)
    assert not inconnus, f"Pays Epoch sans code ISO-3 : {inconnus}"
    p["code_pays"] = p["pays"].map(EPOCH_ISO3)
    p = p.dropna(subset=["code_pays"])
    p = p.groupby(["annee", "code_pays"]).size().rename("valeur").reset_index()
    p["categorie"] = p["code_pays"].map(PAYS_FR)
    return rows(p, source="Epoch AI", indicateur="epoch_pays", unite="modèles",
                note="pays de l'organisation créatrice ; un modèle multi-pays est compté "
                     "dans chaque pays ; organisations « multinationales » exclues")


# ---------------------------------------------------------------- Eurostat
NACE_FR = {
    "C10-S951_X_K": "Ensemble des secteurs couverts",
    "C": "Industrie manufacturière",
    "D": "Énergie (électricité, gaz)",
    "E": "Eau, assainissement, déchets",
    "F": "Construction",
    "G": "Commerce",
    "H": "Transport et entreposage",
    "I": "Hébergement et restauration",
    "J": "Information et communication",
    "L": "Immobilier",
    "M": "Activités scientifiques et techniques",
    "N": "Services administratifs et de soutien",
}
TECH_FR = {
    "E_AI_TTM": "Analyse de texte (text mining)",
    "E_AI_TNLG": "Génération de texte ou de code",
    "E_AI_TSR": "Reconnaissance vocale",
    "E_AI_TPA": "Automatisation / aide à la décision",
    "E_AI_TML": "Apprentissage automatique (analyse de données)",
    "E_AI_TIR": "Reconnaissance d'images",
    "E_AI_TAR": "Robots / machines autonomes",
    "E_AI_TPVSG": "Génération d'images, vidéos, sons",
}
FREINS_FR = {
    "E_AI_BLE": "Manque de compétences",
    "E_AI_BLEG": "Flou juridique",
    "E_AI_BCDP": "Protection des données / vie privée",
    "E_AI_BDDT": "Données insuffisantes ou de mauvaise qualité",
    "E_AI_BINC": "Incompatibilité avec les systèmes existants",
    "E_AI_BCST": "Coûts trop élevés",
    "E_AI_BEC": "Considérations éthiques",
    "E_AI_BNU": "IA jugée inutile pour l'entreprise",
}


def eurostat():
    e = pd.read_csv(RAW / "eurostat_isoc_eb_ain2.csv")
    e = e.rename(columns={"geo\\TIME_PERIOD": "geo"})
    annees = [c for c in e.columns if c.isdigit()]
    e = e.melt(id_vars=["nace_r2", "indic_is", "unit", "geo"], value_vars=annees,
               var_name="annee", value_name="valeur").dropna(subset=["valeur"])
    e["annee"] = e["annee"].astype(int)
    pays = {"EU27_2020": "UE à 27", "FR": "France"}
    e = e[e["geo"].isin(pays)]
    e["sous_categorie"] = e["geo"].map(pays)
    note = "entreprises de 10 salariés ou plus ; pas d'enquête en 2022"

    secteur = e[(e.indic_is == "E_AI_TANY") & (e.unit == "PC_ENT") & e.nace_r2.isin(NACE_FR)]
    secteur = secteur.assign(categorie=secteur.nace_r2.map(NACE_FR))
    tech = e[e.indic_is.isin(TECH_FR) & (e.unit == "PC_ENT") & (e.nace_r2 == "C10-S951_X_K")]
    tech = tech.assign(categorie=tech.indic_is.map(TECH_FR))
    freins = e[e.indic_is.isin(FREINS_FR) & (e.unit == "PC_ENT_AI_EC") & (e.nace_r2 == "C10-S951_X_K")]
    freins = freins.assign(categorie=freins.indic_is.map(FREINS_FR))
    return pd.concat([
        rows(secteur, source="Eurostat", indicateur="eurostat_ia_secteur",
             unite="% des entreprises", note=note + " ; « utilise l'IA » = au moins une technologie d'IA"),
        rows(tech, source="Eurostat", indicateur="eurostat_ia_technologie",
             unite="% des entreprises", note=note),
        rows(freins, source="Eurostat", indicateur="eurostat_ia_freins",
             unite="% des entreprises ayant envisagé l'IA sans l'utiliser", note=note),
    ])


# Codes pays Eurostat (ISO-2, sauf EL = Grèce) -> ISO-3
EUROSTAT_ISO3 = {
    "AL": "ALB", "AT": "AUT", "BA": "BIH", "BE": "BEL", "BG": "BGR", "CY": "CYP",
    "CZ": "CZE", "DE": "DEU", "DK": "DNK", "EE": "EST", "EL": "GRC", "ES": "ESP",
    "FI": "FIN", "FR": "FRA", "HR": "HRV", "HU": "HUN", "IE": "IRL", "IT": "ITA",
    "LT": "LTU", "LU": "LUX", "LV": "LVA", "ME": "MNE", "MK": "MKD", "MT": "MLT",
    "NL": "NLD", "NO": "NOR", "PL": "POL", "PT": "PRT", "RO": "ROU", "RS": "SRB",
    "SE": "SWE", "SI": "SVN", "SK": "SVK", "TR": "TUR",
    # agrégats (pas sur la carte) : EU27_2020 = UE à 27, EA = zone euro
}


def eurostat_pays():
    """Part des entreprises utilisant au moins une technologie d'IA, par pays."""
    e = pd.read_csv(RAW / "eurostat_isoc_eb_ain2.csv").rename(columns={"geo\\TIME_PERIOD": "geo"})
    e = e[(e.indic_is == "E_AI_TANY") & (e.unit == "PC_ENT") & (e.nace_r2 == "C10-S951_X_K")]
    inconnus = set(e["geo"]) - set(EUROSTAT_ISO3) - {"EU27_2020", "EA"}
    assert not inconnus, f"Codes Eurostat sans code ISO-3 : {inconnus}"
    e = e[e["geo"].isin(EUROSTAT_ISO3)]
    annees = [c for c in e.columns if c.isdigit()]
    e = e.melt(id_vars="geo", value_vars=annees, var_name="annee", value_name="valeur")
    e = e.dropna(subset=["valeur"])
    e["annee"] = e["annee"].astype(int)
    e["code_pays"] = e["geo"].map(EUROSTAT_ISO3)
    e["categorie"] = e["code_pays"].map(PAYS_FR)
    return rows(e, source="Eurostat", indicateur="eurostat_ia_pays", unite="% des entreprises",
                note="entreprises de 10 salariés ou plus, hors agriculture, finance, santé, "
                     "éducation ; pas d'enquête en 2022")


# ---------------------------------------------------------------- Kaggle Survey
# Question « algorithmes utilisés régulièrement » : nom de la question par année.
# 2017 : une seule colonne texte à choix multiples séparés par des virgules ;
# 2019-2022 : une colonne par choix ; 2018 : question absente.
KAGGLE_FILES = {
    2017: "kaggle_2017/multipleChoiceResponses.csv",
    2018: "kaggle_2018/multipleChoiceResponses.csv",
    2019: "kaggle_2019/multiple_choice_responses.csv",
    2020: "kaggle_2020/kaggle_survey_2020_responses.csv",
    2021: "kaggle_2021/kaggle_survey_2021_responses.csv",
    2022: "kaggle_2022/kaggle_survey_2022_responses.csv",
}
ALGO_PREFIX = {2019: ("Q24_Part_",), 2020: ("Q17_Part_", "Q17_OTHER"),
               2021: ("Q17_Part_", "Q17_OTHER"), 2022: ("Q18_",)}
SECTEUR_COL = {2017: "EmployerIndustry", 2018: "Q7", 2021: "Q20", 2022: "Q24"}

# Libellés harmonisés (français). 2017 utilisait des intitulés différents.
ALGO_FR = {
    # 2019-2022
    "Linear or Logistic Regression": "Régression linéaire ou logistique",
    "Decision Trees or Random Forests": "Arbres de décision / forêts aléatoires",
    "Gradient Boosting Machines (xgboost, lightgbm, etc)": "Gradient boosting (XGBoost…)",
    "Bayesian Approaches": "Méthodes bayésiennes",
    "Evolutionary Approaches": "Algorithmes évolutionnaires",
    "Dense Neural Networks (MLPs, etc)": "Réseaux de neurones « classiques »",
    "Convolutional Neural Networks": "Réseaux convolutifs (images)",
    "Generative Adversarial Networks": "Réseaux génératifs adverses (GAN)",
    "Recurrent Neural Networks": "Réseaux récurrents (séquences)",
    "Transformer Networks (BERT, gpt-2, etc)": "Transformers (BERT, GPT…)",
    "Transformer Networks (BERT, gpt-3, etc)": "Transformers (BERT, GPT…)",
    "Autoencoder Networks (DAE, VAE, etc)": "Auto-encodeurs",
    "Graph Neural Networks": "Réseaux de neurones sur graphes",
    # 2017
    "Regression/Logistic Regression": "Régression linéaire ou logistique",
    "Decision Trees": "Arbres de décision / forêts aléatoires",
    "Random Forests": "Arbres de décision / forêts aléatoires",
    "Gradient Boosted Machines": "Gradient boosting (XGBoost…)",
    "Bayesian Techniques": "Méthodes bayésiennes",
    "Neural Networks": "Réseaux de neurones « classiques »",
    "CNNs": "Réseaux convolutifs (images)",
    "GANs": "Réseaux génératifs adverses (GAN)",
    "RNNs": "Réseaux récurrents (séquences)",
    # 2017 sans équivalent ensuite (SVMs, Ensemble Methods, HMMs, Markov Logic
    # Networks) et « Other » : non retenus
}
SECTEUR_FR = {
    # 2018-2022
    "Computers/Technology": "Informatique et technologie",
    "Academics/Education": "Enseignement et recherche",
    "Accounting/Finance": "Finance et comptabilité",
    "Online Service/Internet-based Services": "Services Internet",
    "Online Business/Internet-based Sales": "Services Internet",
    "Medical/Pharmaceutical": "Santé et pharmacie",
    "Government/Public Service": "Administration publique",
    "Insurance/Risk Assessment": "Assurance",
    "Manufacturing/Fabrication": "Industrie manufacturière",
    "Marketing/CRM": "Marketing",
    "Retail/Sales": "Commerce",
    "Energy/Mining": "Énergie et mines",
    "Broadcasting/Communications": "Médias et télécommunications",
    "Shipping/Transportation": "Transport et logistique",
    "Non-profit/Service": "Associations",
    "Military/Security/Defense": "Défense et sécurité",
    "Hospitality/Entertainment/Sports": "Tourisme, loisirs et sport",
    "Other": "Autre",
    # 2017
    "Technology": "Informatique et technologie",
    "Academic": "Enseignement et recherche",
    "Financial": "Finance et comptabilité",
    "Internet-based": "Services Internet",
    "Pharmaceutical": "Santé et pharmacie",
    "Government": "Administration publique",
    "Insurance": "Assurance",
    "Manufacturing": "Industrie manufacturière",
    "CRM/Marketing": "Marketing",
    "Retail": "Commerce",
    "Telecommunications": "Médias et télécommunications",
    "Non-profit": "Associations",
    "Military/Security": "Défense et sécurité",
    "Hospitality/Entertainment/Sports": "Tourisme, loisirs et sport",
    "Mix of fields": "Autre",
    # « I am a student » (2018) : exclu, ce n'est pas un secteur
}


def lire_kaggle(annee):
    if annee == 2017:
        return pd.read_csv(RAW / KAGGLE_FILES[annee], encoding="latin-1", low_memory=False)
    # à partir de 2018, la 1re ligne sous l'en-tête contient le texte des questions
    return pd.read_csv(RAW / KAGGLE_FILES[annee], low_memory=False, skiprows=[1])


def kaggle():
    out = []
    for annee in KAGGLE_FILES:
        d = lire_kaggle(annee)

        # --- Algorithmes : une ligne par répondant et par algorithme coché
        if annee == 2017:
            choix = d["WorkAlgorithmsSelect"].dropna().str.split(",")
            longue = choix.explode().to_frame("brut")
        elif annee in ALGO_PREFIX:
            cols = [c for c in d.columns if c.startswith(ALGO_PREFIX[annee])]
            # dropna : depuis pandas 3, stack() conserve les cases vides
            longue = d[cols].stack().dropna().to_frame("brut").reset_index(level=1, drop=True)
        else:
            longue = None
        if longue is None:
            out.append(pd.DataFrame([{"annee": annee, "note": "question absente du questionnaire cette année"}])
                       .pipe(rows, source="Kaggle Survey", indicateur="kaggle_algorithmes", unite="% des répondants"))
        else:
            longue["categorie"] = longue["brut"].str.strip().map(ALGO_FR)
            base = longue.index.nunique()  # répondants ayant coché au moins un choix (« Autre » inclus)
            # un répondant 2017 qui a coché « Decision Trees » ET « Random Forests »
            # n'est compté qu'une fois dans la catégorie fusionnée
            n = (longue.dropna(subset=["categorie"]).reset_index()
                 .drop_duplicates(["index", "categorie"]).groupby("categorie").size())
            res = (100 * n / base).round(1).rename("valeur").reset_index()
            note = ("2017 : question posée aux seuls répondants qui codent au travail ; "
                    "« Neural Networks » (générique) classé en réseaux « classiques » ; "
                    "SVM, méthodes d'ensemble, HMM non repris ensuite"
                    if annee == 2017 else "")
            out.append(rows(res, source="Kaggle Survey", indicateur="kaggle_algorithmes",
                            annee=annee, unite="% des répondants", base=base, note=note))

        # --- Secteur d'activité de l'employeur
        if annee in SECTEUR_COL:
            s = d[SECTEUR_COL[annee]].map(SECTEUR_FR).dropna()
            res = (100 * s.value_counts() / len(s)).round(1).rename("valeur")
            res = res.rename_axis("categorie").reset_index()
            out.append(rows(res, source="Kaggle Survey", indicateur="kaggle_secteurs",
                            annee=annee, unite="% des répondants en emploi", base=len(s),
                            note={2017: "hors non-réponses ; « Mix of fields » classé en Autre",
                                  2018: "hors étudiants (« I am a student ») et non-réponses",
                                  2022: "hors non-réponses ; choix Défense et Tourisme absents du questionnaire 2022",
                                  }.get(annee, "hors non-réponses")))
        else:
            out.append(pd.DataFrame([{"annee": annee, "note": "question absente du questionnaire cette année"}])
                       .pipe(rows, source="Kaggle Survey", indicateur="kaggle_secteurs", unite="% des répondants en emploi"))
    return pd.concat(out)


# ---------------------------------------------------------------- OWID
def owid():
    o = pd.read_csv(RAW / "owid_ai_incidents.csv")
    o = o.rename(columns={"Year": "annee", o.columns[-1]: "valeur"})
    return rows(o[["annee", "valeur"]], source="OWID / AI Incident Database",
                indicateur="owid_incidents", categorie="Monde",
                unite="incidents et controverses signalés",
                note="recensés à partir d'articles de presse : sous-déclaration probable")


# ---------------------------------------------------------------- Vinuesa
ODD_FR = {
    1: "1. Pas de pauvreté", 2: "2. Faim « zéro »", 3: "3. Santé et bien-être",
    4: "4. Éducation de qualité", 5: "5. Égalité femmes-hommes",
    6: "6. Eau propre et assainissement", 7: "7. Énergie propre",
    8: "8. Travail décent et croissance", 9: "9. Industrie, innovation, infrastructures",
    10: "10. Inégalités réduites", 11: "11. Villes durables",
    12: "12. Consommation responsable", 13: "13. Lutte contre le changement climatique",
    14: "14. Vie aquatique", 15: "15. Vie terrestre",
    16: "16. Paix, justice, institutions", 17: "17. Partenariats",
}


def vinuesa():
    v = pd.read_excel(RAW / "vinuesa2020_supplementary_data_1.xlsx", header=None,
                      skiprows=3, usecols=range(4), names=["cible", "texte", "levier", "frein"])
    v["cible"] = v["cible"].astype(str).str.strip()
    # lignes « Goal X » = titres d'objectifs ; les autres = cibles (ex. « 13.2 »).
    # Excel a transformé « 8.10 » en 8.1 : on ne garde que le préfixe (n° d'ODD).
    cibles = v[v["cible"].str.match(r"^\d+\.")].copy()
    cibles["odd"] = cibles["cible"].str.split(".").str[0].astype(int)
    g = cibles.groupby("odd").agg(nb_cibles=("cible", "size"),
                                  levier=("levier", lambda s: (s == 1).sum()),
                                  frein=("frein", lambda s: (s == 1).sum())).reset_index()
    g = g.melt(id_vars=["odd", "nb_cibles"], value_vars=["levier", "frein"],
               var_name="sous_categorie", value_name="valeur")
    g["sous_categorie"] = g["sous_categorie"].map({"levier": "L'IA peut aider (levier)",
                                                   "frein": "L'IA peut nuire (frein)"})
    g["categorie"] = g["odd"].map(ODD_FR)
    return rows(g.rename(columns={"nb_cibles": "base"}), source="Vinuesa et al. 2020",
                indicateur="vinuesa_odd", annee=2020, unite="cibles",
                note="avis d'experts fondé sur la littérature ; une cible peut être levier ET frein")


def format_large(df):
    """Tableau « large » : une ligne par année, une colonne par indicateur et catégorie.

    Nom des colonnes : « indicateur | catégorie | sous-catégorie » (ex.
    « eurostat_ia_secteur | Commerce | UE à 27 »). Une case vide = pas de donnée
    cette année-là (année non couverte, question absente, valeur non publiée).
    Les effectifs utiles aux calculs (« base ») sont ajoutés en colonnes.
    Unités et notes détaillées : voir le tableau long panorama_ia.csv.
    """
    d = df.dropna(subset=["valeur"]).copy()
    morceaux = [d["indicateur"], d["categorie"].fillna(""), d["sous_categorie"].fillna("")]
    d["colonne"] = [" | ".join(p for p in parts if p) for parts in zip(*morceaux)]

    # effectifs de référence (sinon perdus en passant au format large)
    k = df[df["indicateur"] == "kaggle_algorithmes"].dropna(subset=["base"]).drop_duplicates("annee")
    k = k.assign(colonne="kaggle_algorithmes | base (répondants)", valeur=k["base"])
    s = df[df["indicateur"] == "kaggle_secteurs"].dropna(subset=["base"]).drop_duplicates("annee")
    s = s.assign(colonne="kaggle_secteurs | base (répondants en emploi)", valeur=s["base"])
    v = df[df["indicateur"] == "vinuesa_odd"].drop_duplicates("categorie")
    v = v.assign(colonne="vinuesa_odd | " + v["categorie"] + " | nombre de cibles", valeur=v["base"])
    d = pd.concat([d, k, s, v])

    # ordre des colonnes : celui des indicateurs dans le tableau long, puis alphabétique
    ordre = {ind: i for i, ind in enumerate(df["indicateur"].unique())}
    colonnes = (d.drop_duplicates("colonne")
                 .assign(o=lambda x: x["indicateur"].map(ordre))
                 .sort_values(["o", "colonne"])["colonne"])
    doublons = d.duplicated(["annee", "colonne"])
    assert not doublons.any(), f"Plusieurs valeurs pour une même case : {d[doublons].head()}"
    large = d.pivot_table(index="annee", columns="colonne", values="valeur", aggfunc="first")
    return large[colonnes.tolist()].sort_index()


if __name__ == "__main__":
    CLEAN.mkdir(parents=True, exist_ok=True)
    df = pd.concat([epoch(), epoch_pays(), eurostat(), eurostat_pays(), kaggle(), owid(),
                    vinuesa()], ignore_index=True)
    pays = df[df["indicateur"].isin(["epoch_pays", "eurostat_ia_pays"])]
    assert pays["categorie"].notna().all(), "Pays sans nom français"
    df["annee"] = df["annee"].astype("Int64")
    df.to_csv(CLEAN / "panorama_ia.csv", index=False)
    print(df.groupby(["source", "indicateur"]).agg(lignes=("valeur", "size"),
                                                  annee_min=("annee", "min"),
                                                  annee_max=("annee", "max")).to_string())
    print(f"-> {CLEAN / 'panorama_ia.csv'} ({len(df)} lignes)")

    large = format_large(df)
    large.to_csv(CLEAN / "panorama_ia_large.csv")
    print(f"-> {CLEAN / 'panorama_ia_large.csv'} ({large.shape[0]} années x {large.shape[1]} colonnes)")
