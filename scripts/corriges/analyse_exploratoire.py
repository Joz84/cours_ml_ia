"""Corrigé du TP 2 · Analyser (utilisé par scripts/build_tp_donnees.py)."""
from pathlib import Path

NOTEBOOK = "TP2_analyse"
FICHIERS = ["data/clean/releves_meteo_propre.csv"]

TROUS = {
    "reference": '''REFERENCE = 1579              # À COMPLÉTER : la valeur trouvée, par exemple 1500 (un nombre, sans unité)
SOURCE_REFERENCE = "Météo-France, station de Faa'a, moyenne 1991-2020 des cumuls annuels, calculée à partir des données quotidiennes publiées sur data.gouv.fr"
CATEGORIE_A_COMPARER = "Port" # la catégorie du tableau à comparer à cette référence

if COLONNE_DATE:
    par_an = df.groupby([COLONNE_CATEGORIE, df[COLONNE_DATE].dt.year])[VALEUR].agg(AGREGATION)
    cumul = par_an.groupby(level=0).mean()
else:
    cumul = df.groupby(COLONNE_CATEGORIE)[VALEUR].agg(AGREGATION)
print("Cumul annuel de", VALEUR, "par", COLONNE_CATEGORIE, ":")
print(cumul.round(1))

if REFERENCE is None:
    print("\\nTapez une valeur de référence dans REFERENCE, puis relancez.")
else:
    ecart = cumul[CATEGORIE_A_COMPARER] / REFERENCE
    print(f"\\n{CATEGORIE_A_COMPARER} : {cumul[CATEGORIE_A_COMPARER]:.0f} contre {REFERENCE} pour la référence ({SOURCE_REFERENCE})")
    print(f"Rapport : {ecart:.2f}, soit un écart de {(ecart - 1) * 100:+.0f} %")''',
}

REPONSES = {
    'Peng 3': "Q1 · **366 relevés** pour `Port`, `Lagon`, `Plateau` et `Vallée` (toute l'année 2024, qui est bissextile), mais seulement **182 pour `Col`**, et uniquement de novembre à avril, en saison des pluies. Comparer le total de pluie de `Col` aux autres serait faux : 6 mois contre 12. Même sa moyenne par jour est gonflée, puisqu'elle ne contient aucun jour de saison fraîche. Il faut comparer à périodes égales.",
    'Peng 4': "Référence : **1 579 mm par an**, normale 1991-2020 de Faa'a, calculée à partir des données quotidiennes de Météo-France (data.gouv.fr). `Port` reçoit 1 989 mm, soit **+26 %**. C'est possible pour une seule année, mais une partie de l'écart vient de deux journées de 180 et 210 mm en saison fraîche, gardées comme « étonnantes » au TP 1 : sans elles, `Port` serait vers 1 600 mm, très proche de la normale. La validation extérieure montre que le tableau est réaliste, et elle pointe exactement les valeurs à revérifier.",
    'Q2': "**1 646 lignes, 6 colonnes, 136 cases vides (1,4 %) et 0 doublon.** C'est cohérent avec le journal du TP 1 : 1 686 lignes moins les 40 doublons supprimés. Les 136 trous ont été laissés vides : 66 en pluie, 67 en température minimale (66 trous et 1 valeur impossible) et 3 en température maximale (2 impossibles et le « vingt-deux »).",
    'Q3': '« Missing » pour `pluie_mm` (66, soit 4 %) et `temp_min_C` (67) : un problème connu, traité au TP 1 et assumé. « Zeros » pour `pluie_mm` (926, soit 56,3 %) : une **propriété** des données, ce sont les jours sans pluie. « High correlation » entre `altitude_m` et `station` : une propriété, chaque station a une seule altitude. Entre `temp_min_C`, `temp_max_C` et `altitude_m` : une propriété physique, il fait plus frais en montant. Aucune alerte ne signale une erreur.',
    'Q4': "Moyenne : **8,4 mm** ; médiane : **0 mm**. L'histogramme est très asymétrique : un énorme pic à 0, puis une longue traîne jusqu'à 231 mm. La journée typique est racontée par la **médiane** : un jour sur deux, il ne pleut pas du tout. La moyenne est tirée vers le haut par quelques journées très pluvieuses.",
    'Q5': "`temp_min_C` et `temp_max_C` (environ 0,89) : le lien est **positif**, comme attendu, car une nuit chaude est suivie d'une journée chaude. La température minimale est aussi fortement liée à l'altitude, dans le sens négatif (-0,83 avec `corr()` au regard 6) : il fait plus frais en montant.",
    'Q6': "Les points forment **5 colonnes verticales**, parce que l'altitude ne prend que 5 valeurs, une par station. À chaque altitude, il y a beaucoup de zéros. Les grosses pluies sont plus fréquentes à 150 m (`Vallée`) et à 900 m (`Col`). Jour par jour, il n'y a pas de lien en ligne droite (r = 0,12 au regard 6).",
    'Regard 2': "Moyenne **8,4 mm**, médiane **0 mm** : très éloignées. La répartition est très asymétrique : plus d'un jour sur deux sans pluie, et quelques journées très pluvieuses qui tirent la moyenne vers le haut.",
    'Regard 4': "`Col` a la plus forte moyenne (14,7 mm par jour), mais sur 182 jours de saison des pluies seulement. Parmi les stations suivies toute l'année, c'est **`Vallée`** (12,0 mm par jour) : **2,1 fois plus que `Port`** (5,6 mm).",
    'Regard 5': '**Février (16,7 mm par jour) et janvier (16,5)**, puis décembre (13,0). Les plus secs sont août (1,5) et septembre (2,2). Oui, on retrouve la saison des pluies de novembre à avril, même si novembre (6,6) et avril (5,9) sont des mois de transition. Attention : de novembre à avril, la moyenne compte aussi `Col`, la station la plus arrosée, ce qui gonfle un peu ces mois. Encore les n.',
    'Regard 6': "La paire la plus corrélée est **`temp_min_C` et `temp_max_C` (0,89)**. Entre altitude et pluie, jour par jour, **r = 0,12** : un lien très faible, car la plupart des jours il ne pleut nulle part, et la pluie varie énormément d'un jour à l'autre. L'altitude et la température minimale sont fortement liées, dans le sens négatif (-0,83).",
    'Regard 7': 'Non. Le mois le plus pluvieux est février pour `Vallée` (34,0) et `Col` (25,6), janvier pour `Plateau` (28,6), décembre pour `Lagon` (13,6), et juillet pour `Port` (14,0), à cause des deux journées de 180 et 210 mm. Pour `Col`, les cases de mai à octobre sont vides (aucun relevé), pas égales à zéro.',
    'Zuur 1': "`pluie_mm` a **56,3 % de zéros**. Ce n'est pas une erreur, c'est la pluie : plus d'un jour sur deux est sec. Mais cela fausse les moyennes et les corrélations : il faut le dire, et parfois étudier à part les jours de pluie (ce qu'on fera au TP 3).",
    'Zuur 2': "Oui : les cinq stations montent ensemble de décembre à mars et descendent ensemble de mai à septembre, parce qu'elles subissent les mêmes systèmes météo. Leurs relevés **ne sont pas indépendants** : 1 646 relevés ne valent pas 1 646 informations indépendantes. Deux jours qui se suivent, ou deux stations voisines, se ressemblent, ce qui rend les tests statistiques trop optimistes.",
    'Q7': "1. « Vallée reçoit **2,1 fois** plus de pluie par jour que Port (12,0 contre 5,6 mm). » 2. « Plus d'un jour sur deux est sec : **56 %** des relevés ont 0 mm. » 3. « En février, il tombe en moyenne **11 fois** plus de pluie par jour qu'en août (16,7 contre 1,5 mm). » 4. « À Col (900 m), la température minimale est en moyenne **5,1 °C** plus basse qu'à Port (18,4 contre 23,5 °C). » 5. « Col n'a que **182** relevés, deux fois moins que les autres stations, tous en saison des pluies. »",
}


def APRES(racine):
    """Le rapport HTML produit pendant l'exécution n'est pas versionné."""
    (racine / "tp" / "analyse_exploratoire" / "rapport_profil.html").unlink(missing_ok=True)
