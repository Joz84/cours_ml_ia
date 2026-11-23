"""Corrigé du TP 3 · Visualiser (utilisé par scripts/build_tp_donnees.py)."""

NOTEBOOK = "TP3_visualisation"
FICHIERS = ["data/clean/releves_meteo_propre.csv", "tp/visualisation/boite_a_outils_graphiques.py"]

TROUS = {
    "q1": '''barres(df, "station", "pluie_mm", agg="sum",
       titre="Vallée a reçu le plus de pluie en 2024 : 4 203 mm", unite="mm", source=SOURCE)''',
    "q2": '''courbe(df, "date", "pluie_mm", par="station", garder=["Vallée", "Port"], pas="M", agg="sum",
       titre="Vallée reçoit plus de pluie que Port presque chaque mois", unite="mm par mois", source=SOURCE)''',
    "q3": '''histogramme(df[df["pluie_mm"] > 0], "pluie_mm", bins=30,
            titre="La moitié des jours de pluie apportent moins de 11 mm", unite="mm", source=SOURCE)''',
    "q4": '''par_station = df.groupby("station", as_index=False)[["altitude_m", "pluie_mm"]].mean()
nuage(par_station, "altitude_m", "pluie_mm", etiquettes="station",
      titre="Il pleut plus en altitude, mais Vallée fait exception", unite=("m", "mm par jour"), source=SOURCE)''',
    "q5": '''boite(df[df["pluie_mm"] > 0], "station", "pluie_mm",
      titre="Les jours de pluie varient le plus à Vallée et à Col", unite="mm", source=SOURCE)''',
    "q6": '''barres_croisees(df, "station", "saison", "pluie_mm", agg="mean",
                titre="La saison des pluies triple la pluie à Vallée, la double seulement à Port",
                unite="mm par jour", source=SOURCE)''',
    "q7": '''barres(df, "station", "pluie_mm", agg="mean",
       titre="Vallée reçoit 2,1 fois plus de pluie par jour que Port", unite="mm par jour", source=SOURCE)''',
}

REPONSES = {
    'Q1': '(a) **comparer** des catégories ; (b) **`barres`**, avec `agg="sum"` ; (d) « Vallée a reçu le plus de pluie en 2024 : 4 203 mm », plus de 2,6 fois `Lagon` (1 600 mm). Mais regardez les n : `Col` n\'a que 173 valeurs, sur 6 mois. Son total n\'est pas comparable à celui des autres : il faudrait comparer la pluie par jour, ou sur les mêmes mois.',
    'Q2': '(a) **suivre** dans le temps ; (b) **`courbe`**, avec `par="station"`, `garder=["Vallée", "Port"]` et `agg="sum"` ; (d) « Vallée reçoit plus de pluie que Port presque chaque mois », surtout de décembre à mars. En juillet, `Port` dépasse `Vallée` : ce pic vient des deux journées de 180 et 210 mm, les valeurs « étonnantes » gardées au TP 1. Un graphique fait ressortir les valeurs à revérifier.',
    'Q3': "(a) voir la **répartition** ; (b) **`histogramme`**, sur les seuls jours de pluie ; (d) « La moitié des jours de pluie apportent moins de 11 mm ». La moyenne (20,2 mm) est presque le double de la médiane (10,9) : quelques journées très fortes, jusqu'à 231 mm, tirent la moyenne vers le haut.",
    'Q4': '(a) chercher un **lien** ; (b) **`nuage`**, sur un tableau d\'une ligne par station (`par_station`), avec `etiquettes="station"` ; (d) « Il pleut plus en altitude, mais Vallée fait exception ». Le lien n\'est pas parfait : `Vallée`, à 150 m, reçoit plus que `Plateau`, à 450 m, sans doute parce qu\'elle est exposée aux vents humides. Avec un point par station, r = 0,79 : on compare des moyennes, et les énormes variations d\'un jour à l\'autre, avec tous leurs zéros, disparaissent. Prudence : seulement 5 points, et `Col` n\'a que la saison des pluies.',
    'Q5': "(a) **comparer des groupes en détail** ; (b) **`boite`**, sur les jours de pluie ; (d) « Les jours de pluie varient le plus à Vallée et à Col » : leurs boîtes sont les plus hautes et les plus larges, avec un écart interquartile d'environ 28 mm, contre 12 mm à `Lagon`.",
    'Q6': "(a) **croiser** deux catégories, la station et la saison ; (b) **`barres_croisees`** ; (d) « La saison des pluies triple la pluie à Vallée (17,9 contre 6,1 mm par jour), la double seulement à Port (7,5 contre 3,7) ». La différence n'est donc pas la même partout. `Col` n'a qu'une barre : il n'a aucun relevé en saison fraîche.",
    'Q7 (bonus)': 'Constat repris : « Vallée reçoit 2,1 fois plus de pluie par jour que Port ». (a) **comparer** des catégories ; (b) **`barres`**, avec `agg="mean"` ; (d) le constat lui-même devient le titre. La moyenne par jour rend `Col` un peu plus comparable, mais sa barre ne parle que de la saison des pluies.',
    'Q8': "Avec l'axe coupé à 1 500 mm, la barre de `Vallée` (4 203) paraît environ **27 fois** plus haute que celle de `Lagon` (1 600), car on ne voit que ce qui dépasse 1 500 : 2 703 contre 100. Avec l'axe à zéro, elle paraît **2,6 fois** plus haute : c'est le vrai rapport. Une barre se lit par sa longueur ; si elle ne part pas de zéro, sa longueur ment. Une courbe se lit par sa forme et ses variations, pas par sa hauteur depuis zéro : on peut zoomer, à condition de le dire.",
}
