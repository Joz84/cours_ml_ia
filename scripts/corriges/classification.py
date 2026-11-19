"""Corrigé du TP de classification (utilisé par scripts/build_tp.py)."""

CSV = "baignade_taharuu.csv"

# Cellules à trous : code complet du corrigé, par identifiant d'étape
TROUS = {
    "1.3": '''X = df.drop(columns=["date", "conforme"])
y = df["conforme"]

print("X :", X.shape, "  y :", y.shape)
for colonne in X.columns:
    print(f"  {colonne:15} ->", "nombre" if pd.api.types.is_numeric_dtype(X[colonne]) else "texte")''',
}

REPONSES = {
    "1.0": "Le tableau contient **2 193 lignes** (un jour sur une plage) et **8 colonnes**. La cible est "
           "`conforme` : ce n'est pas un nombre mais une **catégorie** (`oui` / `non`). C'est donc un problème de "
           "classification : le modèle répond à « laquelle ? », pas à « combien ? ».",
    "1.1": "Aucune case vide : toutes les colonnes ont 2 193 valeurs. Quatre colonnes sont du texte : `date`, "
           "`plage`, `saison` et `conforme`. Les quatre autres sont des nombres à virgule (`float64`).",
    "1.2": "Non : les classes sont très **déséquilibrées**, avec 83,7 % de `oui` et seulement 16,3 % de `non`. Un "
           "modèle paresseux qui répondrait toujours `oui` aurait raison 83,7 % du temps, sans rien avoir appris. "
           "Il faudra s'en souvenir pour juger le score.",
    "1.3": "`X = df.drop(columns=[\"date\", \"conforme\"])` et `y = df[\"conforme\"]`. Mettre `conforme` dans X "
           "reviendrait à donner la réponse au modèle : il « tricherait » et serait inutile pour un jour où l'on ne "
           "connaît pas encore le résultat d'analyse. La date, elle, ne se répète jamais : le modèle apprendrait le "
           "calendrier passé au lieu des vraies causes (pluie, débit, plage).",
    "1.4": "**1 754 lignes** pour l'entraînement et **439** pour le test. Grâce à `stratify=y`, la part de `non` est "
           "presque identique dans les deux (16,3 % et 16,4 %), comme dans le tableau complet.",
    "1.5": "Quatre colonnes numériques (`pluie_24h_mm`, `pluie_72h_mm`, `temperature_C`, `debit_m3s`) et deux "
           "colonnes de texte (`plage` et `saison`).",
    "1.6": "Ce sont les pluies qui ont les plus grands nombres : `pluie_72h_mm` monte jusqu'à 242,9 mm. La "
           "température varie très peu (écart-type de 1,6 °C, entre 20,8 et 29,7 °C). Après normalisation, toutes "
           "les colonnes ont une moyenne de 0 et un écart-type de 1 : aucune n'écrase les autres à cause de son unité.",
    "1.7": "**5 colonnes** : une par plage (`plage_pk18`, `plage_taharuu`, `plage_vaiava`) et une par saison "
           "(`saison_ete_austral`, `saison_hiver_austral`). Sur chaque ligne, une seule colonne `plage_…` et une "
           "seule colonne `saison_…` valent 1.",
    "1.8": "**`non` reçoit 0 et `oui` reçoit 1**, par ordre alphabétique. Attention : la classe qui nous intéresse "
           "(`non`, la pollution) porte donc le numéro 0. Il faudra le dire explicitement à scikit-learn au TP 2.",
    "1.9": "**9 colonnes** : les 4 colonnes numériques normalisées et les 5 colonnes encodées (3 plages et "
           "2 saisons). Le tableau ne contient plus que des nombres : le modèle peut le lire.",
    "2.1": "Non : le modèle est encore une « boîte vide ». Il n'a vu aucune donnée ; il apprendra avec `fit`.",
    "2.2": "Le modèle ne sait calculer qu'avec des nombres : `y_train` contient du texte (`oui` / `non`), alors que "
           "`y_train_enc` contient les mêmes réponses sous forme de 0 et de 1.",
    "2.3": "L'exactitude est de **91,1 %** : le modèle a raison sur 91 jours de test sur 100. Ce chiffre a l'air "
           "très bon… mais il faut le comparer au modèle paresseux (question Q2).",
    "2.4": "Sur ces 10 jours, le modèle se trompe **une fois** : à la ligne 5, il annonce `oui` alors que l'eau "
           "était non conforme. C'est justement l'erreur la plus grave ici : une pollution ratée.",
    "2.Q1": "Le score est de **91,1 %**. Pour un classifieur, `score` mesure l'**exactitude** : la part des jours de "
            "test pour lesquels la classe prédite est la bonne, ici 400 jours sur 439 (42 + 358, voir Q3). Il ne "
            "dit pas *sur quelle classe* le modèle se trompe.",
    "2.Q2": "Le modèle paresseux, qui répond toujours `oui`, a déjà **83,6 %** d'exactitude, sans rien regarder. "
            "Notre modèle ne fait que 7,5 points de mieux (91,1 %). Un score de 90 % n'a donc rien d'impressionnant "
            "ici : c'est le **piège de l'exactitude** avec une classe rare. Et le modèle paresseux ne détecterait "
            "**aucune** pollution.",
    "2.Q3": "Le modèle a **raté 30 jours non conformes** (FN, en haut à droite) sur les 72 jours vraiment non "
            "conformes du test. Il a repéré les 42 autres (VP). Il a aussi donné **9 fausses alertes** (FP) : des "
            "jours conformes annoncés non conformes.",
    "2.Q4": "**Rappel de `non` = 42 / (42 + 30) = 58,3 %** : le modèle ne repère que 58 % des jours pollués. "
            "**Précision de `non` = 42 / (42 + 9) = 82,4 %** : quand il donne l'alerte, il a raison 8 fois sur 10. "
            "Ici, **rater coûte cher** (des baigneurs malades) alors qu'une fausse alerte ne coûte qu'une "
            "journée de baignade : il faut privilégier le **rappel**, et 58 % est clairement insuffisant.",
    "2.Q5": "Avec un seuil de **0,3**, le modèle déclare `non` plus facilement : le rappel **monte** (69,4 %) et la "
            "précision **baisse** (72,5 %). Avec **0,7**, c'est l'inverse : rappel de 44,4 %, précision de 91,4 %. "
            "Conclusion : baisser le seuil fait rater moins de pollutions au prix de plus de fausses alertes. Pour "
            "protéger les baigneurs, on choisit un seuil **bas** (0,3, voire moins).",
    "2.Q6": "Pour le jour proposé (80 mm de pluie en 3 jours, rivière à 8 m³/s, à Taharu'u), le modèle répond "
            "`non` avec une probabilité de 99 % : c'est logique, tous les signaux de pollution sont au rouge. Pour un "
            "jour sec (0 mm, débit faible), il répond `oui`. Le modèle a l'air raisonnable, mais il ne remplace pas "
            "une vraie analyse d'eau.",
    "2.Q7": "Le coefficient le plus négatif, donc celui qui pousse le plus vers `non`, est celui de "
            "**`pluie_72h_mm`** (−1,59), suivi de `plage_taharuu` (−0,88), puis de `pluie_24h_mm` et `debit_m3s` "
            "(−0,54 chacun). C'est logique : la pluie des derniers jours lessive les sols vers la mer, et Taharu'u est "
            "à l'embouchure d'une grande rivière. À lire avec prudence : la pluie et le débit vont ensemble et se "
            "« partagent » l'effet. C'est d'ailleurs la règle qui a servi à simuler les données.",
}
