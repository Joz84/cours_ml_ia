"""Corrigé du TP de clustering (utilisé par scripts/build_tp.py)."""

CSV = "abonnes_eau.csv"

# Cellules à trous : code complet du corrigé, par identifiant d'étape
TROUS = {
    "1.3": '''X = df[colonnes_num]

print("X :", X.shape)
print(list(X.columns))''',
    "2.Q6": '''from sklearn.metrics import silhouette_score

scores = {}
for k in range(2, 9):
    modele_k = KMeans(n_clusters=k, random_state=42)
    groupes_k = modele_k.fit_predict(X_normalise)
    scores[k] = silhouette_score(X_normalise, groupes_k)
    print(f"K = {k} : silhouette = {scores[k]:.3f}")

meilleur_k = max(scores, key=scores.get)
print("\\nMeilleur K selon la silhouette :", meilleur_k)''',
}

REPONSES = {
    "1.0": "**600 abonnés** (une ligne chacun) et **7 colonnes**.",
    "1.1": "**Aucune** : il n'y a pas de cible, rien à prédire. Sans cible, il n'y a pas de « bonne réponse » à "
           "vérifier sur un jeu de test, donc pas besoin de séparer entraînement et test. Et il n'y a évidemment "
           "pas de cible à encoder. Il ne reste que 3 étapes : trier, choisir les colonnes et normaliser.",
    "1.2": "Deux colonnes ne sont pas numériques : `abonne` (l'identifiant) et `commune`. On les écarte : "
           "l'identifiant ne dit rien des habitudes de consommation, et on cherche des groupes selon *comment* "
           "les abonnés consomment, pas *où* ils habitent.",
    "1.3": "`X = df[colonnes_num]` : X a **5 colonnes** (consommation, pic d'été, part de nuit, nombre de "
           "personnes, surface de jardin) pour 600 abonnés.",
    "1.4": "C'est `surface_jardin_m2` : en moyenne 213 m², jusqu'à 1 048 m², avec un écart-type de 270, contre 1,9 "
           "pour `nb_personnes`. Sans normalisation, les distances seraient presque entièrement décidées par le "
           "jardin. Après normalisation, toutes les colonnes ont une moyenne de 0 et un écart-type de 1.",
    "2.1": "K = 3 est un choix de départ : on soupçonne quelques grands profils d'abonnés, et on vérifiera ce choix "
           "avec la méthode du coude (Q3). **K est choisi par nous**, pas par le modèle. Avec K = 600, chaque abonné "
           "serait seul dans son groupe : l'inertie serait nulle, mais les groupes ne nous apprendraient rien.",
    "2.2": "Parce qu'il n'y a **pas de cible** : le K-Means ne cherche pas à reproduire une bonne réponse, il "
           "cherche seulement des abonnés proches les uns des autres. Il n'a besoin que des features.",
    "2.3": "Non : 0, 1 et 2 ne sont que des **étiquettes**. Le groupe 2 n'est ni « mieux » ni « plus grand » que le "
           "groupe 0, et avec un autre `random_state`, les mêmes groupes pourraient recevoir d'autres numéros. "
           "C'est à nous de comprendre ce qu'ils représentent.",
    "2.4": "Le **groupe 2** consomme le plus (58,7 m³ par mois en moyenne). Le **groupe 0** a de loin les plus grands "
           "jardins (548 m² en moyenne). Le groupe 1 consomme peu (10,3 m³) et n'a presque pas de jardin.",
    "2.Q1": "**208** abonnés dans le groupe 0, **275** dans le groupe 1 et **117** dans le groupe 2. Les groupes "
            "n'ont pas la même taille : le K-Means ne cherche pas des groupes égaux, mais des groupes d'abonnés "
            "qui se ressemblent.",
    "2.Q2": "**Groupe 0, « les familles avec jardin »** : environ 5 personnes, plus de 500 m² de jardin ; leur "
            "consommation grimpe de 44 % l'été et un tiers de l'eau part la nuit, sans doute pour l'arrosage. "
            "**Groupe 1, « les petits foyers »** : 1 à 2 personnes, presque pas de jardin, une consommation "
            "faible (10 m³ par mois) et régulière toute l'année. **Groupe 2, « les pensions et commerces »** : "
            "aucun résident déclaré, mais la plus forte consommation (59 m³ par mois), presque uniquement le jour.",
    "2.Q3": "Le coude est à **K = 3** : l'inertie chute de 3 000 à 1 164 (K = 2) puis à 521 (K = 3). Ensuite, elle ne "
            "baisse presque plus : seulement 73 de moins à K = 4. C'est cohérent avec les **trois profils** "
            "nommés en Q2 : un quatrième groupe découperait un profil en deux sans rien apporter.",
    "2.Q4": "Les groupes changent complètement : **115, 403 et 82** abonnés au lieu de 208, 275 et 117. Le grand "
            "groupe de 403 mélange les petits foyers et les pensions et commerces : leurs consommations, très "
            "différentes (10 et 59 m³), donnent une moyenne trompeuse de 24,7 m³. Les familles, elles, sont coupées "
            "en deux selon la taille du jardin (452 contre 736 m²). C'est **`surface_jardin_m2`** qui a « écrasé » "
            "les autres : ses écarts se comptent en centaines, ceux des autres colonnes en unités ou en dizaines. "
            "Sans normalisation, la distance ne mesure presque que le jardin.",
    "2.Q5": "Le nouvel abonné tombe dans le **groupe 0, les familles avec jardin** : 5 personnes, 700 m² de "
            "jardin, 35 % de sa consommation la nuit. Message possible : « Votre arrosage représente une grande "
            "partie de votre consommation : arrosez au goutte-à-goutte, récupérez l'eau de pluie, et vérifiez que "
            "votre arrosage automatique ne tourne pas toute la nuit. »",
    "2.Q6": "La silhouette est la meilleure pour **K = 3** (0,629, contre 0,577 pour K = 2 et 0,503 pour K = 4) : "
            "elle est d'accord avec le coude. Mais ces indicateurs ne font qu'aider : **c'est nous qui tranchons**, "
            "selon l'usage. Ici, trois groupes donnent trois messages faciles à expliquer et à envoyer : c'est "
            "le bon choix pour la campagne.",
    "2.Q7": "Oui : le **groupe 2 (pensions et commerces)** est concentré à **Punaauia**, avec 70 abonnés sur 117, "
            "soit 60 %. Les familles avec jardin (groupe 0) sont plus nombreuses à Papara (97 sur 208). Pour la "
            "campagne, on peut viser les professionnels à Punaauia, par exemple avec des rencontres avec les "
            "pensions et les commerces, et envoyer le message sur l'arrosage surtout à Papara.",
}
