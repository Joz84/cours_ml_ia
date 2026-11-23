"""Corrigé du TP 1 · Nettoyer (utilisé par scripts/build_tp_donnees.py)."""
import shutil

NOTEBOOK = "TP1_nettoyage"
FICHIERS = ["data/raw/releves_meteo_brut.csv", "data/raw/pluie_mensuelle_large.csv"]

TROUS = {
    "decision_trous": '''decision_trous = "C"        # DÉCISION : "A", "B" ou "C"
justification_trous = "Vide ne veut pas dire zéro : remplir inventerait des mesures, et supprimer les lignes ferait perdre les autres colonnes. Les trous (environ 4 %) seront signalés dans l'analyse."
print("Option choisie :", decision_trous or "AUCUNE : tapez A, B ou C")''',
    "decision_harmoniser": '''CORRESPONDANCES = {"Vallee": "Vallée"}   # DÉCISION : par exemple {"PORT": "Port"}
justification_harmoniser = "Vallee et Vallée désignent la même station : on garde l'écriture officielle, avec l'accent."

for colonne in colonnes_texte:
    df[colonne] = df[colonne].replace(CORRESPONDANCES)
noter("7 Harmoniser", f"correspondances : {CORRESPONDANCES}", "écritures unifiées", justification_harmoniser)
for colonne in colonnes_texte:
    print(colonne, ":", df[colonne].value_counts().to_dict())''',
    "decision_impossibles": '''decision_impossibles = "B"        # DÉCISION : "A", "B" ou "C"
justification_impossibles = "270, 99 et -5 °C ne peuvent pas exister à Tahiti : ce sont des erreurs de saisie. On vide seulement la valeur fausse et on garde la pluie et l'autre température de la ligne."
print("Option choisie :", decision_impossibles or "AUCUNE : tapez A, B ou C")''',
    "decision_incoherentes": '''decision_incoherentes = "B"        # DÉCISION : "A", "B" ou "C"
justification_incoherentes = "Dans ces 5 lignes, les deux températures sont plausibles mais inversées : c'est une erreur de colonne à la saisie, on les remet dans le bon ordre."
print("Option choisie :", decision_incoherentes or "AUCUNE : tapez A, B ou C")''',
}

REPONSES = {
    'Geste 1': "Une ligne est **un relevé d'une journée à une station**. Ce qui est déjà bizarre : les dates sont écrites de trois façons (« 27 nov. 24 », « 2024-07-22 », « 08/05/2024 ») ; certains nombres ont une virgule (« 17,4 »), d'autres un point ; on voit des « NA » et des « -999 » ; `remarque` n'est remplie que sur 604 lignes sur 1 686 ; et toutes les colonnes sont lues comme du texte. Question à poser ensuite : où et quand pleut-il le plus, et fait-il plus frais en altitude ?",
    'Geste 2': "35 cases n'ont pas pu être converties. Les « NA » (16 dans `pluie_mm`, 18 dans `temp_min_C`) sont des **absences de mesure** : rien n'est perdu en les passant en vide. Le « vingt-deux » de `temp_max_C` est une **vraie valeur** (22 °C) écrite en lettres. Un humain la lirait, mais la machine ne devine pas : écrire une règle pour chaque mot serait risqué, on préfère un vide visible et noté dans le journal.",
    'Geste 3': "`remarque` n'est remplie qu'à 35,8 % et ne contient que trois textes (« RAS », « capteur nettoyé », « relevé tardif ») : elle ne répond à aucune question sur la météo. `code_agent` non plus : on retire les deux.",
    'Geste 4': "Le tableau des relevés est déjà **long** : une ligne = une station, un jour. La version longue du tableau mensuel compte **60 lignes** : 5 stations × 12 mois. Les mois de mai à octobre de `Col` sont **vides** parce qu'il n'y a aucun relevé (la station ne mesure qu'en saison des pluies) : écrire 0 voudrait dire « il n'a pas plu », ce qui serait faux.",
    'Geste 5': "Un trou dans `pluie_mm` veut dire « pas de mesure », pas « pas de pluie » : le remplacer par 0 ferait baisser à tort les totaux et les moyennes. Le corrigé choisit l'**option C** : laisser vide et le noter. Pendant l'analyse, il faudra compter les n réels (1 580 valeurs de pluie pour 1 646 jours) et ne pas comparer des totaux de stations qui n'ont pas les mêmes trous sans le dire.",
    'Geste 6': 'Les 40 doublons compteraient deux fois les mêmes journées : la pluie totale des stations concernées serait gonflée, et les nombres de relevés (les n) seraient faux.',
    'Geste 7': 'Au départ, `Port` s\'écrit de **4 façons** : `Port` (306 lignes), `port` (23), `PORT ` (19) et ` Port` (18). Après la normalisation, il n\'en reste **qu\'une** (366 lignes). La machine signale `Vallee` mais ne le corrige pas seule : sans accent, ce pourrait être une autre station au nom proche. C\'est une décision humaine, d\'où le dictionnaire `{"Vallee": "Vallée"}`.',
    'Geste 8': "270 °C n'existe nulle part au sol : c'est forcément une erreur de saisie, hors des bornes, donc **impossible**. 210 mm en un jour est rare mais arrive à Tahiti lors d'une forte dépression : c'est **étonnant**, on vérifie mais on garde. Dans les 5 lignes incohérentes, les deux températures sont plausibles mais dans le mauvais ordre (par exemple 33,2 en minimum et 26,3 en maximum). L'explication la plus probable est une inversion de colonnes à la saisie : les échanger (option B) garde l'information.",
    'Geste 9': 'En grande partie, oui : chaque geste a son constat chiffré, sa décision et sa justification, et les réglages (codes de trou, bornes, règles) sont dans le carnet. Il manquerait la version exacte du fichier brut et la date du nettoyage. On passe de **1 686 à 1 646 lignes** : seules les 40 lignes en double ont été supprimées. Tous les autres problèmes ont été traités sans supprimer de ligne.',
}


def APRES(racine):
    """Range le tableau propre et le journal produits par le corrigé dans data/clean/."""
    dossier = racine / "tp" / "nettoyage"
    for nom in ["releves_meteo_propre.csv", "journal_nettoyage.md"]:
        shutil.move(str(dossier / nom), str(racine / "data" / "clean" / nom))
    # le journal cite le fichier d'origine : on remet son adresse GitHub à la place du chemin local
    journal = racine / "data" / "clean" / "journal_nettoyage.md"
    journal.write_text(journal.read_text().replace(
        str(racine / FICHIERS[0]), "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/" + FICHIERS[0]))
