# Journal de nettoyage

Fichier d'origine : https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/raw/releves_meteo_brut.csv

Avant : 1686 lignes et 8 colonnes. Après : 1646 lignes et 6 colonnes.

| geste | constat | décision | justification |
|---|---|---|---|
| 1 Regarder | 1686 lignes, 8 colonnes | travail sur une copie | L'original doit rester intact. |
| 2 Typer | 35 cases illisibles : NA (pluie_mm), NA (temp_min_C), vingt-deux (temp_max_C) | passées en vide | Ce ne sont pas des valeurs lisibles : on ne devine pas ce qu'elles valaient. |
| 3 Trier | colonnes retirées : code_agent, remarque | retirées | Elles ne servent à aucune question sur la météo. |
| 4 Structurer | tableau déjà long | rien à faire | Une ligne est déjà un relevé (un jour, une station). |
| 5 Trous | 38 valeurs codées [-999, -9999] | transformées en trous | Un code -999 n'est pas une mesure : il fausserait les moyennes. |
| 5 Trous | 139 cases vides | laissées vides | Vide ne veut pas dire zéro : remplir inventerait des mesures, et supprimer les lignes ferait perdre les autres colonnes. Les trous (environ 4 %) seront signalés dans l'analyse. |
| 6 Doublons | 40 lignes en double | supprimées | Une même observation ne doit compter qu'une fois. |
| 7 Harmoniser | correspondances : {'Vallee': 'Vallée'} | écritures unifiées | Vallee et Vallée désignent la même station : on garde l'écriture officielle, avec l'accent. |
| 8 Étonnantes | valeurs à plus de 3 écarts-types | gardées | Rares mais plausibles : on vérifie, on ne supprime pas. |
| 8 Impossibles | 3 valeurs hors bornes | remplacées par vide | 270, 99 et -5 °C ne peuvent pas exister à Tahiti : ce sont des erreurs de saisie. On vide seulement la valeur fausse et on garde la pluie et l'autre température de la ligne. |
| 8 Incohérentes | 5 lignes | valeurs échangées | Dans ces 5 lignes, les deux températures sont plausibles mais inversées : c'est une erreur de colonne à la saisie, on les remet dans le bon ordre. |
