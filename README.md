# UE 5.7 · Data Science et IA

Licence STES · Université de la Polynésie française

Ce cours apprend à **faire parler des données environnementales**, sans être informaticien : les nettoyer, les lire, les montrer, puis en tirer des modèles qui prédisent ou qui regroupent. Les exemples viennent du terrain polynésien : la pluie et le soleil mesurés par Météo-France, la qualité de l'eau de baignade, la consommation d'eau des abonnés.

## Comment on travaille

- **En binôme, dans Google Colab** : rien à installer, un navigateur suffit. Chaque TP a un bouton « Ouvrir dans Colab ».
- **On copie-colle, on ne programme pas.** Chaque TP donne toutes les cellules de code, dans l'ordre, avec le résultat attendu sous chacune.
- **Une seule cellule à modifier : RÉGLAGES.** Elle contient le nom du fichier et des colonnes. Tout le reste fonctionne sur n'importe quel tableau : chaque TP est aussi un **tutoriel réutilisable** pour vos propres données.
- **Un rituel à chaque étape : je prédis → j'exécute → je constate.** On répond aux questions *avant* d'exécuter, puis on écrit deux phrases sur ce qu'on observe.
- **Quand une cellule plante**, on la colle avec son message d'erreur dans ChatGPT ou Claude, et on demande une explication simple. Quand une cellule est à compléter, un encadré « Ce que vous dites à l'assistant » donne la phrase à dire.

## Le plan des TP

### Séance 5 · Des données propres, des données lues

Avant tout modèle, il faut un tableau propre, compris et bien montré. Les trois TP s'enchaînent sur les mêmes relevés météo de 5 stations de Tahiti (données simulées) : le TP 2 et le TP 3 partent du tableau nettoyé au TP 1.

| TP | Ce qu'on apprend | Durée |
|---|---|---|
| [**TP 1 · Nettoyer**](tp/nettoyage/README.md) | Les **9 gestes** du nettoyage, dans l'ordre : regarder, typer, trier, structurer, trous, doublons, harmoniser, impossibles et incohérentes, tracer. Ce que la machine fait seule, ce qu'elle dépiste pour vous, ce qui reste un jugement. Le journal de nettoyage. | 1 h 30 |
| [**TP 2 · Analyser**](tp/analyse_exploratoire/README.md) | La **check-list de Peng** (vérifier ses n, valider avec une source extérieure), **ydata-profiling** (« tout d'un coup, mais pas à la place de vous »), les **7 regards** et les deux réflexes de terrain de Zuur. Cinq constats chiffrés. | 1 h 30 |
| [**TP 3 · Visualiser**](tp/visualisation/README.md) | **Le verbe de la question choisit le graphique.** Une boîte à outils de 8 fonctions qui imposent les 5 points d'un bon graphique. Le piège de l'axe coupé. | 1 h 30 |

### Séance 6 · Apprendre des données : le machine learning

Chaque TP se fait en deux parties de 45 minutes : le **preprocessing** (séparer, trier, normaliser, encoder), puis les **quatre verbes** du modèle (importer et créer, `fit`, `score`, `predict`).

| TP | Ce qu'on apprend | Durée |
|---|---|---|
| [**TP · Régression linéaire**](tp/regression_lineaire/README.md) | Prédire **un nombre** : le rayonnement solaire reçu au sol, à partir de l'ensoleillement, sur de vraies données Météo-France de Polynésie. Le R² et la baseline. | 2 × 45 min |
| [**TP · Classification**](tp/classification/README.md) | Prédire **une catégorie** : la plage est-elle baignable ? Régression logistique, matrice de confusion, rappel et précision, le piège de l'exactitude avec une classe rare, le seuil. | 2 × 45 min |
| [**TP · Clustering**](tp/clustering/README.md) | **Regrouper sans cible** : quels profils d'abonnés à l'eau ? K-Means, la normalisation obligatoire, la méthode du coude. « Le modèle numérote les groupes, c'est vous qui les nommez. » | 2 × 45 min |

### Séance 7 · Deep learning et images

Une photo entre, une classe sort : corail sain, pâle ou blanchi ; type de déchet ; miconia ou autre végétation. On ne part jamais de zéro : on réutilise un réseau pré-entraîné.

| TP | Ce qu'on apprend | Durée |
|---|---|---|
| [**TP · Classer des photos**](tp/classification_images/README.md) | Le **transfert d'apprentissage** (MobileNetV2, poids gelés, dernière couche neuve) ; le **preprocessing des images** (collecter, séparer avant tout, nettoyer, redimensionner, augmenter à la volée) ; les deux courbes et le sur-apprentissage ; la matrice de confusion et le **seuil de confiance** ; la **carte de chaleur** pour vérifier où regarde le modèle. Un carnet-tuto à rejouer sur vos propres photos. | 2 h |

### Annexe

| TP | Ce qu'on apprend |
|---|---|
| [**TP-prompting**](tp/annexe_prompting/README.md) | Explorer, nettoyer et visualiser un jeu de données **en dialoguant avec un assistant IA**, à partir de prompts, sur un panorama de l'IA et du machine learning (Epoch AI, Eurostat, Kaggle, Our World in Data). |

## Le dépôt

```
tp/                    un dossier par TP : README (l'énoncé), notebook ÉLÈVE, notebook CORRIGÉ
data/raw/              les fichiers bruts du TP 1 (les autres données brutes ne sont pas versionnées)
data/clean/            les tableaux prêts à l'emploi, lus par les TP depuis GitHub
data/*.zip             les photos d'exemple du TP « Classer des photos »
data/sources/          les données brutes de l'annexe, une par source (et data/sources_propres/ nettoyées)
scripts/               les scripts qui fabriquent les données et les notebooks
```

Tous les fichiers de données se lisent directement depuis Colab, par leur adresse GitHub, par exemple `https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/clean/releves_meteo_propre.csv`.

Pour regénérer les données et les notebooks en local : `pip install -r requirements.txt`. Les scripts `scripts/prepare_*.py`, `scripts/make_releves_brut.py` et `scripts/make_dataset_exemple.py` fabriquent les données ; les scripts `scripts/build_tp*.py` fabriquent les notebooks à partir des README.

## Les données et leurs sources

| Données | Source | Réelles ou simulées |
|---|---|---|
| Relevés météo de 5 stations (TP 1, 2, 3) | climat calqué sur Météo-France, station de Faa'a | simulées, salies volontairement pour le TP 1 |
| Rayonnement solaire (régression) | Météo-France, données climatologiques quotidiennes, département 987, Licence Ouverte 2.0 | réelles |
| Baignade (classification) | pluie et température Météo-France ; débit et conformité simulés | en partie simulées |
| Abonnés à l'eau (clustering) | — | simulées |
| Photos de fleurs (classer des photos) | jeu « flower_photos » de TensorFlow (Flickr), licence CC BY 2.0, auteurs listés dans chaque zip | réelles |
| Panorama de l'IA (annexe) | Epoch AI, Eurostat, Kaggle, Our World in Data, Vinuesa et al. 2020 | réelles |
