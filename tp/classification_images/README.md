# Annexe · Classer des photos – exhaustif

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Joz84/cours_ml_ia/blob/main/tp/classification_images/TP_classification_images_ELEVE.ipynb)

UE 5.7 « Data Science et IA » · Licence STES · Université de la Polynésie française

> **Version exhaustive**, en annexe : tout le code est visible, cellule par cellule. Pour la séance, préférez la [version simple du TP](../classer_photos/README.md), où chaque cellule tient en une ou deux lignes grâce à une boîte à outils.

Dans le cours de la séance 7, « Deep learning et images », vous avez vu trois cas : reconnaître un **corail sain, pâle ou blanchi** ; trier des photos de **déchets par type** ; repérer le **miconia** au milieu de la végétation. Les trois sont le même problème : **une photo entre, une classe sort.** Ce TP vous fait entraîner un tel modèle de bout en bout, en **réutilisant un réseau pré-entraîné** (le **transfert d'apprentissage**), puis l'évaluer avec les outils de la séance 6, l'utiliser sur des photos nouvelles, et vérifier **où il regarde**.

- **Durée :** environ 2 heures, dont quelques minutes d'entraînement avec un GPU.
- **En binôme**, dans **Google Colab**. Vous n'avez rien à installer.
- **Méthode :** ouvrez le carnet avec le bouton « Ouvrir dans Colab » ci-dessus, ou copiez chaque cellule de code de cette page dans une nouvelle cellule Colab. Exécutez-les **dans l'ordre**, avec Maj + Entrée.
- **Vous ne modifiez que la cellule RÉGLAGES**, plus les quelques cellules marquées **DÉCISION** ou **À MODIFIER**.
- **Si une cellule plante**, copiez-la avec le message d'erreur dans ChatGPT ou Claude et demandez : « Explique-moi cette erreur simplement et corrige la cellule. »

## Un carnet-tuto pour classer vos propres photos

Ce carnet est écrit pour servir à **n'importe quel problème de classification d'images à peu de classes** (2 à 5, au plus une dizaine), pas seulement aux fleurs de l'exemple.

- **Le nombre de classes et leurs noms ne sont écrits nulle part dans le code** : le carnet les lit dans les noms des sous-dossiers. Un dossier par classe, c'est l'étiquette.
- **Tout ce qui change d'un projet à l'autre est dans la cellule RÉGLAGES** : où sont les photos, leur taille, la part gardée pour l'examen, le nombre d'époques, le seuil de confiance et les augmentations permises.
- **Toutes les autres cellules se copient telles quelles**, pour 2, 3, 4 ou 5 classes.

### La structure de dossiers attendue

```text
mes_photos/                 ← le dossier que vous indiquez dans RÉGLAGES (dans votre Drive, ou zippé)
├── sain/                   ← un sous-dossier par classe : son nom EST l'étiquette
│   ├── IMG_0001.jpg
│   ├── IMG_0002.jpg
│   └── …                   ← environ 300 photos par classe
├── pale/
│   └── …
└── blanchi/
    └── …
```

- **Environ 300 photos par classe**, à peu près autant dans chaque classe.
- **Des noms de dossiers sans accent ni espace** : `pale`, pas `pâle` ; `sac_plastique`, pas `sac plastique`. Les noms de fichiers peuvent être quelconques.
- **Des photos en JPEG ou PNG.** Les photos d'iPhone en HEIC ne sont pas lues : exportez-les en JPEG (dans l'app Photos, ou avec Google Photos qui les convertit au téléchargement). Le carnet compte et signale les fichiers qu'il ne lit pas.
- Si vos photos viennent de **plusieurs sites ou de plusieurs sorties**, rangez-les en un sous-dossier par site *dans* chaque classe (`sain/site_A/…`, `sain/site_B/…`) : le bloc 2 vous montre comment s'en servir pour un examen honnête.

### Lancer sur le jeu d'exemple, en 5 minutes

1. Ouvrez le carnet ÉLÈVE avec le bouton « Ouvrir dans Colab ».
2. Activez le GPU : menu **Exécution → Modifier le type d'exécution → GPU (T4)** → Enregistrer.
3. Menu **Exécution → Tout exécuter**. Les réglages pointent déjà vers le jeu d'exemple : il se télécharge seul (23 Mo).

### Rejouer sur vos propres photos

1. Rangez vos photos dans votre Google Drive comme ci-dessus, par exemple dans `MyDrive/coraux/`.
2. Dans la cellule RÉGLAGES, remplacez l'adresse du zip d'exemple par le chemin de votre dossier : `PHOTOS = "/content/drive/MyDrive/coraux"`. Faites de même pour `PHOTOS_NOUVELLES` (un dossier de photos jamais montrées au modèle).
3. Relisez la liste `AUGMENTATIONS` et mettez `False` devant celles qui changeraient la classe de vos photos.
4. Exécutez tout le carnet dans l'ordre. C'est tout : aucune autre cellule ne change.

Vous pouvez aussi zipper votre dossier et donner l'adresse du zip (sur GitHub, ou un lien de téléchargement direct).

## Du cours au carnet

| Étape du cours | Bloc du carnet | Questions |
|---|---|---|
| Activer le GPU | 0 · Activer le GPU | — |
| **0** Collecter et étiqueter (≈ 300 par classe, équilibré, varié, bien étiqueté) | 1 · Collecter et étiqueter | Q1, Q2 |
| **2** Nettoyer (doublons, floues, mal rangées, hors sujet) | 1 · Écarter des photos (DÉCISION) | Q2 |
| **1** Séparer 80 / 20 avant tout le reste | 2 · Séparer | Q3, Q4 |
| **3** Redimensionner en carré et normaliser | 3 · Redimensionner et normaliser | — |
| **+** Augmenter, à la volée, sur l'entraînement seulement | 3 · Augmenter | Q5 |
| **4** Encoder : rien, le dossier fait le travail | 2 · Séparer (les classes sont lues dans les dossiers) | — |
| Les quatre verbes · **importer et créer** (base gelée + dernière couche neuve) | 4 · Importer et créer | Q6 |
| Les quatre verbes · **fit**, et les deux courbes | 5 · Entraîner | Q7 |
| Les quatre verbes · **évaluer** (exactitude, matrice de confusion, rappel) | 6 · Évaluer | Q8, Q9, Q10 |
| Les quatre verbes · **predict**, et le seuil de confiance | 7 · Prédire | Q11, Q12 |
| Vérifier où regarde le modèle (carte de chaleur) | 8 · Où regarde le modèle ? | Q13 |
| Les diagnostics : trop faible, trop beau | 9 · Si le résultat déçoit, si le résultat est trop beau | Q14 (bonus) |
| Les limites ; le journal de la séance 5 | 10 · Sauvegarder et documenter | Q15 |

L'étape 2 « nettoyer » vient dans le carnet juste après la collecte, car on repère les photos à écarter en les regardant ; elle se fait bien sûr **avant** d'entraîner.

## Les ordres de grandeur

Mesurés sur le jeu d'exemple (900 photos, 3 classes), sauf mention contraire.

| | Avec GPU (Colab, T4) | Sans GPU |
|---|---|---|
| Une époque (720 photos d'entraînement) | quelques secondes (non mesuré ici) | 9 s sur un ordinateur portable récent (Apple M1, 8 cœurs) ; nettement plus dans Colab sans GPU, qui n'a que 2 cœurs |
| Les 10 époques | 1 à 2 minutes | 1 minute sur ce portable ; comptez plutôt 10 minutes dans Colab sans GPU |
| Exactitude sur la validation | environ 90 à 95 % | identique |

Sur vos propres photos, comptez à peu près **le même temps par photo** : 1 500 photos prennent environ deux fois plus longtemps que l'exemple. Une exactitude de 80 à 95 % est plausible pour 3 classes bien distinctes ; si vous obtenez 99 % dès la deuxième époque, méfiez-vous (bloc 9).

> **Ce que ce carnet ne fait pas.** Il répond à « quelle classe pour cette photo entière ? ». Il ne sait pas :
> - **trouver et compter les objets** dans une photo (la **détection** : « où sont les déchets sur cette plage ? ») ;
> - **délimiter une zone pixel par pixel** (la **segmentation** : « quelle surface du récif est blanchie ? ») ;
> - gérer **plus d'une dizaine de classes**, ou des classes avec très peu de photos ;
> - traiter des **images médicales**, ou des images qui ne sont pas des photos (radar, hyperspectral).
>
> Pour ces cas, parlez-en avec l'enseignant : d'autres modèles existent, mais ce n'est pas ce carnet.

## Le rituel, à chaque étape

> **Je prédis** : avant d'exécuter une cellule qui donne un chiffre, j'écris mon pari à la ligne « Je prédis ».
> **J'exécute** : je lance la cellule.
> **Je constate** : j'écris deux phrases sur ce que j'observe, et je corrige ma prédiction si besoin.

## Les données d'exemple

Pour que le carnet tourne avant que vous ayez vos photos, il est livré avec **900 photos de fleurs en 3 classes** : `marguerite`, `tournesol` et `tulipe`, 300 photos chacune, en 256 × 256 px, dans [`data/images_exemple.zip`](../../data/images_exemple.zip) (23 Mo). S'y ajoutent **32 photos nouvelles**, jamais montrées au modèle, dans [`data/photos_nouvelles.zip`](../../data/photos_nouvelles.zip) : 8 par classe apprise, plus 4 roses et 4 pissenlits, deux fleurs **que le modèle ne connaît pas**. Leur vraie fleur est dans le fichier `etiquettes.csv` du zip.

- **Source :** le jeu « flower_photos » de TensorFlow (aussi appelé `tf_flowers`), 3 670 photos de fleurs publiées sur Flickr, en 5 classes. Le script [`scripts/make_dataset_exemple.py`](../../scripts/make_dataset_exemple.py) le télécharge, tire les photos au hasard (graine 42), les recadre en carré au centre et les met en 256 × 256 px.
- **Licence :** Creative Commons Attribution 2.0 ([CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)) : réutilisation libre, à condition de citer les auteurs. L'auteur et l'adresse de chaque photo sont dans le fichier `LICENCE_ET_AUTEURS.txt` de chaque zip. Seules des photos listées dans le fichier de licence du jeu d'origine ont été gardées.
- **Pourquoi ce jeu ?** Ce sont de vraies photos, prises au téléphone ou à l'appareil, de qualité inégale, avec des fonds variés : comme les vôtres. Les trois classes se distinguent à l'œil, mais pas toujours facilement (une tulipe jaune, un tournesol vu de dos). La couleur y compte, comme pour le corail : on peut donc discuter des augmentations permises (Q5) et regarder où le modèle regarde (Q13). Le jeu EuroSAT (images satellite : forêt, mer, ville), plus proche des images de drone, a été écarté : ses images font 64 × 64 px, trop petites pour qu'une carte de chaleur se lise.
- **Limite :** les photos d'origine font en général 320 × 240 px ; le carré central (240 px) est un peu agrandi pour atteindre 256 px.

---

## Le carnet

### Réglages · La seule cellule à modifier

Pour le jeu d'exemple, ne changez rien. Pour vos photos, c'est ici, et seulement ici, que vous changez les chemins et les choix. Chaque ligne a son commentaire.

```python
# ===================== RÉGLAGES : la seule cellule à modifier =====================
PHOTOS = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/images_exemple.zip"  # vos photos : l'adresse d'un zip, ou un dossier du Drive, par ex. "/content/drive/MyDrive/coraux"
PHOTOS_NOUVELLES = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/photos_nouvelles.zip"  # bloc 7 : des photos jamais montrées au modèle (zip ou dossier du Drive)
TAILLE = 256              # étape 3 : les photos sont mises en carré, TAILLE x TAILLE pixels
PART_VALIDATION = 0.2     # étape 1 : la part des photos gardée pour l'examen (0.2 = 20 %)
EPOQUES = 10              # bloc 5 : le nombre maximum de passages sur les photos d'entraînement
SEUIL = 0.6               # bloc 7 : en dessous de cette confiance, la photo part « à vérifier par un humain »
AUGMENTATIONS = {         # étape + : les transformations permises (True) ou interdites (False)
    "miroir": True,       # retourner la photo de gauche à droite (False si le sens compte : texte, flèche)
    "rotation": True,     # tourner un peu, de 10 degrés au plus
    "zoom": True,         # zoomer ou dézoomer un peu, de 10 % au plus
    "luminosite": True,   # éclaircir ou assombrir un peu
    "couleur": False,     # changer les teintes : mettre False pour le corail, la couleur EST la classe
}
DOSSIER_SORTIE = "/content/drive/MyDrive/TP_photos"   # bloc 10 : le dossier du Drive où enregistrer le modèle et sa fiche
# ===================================================================================
```

## Bloc 0 · Activer le GPU

Un **GPU** est une carte de calcul qui fait des milliers de petites multiplications en même temps : exactement ce dont un réseau de neurones a besoin. Colab en prête un gratuitement, mais il faut le demander **avant** d'exécuter le carnet :

> Menu **Exécution → Modifier le type d'exécution** → choisir **GPU** (par exemple « T4 GPU ») → **Enregistrer**.

Changer le type d'exécution redémarre Colab : faites-le tout au début, puis exécutez le carnet depuis la première cellule.

### Bloc 0 · Vérifier le GPU et préparer les outils

La cellule vérifie qu'un GPU est présent, importe les bibliothèques et prépare deux petits outils : `recuperer`, qui rend vos photos disponibles (téléchargement du zip ou montage du Drive), et `lire_photo`, qui ouvre une photo à la bonne taille. Exécutez-la sans la modifier.

```python
import os, sys, random, shutil, time, zipfile, urllib.request
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import warnings
warnings.filterwarnings("ignore", message=".*shuffle=True.*")   # un avertissement de Keras sans conséquence

EN_COLAB = "google.colab" in sys.modules
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")   # les formats lus par le carnet
keras.utils.set_random_seed(42)                           # pour que le hasard soit le même à chaque fois

if tf.config.list_physical_devices("GPU"):
    print("GPU trouvé : l'entraînement prendra quelques minutes.")
else:
    print("ATTENTION : pas de GPU. L'entraînement marchera, mais prendra une heure au lieu de quelques minutes "
          "sur un gros dossier de photos.\nMenu Exécution → Modifier le type d'exécution → GPU, puis relancez le carnet depuis le début.")

def recuperer(source, nom):
    """Rend un dossier de photos : télécharge et dézippe un zip, ou monte le Drive."""
    if source.startswith("/content/drive") and EN_COLAB:
        from google.colab import drive
        drive.mount("/content/drive")
    if source.startswith("http"):
        urllib.request.urlretrieve(source, nom + ".zip")
        source = nom + ".zip"
    if source.endswith(".zip"):
        shutil.rmtree(nom, ignore_errors=True)
        with zipfile.ZipFile(source) as z:
            z.extractall(nom)
        source = nom
    dossier = Path(source)
    # un zip contient souvent un seul dossier : on descend dedans
    while True:
        sous = [d for d in dossier.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))]
        images = [f for f in dossier.iterdir() if f.suffix.lower() in EXTENSIONS]
        if len(sous) == 1 and not images:
            dossier = sous[0]
        else:
            return dossier

def lire_photo(chemin):
    """Ouvre une photo, la recadre en carré au centre, la met en TAILLE x TAILLE."""
    photo = keras.utils.load_img(chemin, target_size=(TAILLE, TAILLE), keep_aspect_ratio=True)
    return keras.utils.img_to_array(photo).astype("uint8")

def photos_de(dossier):
    """Toutes les photos d'un dossier et de ses sous-dossiers."""
    return sorted(f for f in Path(dossier).rglob("*") if f.suffix.lower() in EXTENSIONS)

print("TensorFlow", tf.__version__, "· outils prêts.")
```

**Résultat attendu**

```text
GPU trouvé : l'entraînement prendra quelques minutes.
TensorFlow 2.x.x · outils prêts.
```

Sans GPU, vous lisez le message « ATTENTION : pas de GPU » : suivez-le, ou continuez si vous acceptez d'attendre.

## Bloc 1 · Collecter et étiqueter

*Étape 0 du cours.* Un modèle n'apprend que ce qu'on lui montre : **≈ 300 photos par classe, équilibré, varié, bien étiqueté, représentatif** du terrain où il servira. Ici, la collecte est déjà faite ; on vérifie qu'elle est bonne.

### Bloc 1a · Lire les sous-dossiers

**La cible est le nom du dossier.** La cellule récupère vos photos, lit les noms des sous-dossiers (ce sont les classes), compte les photos de chacun, et signale les fichiers qu'elle ne sait pas lire (HEIC, vidéos, documents).

**Je prédis :** combien de classes, et combien de photos en tout ? …

```python
DOSSIER = recuperer(PHOTOS, "mes_photos")
CLASSES = sorted(d.name for d in DOSSIER.iterdir() if d.is_dir() and not d.name.startswith((".", "_")))
K = len(CLASSES)

comptes = pd.DataFrame({
    "classe": CLASSES,
    "photos": [len(photos_de(DOSSIER / c)) for c in CLASSES],
    "fichiers non lus": [sum(1 for f in (DOSSIER / c).rglob("*") if f.is_file() and f.suffix.lower() not in EXTENSIONS)
                         for c in CLASSES],
})
print("Dossier :", DOSSIER)
print(K, "classes :", CLASSES)
display(comptes)

plt.figure(figsize=(6, 3))
plt.bar(comptes["classe"], comptes["photos"], color="#2a78d6")
plt.title("Nombre de photos par classe")
plt.ylabel("photos")
plt.show()
```

**Résultat attendu**

```text
Dossier : mes_photos/images_exemple
3 classes : ['marguerite', 'tournesol', 'tulipe']

       classe  photos  fichiers non lus
0  marguerite     300                 0
1   tournesol     300                 0
2      tulipe     300                 0
```

Et un graphique en barres, une barre par classe, toutes à la même hauteur.

### Bloc 1b · Neuf photos au hasard, avec leur étiquette

```python
toutes = [(f, c) for c in CLASSES for f in photos_de(DOSSIER / c)]
plt.figure(figsize=(9, 9))
for i, (f, c) in enumerate(random.sample(toutes, 9)):
    plt.subplot(3, 3, i + 1)
    plt.imshow(lire_photo(f))
    plt.title(c)
    plt.axis("off")
plt.show()
```

**Résultat attendu :** une grille de 3 × 3 photos, chacune avec le nom de sa classe au-dessus.

### Q1 · Le jeu est-il équilibré ?

**Questions**
- Combien de classes, combien de photos par classe ?
- Le jeu est-il équilibré ?
- Si une classe avait moins de la moitié des photos des autres, que proposeriez-vous ? (Cours : sous-échantillonner les autres, ou aller chercher des photos.)

### Q2 · Regarder les photos une par une

*Étape 2 du cours, nettoyer : doublons, photos floues, mal rangées, hors sujet.* La cellule affiche **20 photos au hasard de chaque classe**, avec le nom de leur fichier. Regardez-les vraiment, une par une.

```python
for c in CLASSES:
    tirage = random.sample(photos_de(DOSSIER / c), min(20, len(photos_de(DOSSIER / c))))
    plt.figure(figsize=(13, 12))
    plt.suptitle(f"Classe « {c} » : 20 photos au hasard", fontsize=14)
    for i, f in enumerate(tirage):
        plt.subplot(4, 5, i + 1)
        plt.imshow(lire_photo(f))
        plt.title(f"{c}/{f.relative_to(DOSSIER / c)}", fontsize=7)
        plt.axis("off")
    plt.show()
```

**Résultat attendu :** une planche de 20 photos par classe. Le nom au-dessus de chaque photo est celui à recopier dans la cellule suivante si vous voulez l'écarter.

**Questions**
- Y a-t-il des photos floues, hors sujet, ou mal rangées (dans la mauvaise classe) ? Notez leurs noms.
- Voyez-vous des quasi-doublons (la même scène, prise deux fois) ?

### Bloc 1c · DÉCISION : écarter des photos

La cellule copie vos photos dans un **dossier de travail**, **sans les photos que vous écartez**, et sans les fichiers abîmés qu'elle ne sait pas ouvrir. **Rien n'est supprimé de votre Drive** : l'original reste intact, comme au geste 1 de la séance 5. Recopiez dans la liste les noms notés à Q2, entre guillemets, séparés par des virgules. Laisser la liste vide est aussi une décision : dites pourquoi.

<!-- id: ecarter -->
```python
A_ECARTER = []        # DÉCISION : par ex. ["tulipe/1234_abcd_n.jpg", "tournesol/5678_efgh_m.jpg"]
justification_ecarter = ""   # DÉCISION : une phrase

TRAVAIL = Path("photos_travail")
shutil.rmtree(TRAVAIL, ignore_errors=True)
ecartees, abimees = [], []
for c in CLASSES:
    for f in photos_de(DOSSIER / c):
        nom = f"{c}/{f.relative_to(DOSSIER / c)}"
        if nom in A_ECARTER:
            ecartees.append(nom)
            continue
        try:
            lire_photo(f)
        except Exception:
            abimees.append(nom)
            continue
        (TRAVAIL / nom).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, TRAVAIL / nom)

print("Écartées par vous :", len(ecartees), ecartees)
print("Abîmées, illisibles :", len(abimees), abimees)
print("Introuvables (vérifiez l'orthographe) :", sorted(set(A_ECARTER) - set(ecartees)))
gardees = {c: len(photos_de(TRAVAIL / c)) for c in CLASSES}
print("Photos gardées par classe :", gardees)
```

**Résultat attendu**

```text
Écartées par vous : 6 ['marguerite/21805938544_bf6bb0e4bc.jpg', 'tournesol/15054750690_198b6ab0f2_n.jpg', …]
Abîmées, illisibles : 0 []
Introuvables (vérifiez l'orthographe) : []
Photos gardées par classe : {'marguerite': 299, 'tournesol': 298, 'tulipe': 297}
```

## Bloc 2 · Séparer 80 / 20, avant tout le reste

*Étape 1 du cours.* On met de côté **20 % des photos pour l'examen** (la **validation**) **avant** toute autre chose : le modèle ne les verra jamais pendant l'entraînement, et c'est sur elles qu'on le jugera. La fonction `image_dataset_from_directory` fait trois choses d'un coup : elle lit les photos, elle **donne à chaque photo l'étiquette de son dossier** (*étape 4, encoder : rien à faire, le dossier fait le travail*), et elle sépare. On l'appelle deux fois, une fois pour l'entraînement (`subset="training"`), une fois pour la validation (`subset="validation"`), **avec la même graine** (`seed=42`) : c'est ce qui garantit que les deux parties ne se recouvrent pas.

Les photos sont rangées par **lots** de 32 : le modèle les regarde 32 par 32.

### Bloc 2a · Séparer

**Je prédis :** combien de photos iront dans l'entraînement, combien dans la validation ? …

```python
reglages_lecture = dict(class_names=CLASSES, validation_split=PART_VALIDATION, seed=42,
                        image_size=(TAILLE, TAILLE), batch_size=32, crop_to_aspect_ratio=True)
apprentissage = keras.utils.image_dataset_from_directory(TRAVAIL, subset="training", **reglages_lecture)
validation = keras.utils.image_dataset_from_directory(TRAVAIL, subset="validation", **reglages_lecture)

def par_classe(partie):
    """Le nombre de photos de chaque classe dans une partie : la classe est le nom du dossier."""
    classes = [next(d for d in reversed(Path(f).parts[:-1]) if d in CLASSES) for f in partie.file_paths]
    return pd.Series(classes).value_counts().reindex(CLASSES, fill_value=0)

print()
print("Entraînement :", len(apprentissage.file_paths), "photos en", len(apprentissage), "lots")
print("Validation   :", len(validation.file_paths), "photos en", len(validation), "lots")
display(pd.DataFrame({"entraînement": par_classe(apprentissage), "validation": par_classe(validation)}))
```

**Résultat attendu**

```text
Found 894 files belonging to 3 classes.
Using 716 files for training.
Found 894 files belonging to 3 classes.
Using 178 files for validation.

Entraînement : 716 photos en 23 lots
Validation   : 178 photos en 6 lots

            entraînement  validation
marguerite           238          61
tournesol            236          62
tulipe               242          55
```

### Q3 · Vérifier le partage

**Questions**
- Votre prédiction était-elle juste ? Retrouvez le calcul : 80 % de combien ?
- Chaque classe a-t-elle à peu près 20 % de ses photos dans la validation ?

### Q4 · Le piège des rafales

Si vos photos sont **prises en rafale**, ou **plusieurs par site**, deux photos presque identiques peuvent tomber l'une dans l'entraînement, l'autre dans la validation. Le modèle « reconnaît » alors la photo d'examen parce qu'il a vu sa jumelle : l'examen est truqué. La parade du cours : **séparer par site ou par sortie**, pas photo par photo.

La cellule bonus ci-dessous le fait, **si vos photos sont rangées en un sous-dossier par site dans chaque classe** (`sain/site_A/…`). Listez les sites réservés à l'examen ; laissée vide, la cellule ne change rien.

<!-- id: sites -->
```python
SITES_VALIDATION = []   # BONUS, DÉCISION : par ex. ["site_B"] ; toutes les photos de ces sites vont dans la validation

if SITES_VALIDATION:
    for partie in ["app", "val"]:
        shutil.rmtree(f"photos_{partie}", ignore_errors=True)
    for f in photos_de(TRAVAIL):
        rel = f.relative_to(TRAVAIL)          # classe/site/photo.jpg
        partie = "val" if rel.parts[1] in SITES_VALIDATION else "app"
        (Path(f"photos_{partie}") / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, Path(f"photos_{partie}") / rel)
    lecture_sites = dict(class_names=CLASSES, seed=42, image_size=(TAILLE, TAILLE), batch_size=32, crop_to_aspect_ratio=True)
    apprentissage = keras.utils.image_dataset_from_directory("photos_app", **lecture_sites)
    validation = keras.utils.image_dataset_from_directory("photos_val", **lecture_sites)
    print("Séparation par site : sites de validation =", SITES_VALIDATION)
else:
    print("Pas de séparation par site : on garde le partage photo par photo du bloc 2a.")
```

**Résultat attendu**

```text
Pas de séparation par site : on garde le partage photo par photo du bloc 2a.
```

**Questions**
- Vos photos (celles de votre projet) sont-elles prises en rafale ou plusieurs par site ?
- Si oui, expliquez en deux phrases pourquoi le partage photo par photo truque l'examen.

## Bloc 3 · Redimensionner, normaliser, augmenter

*Étape 3 du cours.* Un réseau veut des photos **toutes de la même taille** et des pixels **dans une petite plage de valeurs**. Une photo, pour la machine, c'est un tableau de nombres : **256 × 256 × 3 = 196 608 nombres** (3 pour rouge, vert, bleu), chacun entre 0 et 255. **Les features sont des pixels.**

### Bloc 3a · Une photo, avant et après

La cellule montre une photo telle qu'elle est dans le dossier, puis recadrée en carré au centre et mise en 256 × 256, puis ses pixels **normalisés**. MobileNetV2, le réseau qu'on va réutiliser, attend des pixels entre −1 et 1 : on calcule `pixel / 127,5 − 1`. Vous n'aurez pas à le faire vous-même : le modèle du bloc 4 contient cette normalisation, et il redimensionne aussi la photo en 224 × 224, la taille d'entrée de MobileNetV2.

```python
f = photos_de(TRAVAIL / CLASSES[0])[0]
originale = keras.utils.load_img(f)
carree = lire_photo(f)
normalisee = carree / 127.5 - 1

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.imshow(originale); plt.title(f"Originale : {originale.width} x {originale.height} px")
plt.subplot(1, 2, 2); plt.imshow(carree); plt.title(f"Carrée : {TAILLE} x {TAILLE} px")
plt.show()
print("Nombres par photo :", carree.size, "=", " x ".join(str(n) for n in carree.shape))
print("Pixels avant : de", carree.min(), "à", carree.max(), "· le pixel en haut à gauche (R, V, B) :", carree[0, 0])
print("Pixels après : de", normalisee.min().round(2), "à", normalisee.max().round(2), "· le même pixel :", normalisee[0, 0].round(2))
```

**Résultat attendu**

```text
Nombres par photo : 196608 = 256 x 256 x 3
Pixels avant : de 0 à 235 · le pixel en haut à gauche (R, V, B) : [153 153 161]
Pixels après : de -1.0 à 0.84 · le même pixel : [0.2  0.2  0.26]
```

Pour le jeu d'exemple, l'originale est déjà en 256 × 256 (le script l'a préparée) ; pour vos photos, vous verrez la taille de votre appareil, par exemple 4 032 × 3 024 px.

### Bloc 3b · Augmenter : six variantes d'une même photo

*Étape « + » du cours.* **Augmenter**, c'est montrer au modèle, à chaque époque, une version un peu transformée de chaque photo : retournée, tournée, zoomée, éclaircie. Il apprend ainsi que ces changements ne comptent pas. La cellule construit l'augmentation **à partir de la liste `AUGMENTATIONS` de RÉGLAGES**, puis montre six variantes de la même photo.

```python
transformations = [layers.Identity()]
if AUGMENTATIONS["miroir"]:
    transformations.append(layers.RandomFlip("horizontal"))
if AUGMENTATIONS["rotation"]:
    transformations.append(layers.RandomRotation(0.03))           # 0,03 tour = environ 10 degrés
if AUGMENTATIONS["zoom"]:
    transformations.append(layers.RandomZoom(0.1))
if AUGMENTATIONS["luminosite"]:
    transformations.append(layers.RandomBrightness(0.1, value_range=(0, 255)))
if AUGMENTATIONS["couleur"]:
    transformations.append(layers.RandomHue(0.1, value_range=(0, 255)))
augmenter = keras.Sequential(transformations, name="augmentation")
print("Augmentations actives :", [nom for nom, oui in AUGMENTATIONS.items() if oui])

plt.figure(figsize=(12, 4))
for i in range(6):
    plt.subplot(1, 6, i + 1)
    plt.imshow(np.clip(augmenter(carree[None].astype("float32"), training=True)[0], 0, 255).astype("uint8"))
    plt.title(f"variante {i + 1}")
    plt.axis("off")
plt.show()

# l'augmentation est branchée sur l'ENTRAÎNEMENT seulement
apprentissage_augmente = apprentissage.map(lambda x, y: (augmenter(x, training=True), y)).prefetch(tf.data.AUTOTUNE)
```

**Résultat attendu**

```text
Augmentations actives : ['miroir', 'rotation', 'zoom', 'luminosite']
```

Et six photos presque pareilles : retournées ou non, un peu tournées, un peu zoomées, un peu plus claires ou plus sombres.

### Bloc 3c · Vérifier : l'examen reste brut

**L'augmentation se fait à la volée, pendant l'entraînement, sur l'entraînement seulement.** Aucune photo n'est ajoutée au dossier : il ne grossit pas. Chaque époque, chaque photo d'entraînement est transformée au hasard, différemment. **Les photos d'examen restent brutes** : on juge le modèle sur des photos comme celles du terrain. La cellule le vérifie : en haut, un lot d'entraînement tel que le modèle le voit ; en bas, un lot de validation.

```python
for titre, partie in [("Entraînement : photos augmentées", apprentissage_augmente), ("Validation : photos brutes", validation)]:
    x, y = next(iter(partie))
    plt.figure(figsize=(12, 2.5))
    plt.suptitle(titre)
    for i in range(6):
        plt.subplot(1, 6, i + 1)
        plt.imshow(np.clip(x[i], 0, 255).astype("uint8"))
        plt.title(CLASSES[int(y[i])])
        plt.axis("off")
    plt.show()
```

**Résultat attendu :** en haut, des photos parfois inclinées, avec des coins noirs ou étirés, signes de l'augmentation ; en bas, des photos nettes et droites.

### Q5 · Quelle augmentation interdire ?

**Jamais une transformation qui change la classe.** Pour le corail, changer la couleur transformerait un corail sain en corail pâle : `"couleur": False`.

**Questions**
- Pour le jeu d'exemple (marguerite, tournesol, tulipe), quelle augmentation faudrait-il interdire, et pourquoi ?
- Et pour vos classes à vous ?

## Bloc 4 · Importer et créer le modèle

*Les quatre verbes, 1 : importer et créer.* **Un neurone, c'est la régression logistique de la séance 6** : il pèse ses entrées et donne une probabilité. Un **réseau** empile des neurones en couches. Pour les images, ce sont des **filtres qui glissent** sur la photo (un **réseau convolutif**) : les premières couches trouvent des traits, les suivantes des textures, les dernières des objets. **La dernière couche donne une probabilité par classe.**

**On ne part jamais de zéro.** On réutilise **MobileNetV2**, un réseau déjà entraîné sur **ImageNet** (1,4 million de photos, 1 000 classes : chiens, voitures, fleurs…). Ses filtres savent déjà voir des traits, des textures et des formes. On **gèle ses poids** (`trainable = False` : on n'y touche plus) et on remplace sa **dernière couche** par une neuve, à **K sorties**, une par classe : c'est elle seule qui apprend vos classes. C'est le **transfert d'apprentissage**, et c'est pour ça que 300 photos par classe suffisent.

Le modèle, couche par couche :
- `redimensionner` et `normaliser` : la photo passe en 224 × 224 et ses pixels entre −1 et 1 (bloc 3a) ;
- `mobilenetv2…` : le réseau pré-entraîné, **gelé** ; il transforme la photo en 1 280 nombres qui la décrivent ;
- `moyenne` : résume chaque carte de filtres par sa moyenne ;
- `oubli` (*Dropout*) : pendant l'entraînement seulement, éteint au hasard 20 % des 1 280 nombres, pour que le modèle ne compte pas sur un seul détail : une protection contre l'apprentissage par cœur ;
- `decision` : la **dernière couche, neuve**, avec K sorties et la fonction *softmax*, qui transforme les sorties en probabilités dont la somme fait 1.

### Bloc 4a · Créer le modèle

```python
base = keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base.trainable = False                       # les poids appris sur ImageNet sont GELÉS

entree = keras.Input(shape=(TAILLE, TAILLE, 3))
x = layers.Resizing(224, 224, name="redimensionner")(entree)
x = layers.Rescaling(1 / 127.5, offset=-1, name="normaliser")(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D(name="moyenne")(x)
x = layers.Dropout(0.2, name="oubli")(x)
sortie = layers.Dense(K, activation="softmax", name="decision")(x)   # K sorties : une probabilité par classe
model = keras.Model(entree, sortie)
model.summary()
```

**Résultat attendu**

```text
Model: "functional_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)        │ (None, 256, 256, 3)    │             0 │
│ redimensionner (Resizing)       │ (None, 224, 224, 3)    │             0 │
│ normaliser (Rescaling)          │ (None, 224, 224, 3)    │             0 │
│ mobilenetv2_1.00_224            │ (None, 7, 7, 1280)     │     2,257,984 │
│ (Functional)                    │                        │               │
│ moyenne                         │ (None, 1280)           │             0 │
│ (GlobalAveragePooling2D)        │                        │               │
│ oubli (Dropout)                 │ (None, 1280)           │             0 │
│ decision (Dense)                │ (None, 3)              │         3,843 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 2,261,827 (8.63 MB)
 Trainable params: 3,843 (15.01 KB)
 Non-trainable params: 2,257,984 (8.61 MB)
```

### Bloc 4b · Compter les poids gelés et entraînés

**Je prédis :** quelle part des poids va vraiment apprendre ? Plus de la moitié, environ 10 %, moins de 1 % ? …

```python
geles = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
entraines = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
en_clair = lambda n: f"{n:,}".replace(",", " ")   # 2257984 -> « 2 257 984 »
print("Poids gelés (appris sur ImageNet, on n'y touche pas) :", en_clair(geles))
print("Poids entraînés (la dernière couche, neuve)           :", en_clair(entraines))
print(f"Part des poids qui apprennent : {entraines / (geles + entraines):.2%}")
```

**Résultat attendu**

```text
Poids gelés (appris sur ImageNet, on n'y touche pas) : 2 257 984
Poids entraînés (la dernière couche, neuve)           : 3 843
Part des poids qui apprennent : 0.17%
```

### Q6 · On ne part jamais de zéro

**Questions**
- Combien de poids sont gelés, combien sont entraînés ? D'où vient le nombre de poids entraînés ? (Indice : 1 280 nombres en entrée de la dernière couche, K sorties, plus un par sortie.)
- Reliez ce chiffre à la diapo « on ne part jamais de zéro » : pourquoi 300 photos par classe suffisent-elles ?

## Bloc 5 · Entraîner

*Les quatre verbes, 2 : `fit`.* Une **époque**, c'est un passage du modèle sur toutes les photos d'entraînement. À la fin de chaque époque, Keras affiche l'exactitude sur l'entraînement (`accuracy`) et sur la validation (`val_accuracy`), ainsi que la **perte** (`loss`, `val_loss`) : un nombre qui mesure à quel point le modèle se trompe, et qu'il cherche à faire baisser.

- `compile` choisit la façon d'apprendre : **Adam**, une méthode qui ajuste les poids pas à pas, et l'**entropie croisée**, la perte habituelle quand on prédit une probabilité par classe.
- `EarlyStopping`, c'est **l'arrêt automatique quand la validation ne progresse plus** : si l'exactitude de validation ne s'améliore pas pendant 3 époques (`patience=3`), l'entraînement s'arrête, et le modèle **reprend les poids de sa meilleure époque** (`restore_best_weights=True`).

### Bloc 5a · Entraîner

**Je prédis :** quelle exactitude de validation à la fin ? Et combien de temps ? …

```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
arret = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1)

debut = time.time()
historique = model.fit(apprentissage_augmente, validation_data=validation, epochs=EPOQUES, callbacks=[arret])
duree = time.time() - debut
epoques_faites = len(historique.history["accuracy"])
print(f"\n{epoques_faites} époques en {duree / 60:.1f} minutes, soit {duree / epoques_faites:.0f} s par époque.")
```

**Résultat attendu** (les chiffres changent un peu d'une exécution à l'autre)

```text
Epoch 1/10
23/23 ━━━━━━━━━━━━━━━━━━━━ 12s - accuracy: 0.5642 - loss: 0.9466 - val_accuracy: 0.8202 - val_loss: 0.5163
Epoch 2/10
23/23 ━━━━━━━━━━━━━━━━━━━━ 8s - accuracy: 0.8282 - loss: 0.4664 - val_accuracy: 0.8933 - val_loss: 0.3432
Epoch 3/10
23/23 ━━━━━━━━━━━━━━━━━━━━ 8s - accuracy: 0.8994 - loss: 0.3133 - val_accuracy: 0.9270 - val_loss: 0.2651
Epoch 4/10
23/23 ━━━━━━━━━━━━━━━━━━━━ 9s - accuracy: 0.8939 - loss: 0.2826 - val_accuracy: 0.9326 - val_loss: 0.2299
Epoch 5/10
23/23 ━━━━━━━━━━━━━━━━━━━━ 11s - accuracy: 0.9190 - loss: 0.2331 - val_accuracy: 0.9326 - val_loss: 0.2131
Epoch 6/10
23/23 ━━━━━━━━━━━━━━━━━━━━ 8s - accuracy: 0.9511 - loss: 0.1927 - val_accuracy: 0.9326 - val_loss: 0.1937
Epoch 7/10
23/23 ━━━━━━━━━━━━━━━━━━━━ 7s - accuracy: 0.9427 - loss: 0.1715 - val_accuracy: 0.9326 - val_loss: 0.1860
Epoch 7: early stopping
Restoring model weights from the end of the best epoch: 4.

7 époques en 1.1 minutes, soit 9 s par époque.
```

### Bloc 5b · Les deux courbes

*Pendant l'entraînement, deux courbes* : l'entraînement et la validation. Tant que la validation monte, le modèle apprend des choses utiles. **Quand la validation cesse de monter alors que l'entraînement continue**, le modèle commence à apprendre par cœur ses photos d'entraînement : c'est le **sur-apprentissage**. Le point rouge marque la meilleure époque, celle dont l'arrêt automatique a repris les poids.

```python
h = historique.history
epoques = range(1, len(h["accuracy"]) + 1)
meilleure = int(np.argmax(h["val_accuracy"])) + 1

plt.figure(figsize=(12, 4))
for i, (mesure, nom) in enumerate([("accuracy", "Exactitude"), ("loss", "Perte")]):
    plt.subplot(1, 2, i + 1)
    plt.plot(epoques, h[mesure], "o-", color="#2a78d6", label="entraînement")
    plt.plot(epoques, h["val_" + mesure], "o-", color="#eb6834", label="validation")
    plt.plot(meilleure, h["val_" + mesure][meilleure - 1], "o", color="red", markersize=12, label=f"meilleure époque ({meilleure})")
    plt.xlabel("époque"); plt.title(nom); plt.legend()
plt.show()
print(f"Meilleure époque : {meilleure}, exactitude de validation {h['val_accuracy'][meilleure - 1]:.1%}")
```

**Résultat attendu**

```text
Meilleure époque : 4, exactitude de validation 93.3%
```

Et deux graphiques : à gauche l'exactitude, qui monte vite puis plafonne ; à droite la perte, qui descend.

### Q7 · Lire les deux courbes

**Questions**
- Les deux courbes d'exactitude se suivent-elles, se croisent-elles, s'écartent-elles ?
- À quelle époque fallait-il s'arrêter ?
- Y a-t-il sur-apprentissage ? À quoi le voyez-vous ?
- Au début, l'exactitude de validation est souvent *plus haute* que celle d'entraînement : pourquoi, d'après vous ? (Indice : sur quelles photos agissent l'augmentation et l'`oubli` ?)

## Bloc 6 · Évaluer, avec les outils de la séance 6

*Les quatre verbes, 3 : évaluer.* Les mêmes outils qu'à la séance 6, pas un de plus : l'**exactitude**, la **matrice de confusion**, le **rappel** et la **précision** par classe.

### Bloc 6a · L'exactitude sur la validation

La cellule fait prédire le modèle sur toutes les photos de validation, et garde pour chacune la vraie classe, la classe prédite et la **confiance** (la plus grande des K probabilités).

**Je prédis :** quelle exactitude ? …

```python
images_val, vraies = [], []
for x, y in validation:
    images_val.append(x.numpy().astype("uint8"))
    vraies.append(y.numpy())
images_val, vraies = np.concatenate(images_val), np.concatenate(vraies)

probas = model.predict(images_val, verbose=0)
predites, confiances = probas.argmax(axis=1), probas.max(axis=1)
exactitude = (predites == vraies).mean()
print(f"Exactitude sur la validation : {exactitude:.1%} ({(predites == vraies).sum()} photos justes sur {len(vraies)})")
print(f"Meilleure époque de l'entraînement : {max(historique.history['val_accuracy']):.1%} : ce sont bien ses poids.")
```

**Résultat attendu**

```text
Exactitude sur la validation : 93.3% (166 photos justes sur 178)
Meilleure époque de l'entraînement : 93.3% : ce sont bien ses poids.
```

Les deux nombres sont égaux : l'arrêt automatique a bien rendu les poids de la meilleure époque.

### Bloc 6b · La matrice de confusion

En **lignes**, la classe **vraiment** ; en **colonnes**, la classe **prédite**. La diagonale compte les photos bien classées ; tout le reste, les confusions. Les classes sont dans l'ordre des sous-dossiers.

```python
matrice = confusion_matrix(vraies, predites, labels=range(K))
affichage = ConfusionMatrixDisplay(matrice, display_labels=CLASSES)
affichage.plot(cmap="Blues", colorbar=False)
affichage.ax_.set_xlabel("prédit")
affichage.ax_.set_ylabel("vraiment")
affichage.ax_.set_title("Matrice de confusion (validation)")
plt.show()
```

**Résultat attendu** (la cellule affiche ce tableau en image, en dégradé de bleu)

```text
                 prédit
vraiment     marguerite  tournesol  tulipe
marguerite           58          2       1
tournesol             3         57       2
tulipe                0          4      51
```

### Bloc 6c · Rappel et précision, classe par classe

- **Rappel** d'une classe : parmi les photos **vraiment** de cette classe (sa ligne), la part que le modèle a trouvée.
- **Précision** d'une classe : parmi les photos **prédites** de cette classe (sa colonne), la part qui l'étaient vraiment.

```python
rapport = classification_report(vraies, predites, labels=range(K), target_names=CLASSES, output_dict=True, zero_division=0)
rapport = pd.DataFrame(rapport).T.loc[CLASSES, ["precision", "recall", "support"]]
rapport.columns = ["précision", "rappel", "photos"]
rapport["précision"], rapport["rappel"] = rapport["précision"].round(2), rapport["rappel"].round(2)
rapport["photos"] = rapport["photos"].astype(int)
display(rapport)
```

**Résultat attendu**

```text
            précision  rappel  photos
marguerite       0.95    0.95      61
tournesol        0.90    0.92      62
tulipe           0.94    0.93      55
```

### Bloc 6d · Les photos mal classées

Regarder les erreurs, c'est souvent là qu'on comprend le plus. Au-dessus de chaque photo : vraie classe → classe prédite (confiance).

```python
erreurs = np.where(predites != vraies)[0]
print(len(erreurs), "photos mal classées sur", len(vraies))
plt.figure(figsize=(15, 3 * ((min(len(erreurs), 12) + 5) // 6)))
for i, j in enumerate(erreurs[:12]):
    plt.subplot((min(len(erreurs), 12) + 5) // 6, 6, i + 1)
    plt.imshow(images_val[j])
    plt.title(f"{CLASSES[vraies[j]]} → {CLASSES[predites[j]]} ({confiances[j]:.0%})", fontsize=9)
    plt.axis("off")
plt.show()
```

**Résultat attendu**

```text
12 photos mal classées sur 178
```

Et une grille des erreurs (12 au plus).

### Q8 · Quelle confusion ?

**Questions**
- Quelle classe est la plus souvent confondue, et avec laquelle ?
- En regardant les photos mal classées, est-ce une erreur qu'un humain ferait aussi ?

### Q9 · Quelle erreur coûte cher ?

**Questions**
- Dans votre problème (corail, déchets, miconia, ou le vôtre), quelle erreur coûte le plus cher ? Par exemple, dire « sain » à un corail blanchi, ou « autre végétation » à un miconia.
- Quel rappel surveillez-vous en priorité ?

### Q10 · La baseline

Un modèle « paresseux » répond **toujours la classe la plus fréquente** de l'entraînement, sans regarder la photo. C'est la **baseline** : un vrai modèle doit la battre nettement.

```python
majoritaire = par_classe(apprentissage).idxmax()
exactitude_paresseux = (vraies == CLASSES.index(majoritaire)).mean()
print("Classe la plus fréquente à l'entraînement :", majoritaire)
print(f"Exactitude du modèle paresseux : {exactitude_paresseux:.1%}")
print(f"Exactitude de notre modèle     : {exactitude:.1%}")
```

**Résultat attendu**

```text
Classe la plus fréquente à l'entraînement : tulipe
Exactitude du modèle paresseux : 30.9%
Exactitude de notre modèle     : 93.3%
```

**Questions**
- Votre modèle bat-il nettement la baseline ? De combien de points ?
- Avec K classes équilibrées, quelle est à peu près l'exactitude du modèle paresseux ? Et si une classe faisait 90 % des photos ?

## Bloc 7 · Prédire

*Les quatre verbes, 4 : `predict`.* Pour une photo nouvelle, le modèle donne **une probabilité par classe**. On retient la plus grande, mais **si elle est sous le seuil de confiance** (`SEUIL` dans RÉGLAGES), le modèle hésite : la photo part **« à vérifier par un humain »**.

### Bloc 7a · Une photo nouvelle

La cellule récupère le dossier `PHOTOS_NOUVELLES`, prend la photo choisie, et affiche ses K probabilités. Pour Q11, changez le numéro trois fois, ou tapez `"televerser"` pour envoyer une photo de votre ordinateur (une photo prise avec votre téléphone, par exemple).

<!-- id: une_photo -->
```python
CHOIX = 1   # À MODIFIER pour Q11 : le numéro d'une photo du dossier PHOTOS_NOUVELLES (1, 2, 3…), ou "televerser"

NOUVELLES = recuperer(PHOTOS_NOUVELLES, "nouvelles")
if CHOIX == "televerser":
    from google.colab import files
    chemin = Path(list(files.upload())[0])
else:
    chemin = photos_de(NOUVELLES)[CHOIX - 1]

photo = lire_photo(chemin)
p = model.predict(photo[None], verbose=0)[0]
classe, confiance = CLASSES[p.argmax()], p.max()
decision = classe if confiance >= SEUIL else "à vérifier par un humain"

plt.figure(figsize=(10, 3.5))
plt.subplot(1, 2, 1); plt.imshow(photo); plt.axis("off"); plt.title(chemin.name)
plt.subplot(1, 2, 2); plt.barh(CLASSES, p, color="#2a78d6"); plt.xlim(0, 1)
plt.axvline(SEUIL, color="#eb6834", linestyle="--", label=f"seuil {SEUIL}"); plt.legend()
plt.title(f"{classe} ({confiance:.0%}) → {decision}")
plt.show()
print({c: round(float(v), 3) for c, v in zip(CLASSES, p)})
print("Décision :", decision)
```

**Résultat attendu**, pour `CHOIX = 16` (une rose, une fleur que le modèle ne connaît pas)

```text
{'marguerite': 0.006, 'tournesol': 0.028, 'tulipe': 0.966}
Décision : tulipe
```

Et la photo, à côté de ses K probabilités en barres horizontales, avec le seuil en pointillés.

### Bloc 7b · Tout un dossier de photos nouvelles

La même chose sur **toutes** les photos du dossier, d'un coup. Le résultat est un tableau `fichier, classe, confiance, decision`, enregistré dans `predictions.csv`, puis compté avec `value_counts()`. **C'est la statistique du cas 2** (combien de photos de chaque type de déchet ?) et le pont avec les séances 4 et 5 : ce tableau se nettoie, s'analyse et se trace comme n'importe quel autre.

**Je prédis :** sur les photos nouvelles, combien partiront « à vérifier » ? …

```python
fichiers = photos_de(NOUVELLES)
p = model.predict(np.stack([lire_photo(f) for f in fichiers]), verbose=0)
resultats = pd.DataFrame({"fichier": [f.name for f in fichiers],
                          "classe": [CLASSES[i] for i in p.argmax(axis=1)],
                          "confiance": p.max(axis=1).round(3)})
resultats["decision"] = np.where(resultats["confiance"] >= SEUIL, resultats["classe"], "à vérifier par un humain")
resultats.to_csv("predictions.csv", index=False)
display(resultats.head(10))
print(resultats["decision"].value_counts())
```

**Résultat attendu**

```text
           fichier      classe  confiance    decision
0  nouvelle_01.jpg  marguerite      0.920  marguerite
1  nouvelle_02.jpg   tournesol      0.917   tournesol
2  nouvelle_03.jpg  marguerite      0.849  marguerite
…
decision
marguerite                  10
tulipe                      10
tournesol                    9
à vérifier par un humain     3
```

### Q11 · Trois photos nouvelles

Pour le jeu d'exemple, la vraie fleur de chaque photo nouvelle est dans le fichier `etiquettes.csv` du dossier `nouvelles/photos_nouvelles/` (ouvrez-le dans le panneau Fichiers de Colab, à gauche). Attention : 8 de ces photos sont des **roses** et des **pissenlits**, deux fleurs que le modèle n'a jamais vues.

**Questions**
- Choisissez trois photos nouvelles, dont une difficile (une rose ou un pissenlit pour l'exemple ; une photo floue ou ambiguë pour vous). Le modèle a-t-il raison ? Avec quelle confiance ?
- Que répond le modèle devant une fleur qu'il ne connaît pas ? Pourquoi ne peut-il pas répondre « je ne sais pas » tout seul ?

### Q12 · Choisir le seuil

La cellule compte les photos qui partiraient « à vérifier » pour trois seuils.

```python
for s in [0.5, 0.6, 0.8]:
    print(f"Seuil {s} : {(resultats['confiance'] < s).sum():>3} photos à vérifier sur {len(resultats)}")
```

**Résultat attendu**

```text
Seuil 0.5 :   0 photos à vérifier sur 32
Seuil 0.6 :   3 photos à vérifier sur 32
Seuil 0.8 :   9 photos à vérifier sur 32
```

**Questions**
- Combien de photos partent « à vérifier » pour 0,5, 0,6 et 0,8 ?
- Pour l'association du cas 2, qui trie des milliers de photos de déchets avec deux bénévoles, quel seuil choisiriez-vous ? Qu'est-ce qu'on gagne et qu'est-ce qu'on perd en le montant ?

## Bloc 8 · Où regarde le modèle ?

Un bon score ne suffit pas : il faut **vérifier où regarde le modèle**. La **carte de chaleur** (méthode *Grad-CAM*) colore en rouge les zones de la photo qui ont le plus pesé dans sa décision. Elle se calcule sur la **dernière couche de filtres** de MobileNetV2 : on regarde comment la probabilité de la classe choisie change quand chaque zone change. **S'il regarde le fond, le site ou le matériel** (un cadre, une règle, la couleur du seau), c'est un **raccourci** : il a trouvé un indice qui marche sur vos photos, mais pas sur le terrain. Retour à l'étape 0.

### Bloc 8 · Six cartes de chaleur

La cellule est prête à l'emploi : exécutez-la sans la modifier.

```python
def carte_chaleur(photo):
    """La carte de chaleur Grad-CAM d'une photo, sur la dernière couche de filtres de MobileNetV2."""
    x = model.get_layer("normaliser")(model.get_layer("redimensionner")(photo[None].astype("float32")))
    with tf.GradientTape() as bande:
        filtres = base(x, training=False)            # la sortie de la dernière couche de filtres
        bande.watch(filtres)
        p = model.get_layer("decision")(model.get_layer("moyenne")(filtres))
        choisie = int(tf.argmax(p[0]))
        score = p[:, choisie]
    gradients = bande.gradient(score, filtres)
    poids = tf.reduce_mean(gradients, axis=(1, 2))   # l'importance de chaque filtre
    carte = tf.nn.relu(tf.reduce_sum(filtres * poids[:, None, None, :], axis=-1))[0]
    carte = carte / (tf.reduce_max(carte) + 1e-8)
    return tf.image.resize(carte[..., None], (TAILLE, TAILLE))[..., 0].numpy(), choisie, float(p[0, choisie])

plt.figure(figsize=(15, 5.5))
for i, j in enumerate(np.random.default_rng(42).choice(len(images_val), 6, replace=False)):
    carte, choisie, confiance = carte_chaleur(images_val[j])
    plt.subplot(2, 6, i + 1); plt.imshow(images_val[j]); plt.axis("off")
    plt.title(f"vraiment : {CLASSES[vraies[j]]}", fontsize=9)
    plt.subplot(2, 6, i + 7); plt.imshow(images_val[j]); plt.imshow(carte, cmap="jet", alpha=0.45); plt.axis("off")
    plt.title(f"prédit : {CLASSES[choisie]} ({confiance:.0%})", fontsize=9)
plt.show()
```

**Résultat attendu :** en haut six photos de validation, en bas les mêmes avec leur carte de chaleur, du bleu (sans importance) au rouge (décisif).

### Q13 · Sur le sujet ou sur le fond ?

**Questions**
- Les zones rouges sont-elles sur le sujet (la fleur, le corail, le déchet) ou sur le fond ?
- Si c'est le fond, quelle étape du preprocessing faut-il reprendre, et comment ? (Cours : étape 0, collecter des photos plus variées.)

## Bloc 9 · Si le résultat déçoit, si le résultat est trop beau

### Diagnostic 1 · Si le résultat déçoit

Le diagnostic du cours, **dans cet ordre** :

1. **Vérifier les étiquettes** : regardez les photos mal classées (bloc 6d). Beaucoup sont-elles dans le mauvais dossier ? Corrigez le dossier, pas le modèle.
2. **Ajouter de la variété** : d'autres sites, d'autres heures, d'autres appareils, d'autres angles. Retour à l'étape 0.
3. **Augmenter davantage** : activez une augmentation de plus dans RÉGLAGES, si elle ne change pas la classe.
4. **Seulement ensuite, dégeler les dernières couches** du réseau pré-entraîné (le *fine-tuning*, cellule bonus ci-dessous).

### Diagnostic 2 · Si le résultat est trop beau

**99 % dès la 2ᵉ époque ?** Méfiez-vous avant de vous réjouir. Deux causes fréquentes :

- **une fuite entre entraînement et validation** : des photos en double, ou en rafale, réparties des deux côtés (bloc 2, Q4) ; le modèle « reconnaît » au lieu de « classer » ;
- **un raccourci** : toutes les photos d'une classe ont le même fond, le même site, le même appareil ; la carte de chaleur (bloc 8) le montre.

### Bloc 9 · BONUS : dégeler les 20 dernières couches (fine-tuning)

**À n'exécuter qu'en dernier recours**, quand le reste a été essayé : on **dégèle les 20 dernières couches** de MobileNetV2 et on réentraîne avec un **pas d'apprentissage dix fois plus petit** (0,0001 au lieu de 0,001), pour ajuster doucement ses filtres à vos photos sans effacer ce qu'il a appris sur ImageNet. Cela coûte des minutes de GPU en plus, et le risque d'apprendre par cœur augmente. Pour l'exécuter, mettez `FINE_TUNING = True`.

<!-- id: fine_tuning -->
```python
FINE_TUNING = False   # DÉCISION : True pour lancer le fine-tuning

if FINE_TUNING:
    avant = max(historique.history["val_accuracy"])
    base.trainable = True
    for couche in base.layers[:-20]:
        couche.trainable = False             # tout reste gelé, sauf les 20 dernières couches
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print("Poids entraînés maintenant :", en_clair(sum(int(np.prod(w.shape)) for w in model.trainable_weights)))
    debut = time.time()
    historique_fin = model.fit(apprentissage_augmente, validation_data=validation, epochs=5,
                               callbacks=[keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)])
    duree_fin = time.time() - debut
    apres = max(historique_fin.history["val_accuracy"])

    hf = historique_fin.history
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(hf["accuracy"]) + 1), hf["accuracy"], "o-", color="#2a78d6", label="entraînement")
    plt.plot(range(1, len(hf["accuracy"]) + 1), hf["val_accuracy"], "o-", color="#eb6834", label="validation")
    plt.axhline(avant, color="gray", linestyle="--", label="validation avant fine-tuning")
    plt.xticks(range(1, len(hf["accuracy"]) + 1))
    plt.xlabel("époque de fine-tuning"); plt.title("Exactitude pendant le fine-tuning"); plt.legend()
    plt.show()
    print(f"Validation : {avant:.1%} avant, {apres:.1%} après ({(apres - avant) * 100:+.1f} points), "
          f"pour {duree_fin / 60:.1f} minutes de plus.")
else:
    print("Fine-tuning non lancé : le modèle reste celui du bloc 5.")
```

**Résultat attendu**

```text
Fine-tuning non lancé : le modèle reste celui du bloc 5.
```

Si vous le lancez, une courbe de 5 époques au plus, et une ligne qui compare la validation avant et après.

### Q14 (bonus) · Le fine-tuning valait-il le coup ?

**Questions**
- Le fine-tuning a-t-il amélioré la validation ? De combien de points ?
- Cela valait-il les minutes de GPU en plus ?

## Bloc 10 · Sauvegarder et documenter

Un modèle sans sa fiche, c'est un tableau propre sans son journal de nettoyage : personne ne peut savoir ce qu'il vaut ni le refaire. La cellule enregistre le **modèle** (`modele_photos.keras`), le tableau des **prédictions** et une **fiche du modèle** (`fiche_modele.md`) dans votre Drive : c'est **le journal de la séance 5, appliqué à un modèle**.
### Bloc 10 · Le modèle, ses prédictions et sa fiche

Complétez d'abord les **limites connues** : les sites, les appareils et les conditions que le modèle a vus, et ceux qu'il n'a pas vus.

<!-- id: fiche -->
```python
LIMITES = ""   # DÉCISION : les sites, appareils et conditions vus, et ceux qui ne l'ont pas été

if EN_COLAB and DOSSIER_SORTIE.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
    SORTIE = Path(DOSSIER_SORTIE)
else:
    SORTIE = Path("sorties")
SORTIE.mkdir(parents=True, exist_ok=True)

# on réévalue le modèle tel qu'il est maintenant (avec ou sans fine-tuning)
predites_fin = model.predict(images_val, verbose=0).argmax(axis=1)
rapport_fin = pd.DataFrame(classification_report(vraies, predites_fin, labels=range(K), target_names=CLASSES,
                                                 output_dict=True, zero_division=0)).T.loc[CLASSES]
lignes_classes = "\n".join(f"| {c} | {gardees[c]} | {rapport_fin.loc[c, 'recall']:.2f} | {rapport_fin.loc[c, 'precision']:.2f} |"
                           for c in CLASSES)
fiche = f"""# Fiche du modèle : {', '.join(CLASSES)}

- **Date :** {date.today():%d/%m/%Y}
- **Photos :** `{PHOTOS}`
- **Modèle :** MobileNetV2 pré-entraîné sur ImageNet, poids gelés{', 20 dernières couches dégelées (fine-tuning)' if FINE_TUNING else ''} ; dernière couche neuve à {K} sorties
- **Préparation :** photos recadrées en carré, {TAILLE} x {TAILLE} px ; {len(ecartees)} photo(s) écartée(s) à la main, {len(abimees)} illisible(s)
- **Partage :** {len(apprentissage.file_paths)} photos pour l'entraînement, {len(validation.file_paths)} pour la validation ({'par site : ' + ', '.join(SITES_VALIDATION) if SITES_VALIDATION else 'photo par photo, graine 42'})
- **Augmentations :** {', '.join(n for n, oui in AUGMENTATIONS.items() if oui) or 'aucune'} (interdites : {', '.join(n for n, oui in AUGMENTATIONS.items() if not oui) or 'aucune'})
- **Entraînement :** {epoques_faites} époques sur {EPOQUES} au plus, meilleure époque {int(np.argmax(historique.history['val_accuracy'])) + 1}, {duree / 60:.1f} minutes
- **Exactitude sur la validation :** {(predites_fin == vraies).mean():.1%} (modèle paresseux : {exactitude_paresseux:.1%})
- **Seuil de confiance :** {SEUIL} ; en dessous, la photo est à vérifier par un humain

| Classe | Photos | Rappel | Précision |
|---|---|---|---|
{lignes_classes}

**Limites connues**

{LIMITES or "À COMPLÉTER : les sites, appareils et conditions vus, et ceux qui ne l'ont pas été."}

Le modèle propose, un humain décide.
"""
model.save(SORTIE / "modele_photos.keras")
resultats.to_csv(SORTIE / "predictions.csv", index=False)
(SORTIE / "fiche_modele.md").write_text(fiche, encoding="utf-8")
print("Enregistrés dans", SORTIE, ":", sorted(f.name for f in SORTIE.iterdir()))
print()
print(fiche)
```

**Résultat attendu**

```text
Enregistrés dans /content/drive/MyDrive/TP_photos : ['fiche_modele.md', 'modele_photos.keras', 'predictions.csv']

# Fiche du modèle : marguerite, tournesol, tulipe

- **Date :** …
- **Photos :** `https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/images_exemple.zip`
- **Modèle :** MobileNetV2 pré-entraîné sur ImageNet, poids gelés ; dernière couche neuve à 3 sorties
- **Préparation :** photos recadrées en carré, 256 x 256 px ; 6 photo(s) écartée(s) à la main, 0 illisible(s)
- **Partage :** 716 photos pour l'entraînement, 178 pour la validation (photo par photo, graine 42)
- **Augmentations :** miroir, rotation, zoom, luminosite (interdites : couleur)
- **Entraînement :** 7 époques sur 10 au plus, meilleure époque 4, 1.1 minutes
- **Exactitude sur la validation :** 93.3% (modèle paresseux : 30.9%)
- **Seuil de confiance :** 0.6 ; en dessous, la photo est à vérifier par un humain
…
```

Pour réutiliser le modèle plus tard, sans le réentraîner : `model = keras.models.load_model("/content/drive/MyDrive/TP_photos/modele_photos.keras")`, puis les cellules du bloc 7.

### Q15 · Trois phrases pour un élu

**Les limites du cours** : un modèle ne se généralise pas tout seul à un autre site ou un autre appareil (il faut le tester là-bas) ; il hérite des **biais de la collecte** ; il coûte du **GPU** ; les photos peuvent toucher à la **vie privée** (des personnes, des plaques d'immatriculation) ; et **l'humain signe la décision**.

**Questions**
- Écrivez en trois phrases, pour un élu : ce que le modèle sait faire, ce qu'il ne sait pas faire, et qui décide à la fin.

## Ce qu'il faut rendre

Un notebook Colab par binôme, avec les deux noms en haut. Il doit contenir :
- toutes les cellules, **exécutées**, avec leurs résultats visibles ;
- vos prédictions (« Je prédis ») et vos réponses aux questions Q1 à Q15 ;
- vos décisions et leurs justifications dans les cellules DÉCISION ;
- la fiche `fiche_modele.md` produite à la fin, avec ses limites complétées.

Partagez-le avec votre enseignant via « Partager » dans Colab, ou téléchargez-le avec Fichier > Télécharger > .ipynb.

## Testé sur d'autres dossiers

Pour vérifier que le carnet est réutilisable, le carnet ÉLÈVE a été exécuté de bout en bout sur trois autres dossiers, fabriqués avec le même script à partir des autres fleurs du jeu source, **en ne changeant que la cellule RÉGLAGES** (et `EPOQUES = 2`, pour aller vite) :

| Dossier | Classes | Partage | Exactitude (2 époques) | Modèle paresseux |
|---|---|---|---|---|
| 2 classes | `marguerite`, `tournesol` | 480 / 120 | 90,0 % | 49,2 % |
| 4 classes | `marguerite`, `rose`, `tournesol`, `tulipe` | 960 / 240 | 87,5 % | 21,2 % |
| 2 classes, rangées par site (`classe/site_A/…` à `site_E`) | `marguerite`, `tournesol` | par site : `SITES_VALIDATION = ["site_B"]`, 494 / 106 | 89,6 % | 47,2 % |

Toutes les cellules tournent, y compris le fine-tuning et la fiche : le nombre de classes, leurs noms, la taille de la dernière couche (2 562 poids pour 2 classes, 5 124 pour 4), la matrice de confusion et le tableau des probabilités suivent seuls.

## Glossaire

| Mot | En une phrase |
|---|---|
| **Pixel** | Un point de la photo, décrit par 3 nombres entre 0 et 255 (rouge, vert, bleu). Les features sont des pixels. |
| **Classe, étiquette** | Ce qu'on veut prédire ; ici, le nom du dossier de la photo. |
| **Neurone** | Une régression logistique : il pèse ses entrées et donne une probabilité. |
| **Réseau convolutif** | Un réseau de neurones fait de filtres qui glissent sur la photo : traits, puis textures, puis objets. |
| **Transfert d'apprentissage** | Réutiliser un réseau déjà entraîné (MobileNetV2, sur ImageNet), poids gelés, et ne remplacer que la dernière couche. |
| **Poids gelés** | Les poids qu'on n'entraîne plus (`trainable = False`). |
| **Époque** | Un passage du modèle sur toutes les photos d'entraînement. |
| **Lot** | Un paquet de photos (ici 32) que le modèle regarde en même temps. |
| **Validation** | Les 20 % de photos mises de côté avant tout, pour l'examen. |
| **Augmenter** | Transformer un peu les photos d'entraînement, à la volée, sans changer leur classe. |
| **Perte** | Un nombre qui mesure à quel point le modèle se trompe ; il cherche à le faire baisser. |
| **Sur-apprentissage** | Apprendre par cœur : l'entraînement progresse, la validation non. |
| **Arrêt automatique (EarlyStopping)** | Arrêter quand la validation ne progresse plus, et reprendre les poids de la meilleure époque. |
| **Confiance** | La plus grande des probabilités données par le modèle. |
| **Seuil de confiance** | En dessous, la photo va « à vérifier par un humain ». |
| **Carte de chaleur (Grad-CAM)** | Les zones de la photo qui ont le plus pesé dans la décision. |
| **Raccourci** | Le modèle a appris un indice du fond, du site ou du matériel, pas le sujet. |
| **Fine-tuning** | Dégeler les dernières couches du réseau pré-entraîné et les réentraîner doucement. |
| **GPU** | Une carte de calcul qui accélère beaucoup l'entraînement des réseaux. |
