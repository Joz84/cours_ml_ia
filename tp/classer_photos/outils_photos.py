"""Boîte à outils du TP « Classer des photos » (UE 5.7 « Data Science et IA », UPF).

Une fonction par étape du cours. Chaque fonction affiche ce qu'il faut voir, en
français, et range ses résultats dans un objet `projet` qu'on passe d'une étape à
l'autre. Le nombre de classes et leurs noms ne sont écrits nulle part : ils sont lus
dans les noms des sous-dossiers.

    projet = lire_photos(PHOTOS, TAILLE)        étape 0 : collecter et étiqueter
    montrer_planches(projet)                    étape 2 : nettoyer (regarder)
    ecarter(projet, A_ECARTER)                  étape 2 : nettoyer (écarter, sans toucher à l'original)
    separer(projet, PART_VALIDATION)            étape 1 : séparer 80 / 20
    avant_apres(projet)                         étape 3 : redimensionner, normaliser
    augmenter(projet, AUGMENTATIONS)            étape + : augmenter, sur l'entraînement seulement
    creer_modele(projet)                        verbe 1 : importer et créer
    entrainer(projet, EPOQUES)                  verbe 2 : fit, et les deux courbes
    evaluer(projet)                             verbe 3 : évaluer
    predire_une_photo(projet, ...)              verbe 4 : predict, avec le seuil de confiance
    predire_dossier(projet, ...)                verbe 4 : sur tout un dossier
    cartes_de_chaleur(projet)                   où regarde le modèle ?
    enregistrer(projet, ...)                    sauvegarder et documenter
    fine_tuning(projet)                         bonus : dégeler les dernières couches

La version « exhaustive » du TP (tp/classification_images/) montre tout ce code
cellule par cellule.

Utilisation dans Colab :

    import urllib.request
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/tp/classer_photos/outils_photos.py",
        "outils_photos.py")
    from outils_photos import *
"""
import contextlib
import io
import os
import shutil
import sys
import time
import urllib.request
import warnings
import zipfile
from datetime import date
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # moins de messages techniques de TensorFlow

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

warnings.filterwarnings("ignore", message=".*shuffle=True.*")   # un avertissement de Keras sans conséquence

EN_COLAB = "google.colab" in sys.modules
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")   # les formats lus
BLEU, ORANGE = "#2a78d6", "#eb6834"
LOT = 32                                                 # les photos passent dans le modèle par lots de 32

__all__ = ["verifier_gpu", "lire_photos", "montrer_planches", "ecarter", "separer", "avant_apres", "augmenter",
           "creer_modele", "entrainer", "evaluer", "predire_une_photo", "predire_dossier", "cartes_de_chaleur",
           "enregistrer", "fine_tuning"]


class Projet:
    """Tout ce que le carnet apprend au fil des étapes : photos, partage, modèle, résultats."""

    def __repr__(self):
        return f"Projet de classification de photos : {len(self.classes)} classes {self.classes}"


# ---------------------------------------------------------------- outils internes

def _recuperer(source, nom):
    """Rend un dossier de photos : télécharge et dézippe un zip, ou monte le Drive."""
    source = str(source)
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
    if not dossier.is_dir():
        raise FileNotFoundError(f"Le dossier « {source} » n'existe pas. Vérifiez le chemin dans RÉGLAGES.")
    while True:   # un zip contient souvent un seul dossier : on descend dedans
        sous = [d for d in dossier.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))]
        images = [f for f in dossier.iterdir() if f.suffix.lower() in EXTENSIONS]
        if len(sous) == 1 and not images:
            dossier = sous[0]
        else:
            return dossier


def _photos_de(dossier):
    return sorted(f for f in Path(dossier).rglob("*") if f.suffix.lower() in EXTENSIONS)


def _lire_photo(chemin, taille):
    """Ouvre une photo, la recadre en carré au centre, la met en taille x taille."""
    photo = keras.utils.load_img(chemin, target_size=(taille, taille), keep_aspect_ratio=True)
    return keras.utils.img_to_array(photo).astype("uint8")


def _classe_de(chemin, classes):
    return next(d for d in reversed(Path(chemin).parts[:-1]) if d in classes)


def _par_classe(partie, classes):
    return pd.Series([_classe_de(f, classes) for f in partie.file_paths]).value_counts().reindex(classes, fill_value=0)


def _pourcent(x):
    return f"{x * 100:.1f} %".replace(".", ",")


def _en_clair(n):
    return f"{n:,}".replace(",", " ")


def _verifier(projet, etape, attribut):
    if not hasattr(projet, attribut):
        raise RuntimeError(f"Exécutez d'abord l'étape « {etape} » (les cellules s'exécutent dans l'ordre).")


# ---------------------------------------------------------------- étape 0

def verifier_gpu():
    """Dit si Colab prête un GPU."""
    if tf.config.list_physical_devices("GPU"):
        print("GPU trouvé : l'entraînement prendra quelques minutes.")
    else:
        print("ATTENTION : pas de GPU. L'entraînement marchera, mais prendra une heure au lieu de quelques minutes "
              "sur un gros dossier de photos.\nMenu Exécution → Modifier le type d'exécution → GPU, "
              "puis relancez le carnet depuis le début.")


def lire_photos(source, taille=256):
    """Étape 0 · Collecter et étiqueter : lit les sous-dossiers (les classes) et compte les photos."""
    keras.utils.set_random_seed(42)   # le même hasard à chaque fois qu'on reprend du début
    projet = Projet()
    projet.source, projet.taille = str(source), taille
    projet.dossier = _recuperer(source, "mes_photos")
    projet.classes = sorted(d.name for d in projet.dossier.iterdir() if d.is_dir() and not d.name.startswith((".", "_")))
    if len(projet.classes) < 2:
        raise ValueError(f"Il faut au moins 2 sous-dossiers, un par classe, dans « {projet.dossier} ». "
                         f"Trouvés : {projet.classes}")
    projet.fichiers = {c: _photos_de(projet.dossier / c) for c in projet.classes}
    comptes = pd.DataFrame({
        "classe": projet.classes,
        "photos": [len(projet.fichiers[c]) for c in projet.classes],
        "fichiers non lus": [sum(1 for f in (projet.dossier / c).rglob("*") if f.is_file() and f.suffix.lower() not in EXTENSIONS)
                             for c in projet.classes]})
    print(len(projet.classes), "classes, lues dans les noms des sous-dossiers :", projet.classes)
    print("Photos en tout :", comptes["photos"].sum())
    print(comptes.to_string(index=False))
    if comptes["fichiers non lus"].sum():
        print("ATTENTION : des fichiers ne sont pas des photos lisibles (HEIC, vidéos, documents…) : exportez-les en JPEG.")
    petites = comptes.loc[comptes["photos"] < comptes["photos"].max() / 2, "classe"].tolist()
    if petites:
        print("ATTENTION : moins de la moitié des photos des autres classes pour :", petites)

    plt.figure(figsize=(6, 2.5))
    plt.bar(comptes["classe"], comptes["photos"], color=BLEU)
    plt.title("Nombre de photos par classe")
    plt.ylabel("photos")
    plt.show()

    toutes = [(f, c) for c in projet.classes for f in projet.fichiers[c]]
    plt.figure(figsize=(8, 8))
    for i, k in enumerate(np.random.default_rng(42).choice(len(toutes), 9, replace=False)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(_lire_photo(toutes[k][0], taille))
        plt.title(toutes[k][1])
        plt.axis("off")
    plt.suptitle("Neuf photos au hasard, avec leur étiquette (le nom de leur dossier)")
    plt.show()
    return projet


# ---------------------------------------------------------------- étape 2 : nettoyer

def montrer_planches(projet, tirage=1):
    """Étape 2 · Nettoyer : 20 photos au hasard par classe, chacune avec son numéro."""
    rng = np.random.default_rng(tirage)
    for c in projet.classes:
        fichiers = projet.fichiers[c]
        choix = sorted(rng.choice(len(fichiers), min(20, len(fichiers)), replace=False))
        plt.figure(figsize=(12, 10.5))
        plt.suptitle(f"Classe « {c} » : 20 photos au hasard sur {len(fichiers)}", fontsize=14)
        for i, k in enumerate(choix):
            plt.subplot(4, 5, i + 1)
            plt.imshow(_lire_photo(fichiers[k], projet.taille))
            plt.title(f"n° {k + 1}")
            plt.axis("off")
        plt.show()


def ecarter(projet, a_ecarter):
    """Étape 2 · Nettoyer : copie les photos dans un dossier de travail, sans celles qu'on écarte.

    a_ecarter : {"classe": [numéros]}. L'original n'est jamais modifié.
    """
    a_ecarter = a_ecarter or {}
    inconnues = [c for c in a_ecarter if c not in projet.classes]
    if inconnues:
        raise ValueError(f"Classe(s) inconnue(s) dans A_ECARTER : {inconnues}. Classes possibles : {projet.classes}")
    projet.travail = Path("photos_travail")
    shutil.rmtree(projet.travail, ignore_errors=True)
    projet.ecartees, projet.abimees, projet.gardees = [], [], {}
    for c in projet.classes:
        numeros = set(a_ecarter.get(c, []))
        hors = [n for n in numeros if not 1 <= n <= len(projet.fichiers[c])]
        if hors:
            raise ValueError(f"Numéro(s) {hors} impossible(s) pour « {c} » : les photos vont de 1 à {len(projet.fichiers[c])}.")
        for k, f in enumerate(projet.fichiers[c], 1):
            nom = f"{c}/{f.relative_to(projet.dossier / c)}"
            if k in numeros:
                projet.ecartees.append(f"{c} n° {k} ({f.name})")
                continue
            try:
                _lire_photo(f, projet.taille)
            except Exception:
                projet.abimees.append(nom)
                continue
            (projet.travail / nom).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, projet.travail / nom)
        projet.gardees[c] = len(_photos_de(projet.travail / c))
    print("Écartées par vous :", len(projet.ecartees))
    for e in projet.ecartees:
        print("  -", e)
    print("Abîmées, illisibles :", len(projet.abimees), projet.abimees)
    print("Photos gardées par classe :", projet.gardees)
    print("L'original n'a pas été modifié : le carnet travaille sur une copie.")


# ---------------------------------------------------------------- étape 1 : séparer

def separer(projet, part_validation=0.2):
    """Étape 1 · Séparer : une part des photos est mise de côté pour l'examen (la validation)."""
    if not hasattr(projet, "travail"):
        ecarter(projet, {})
    reglages = dict(class_names=projet.classes, validation_split=part_validation, seed=42,
                    image_size=(projet.taille, projet.taille), batch_size=LOT, crop_to_aspect_ratio=True)
    with contextlib.redirect_stdout(io.StringIO()):   # on remplace les messages anglais de Keras par les nôtres
        projet.apprentissage = keras.utils.image_dataset_from_directory(projet.travail, subset="training", **reglages)
        projet.validation = keras.utils.image_dataset_from_directory(projet.travail, subset="validation", **reglages)
    projet.part_validation = part_validation
    n_app, n_val = len(projet.apprentissage.file_paths), len(projet.validation.file_paths)
    print(f"{n_app + n_val} photos : {n_app} pour l'entraînement, {n_val} pour l'examen (la validation).")
    print(f"Le modèle les regardera par lots de {LOT} : {len(projet.apprentissage)} lots d'entraînement, "
          f"{len(projet.validation)} lots de validation.")
    print("L'étiquette de chaque photo est le nom de son dossier : il n'y a rien à encoder.")
    print(pd.DataFrame({"entraînement": _par_classe(projet.apprentissage, projet.classes),
                        "validation": _par_classe(projet.validation, projet.classes)}).to_string())


# ---------------------------------------------------------------- étape 3 : redimensionner, normaliser, augmenter

def avant_apres(projet):
    """Étape 3 · Une photo telle qu'elle est, puis en carré, puis ses pixels normalisés."""
    f = _photos_de(projet.travail / projet.classes[0])[0]
    originale = keras.utils.load_img(f)
    carree = _lire_photo(f, projet.taille)
    normalisee = carree / 127.5 - 1
    plt.figure(figsize=(9, 3.8))
    plt.subplot(1, 2, 1); plt.imshow(originale); plt.title(f"Originale : {originale.width} x {originale.height} px")
    plt.subplot(1, 2, 2); plt.imshow(carree); plt.title(f"Carrée : {projet.taille} x {projet.taille} px")
    plt.show()
    print("Une photo, pour la machine :", _en_clair(carree.size), "nombres =", " x ".join(str(n) for n in carree.shape))
    print("Pixels avant : de", carree.min(), "à", carree.max(), "· le pixel en haut à gauche (rouge, vert, bleu) :", carree[0, 0])
    print("Pixels après : de", normalisee.min().round(2), "à", normalisee.max().round(2), "· le même pixel :", normalisee[0, 0].round(2))


def augmenter(projet, augmentations):
    """Étape + · Prépare l'augmentation à partir de la liste de RÉGLAGES, et la montre."""
    _verifier(projet, "séparer", "apprentissage")
    transformations = [layers.Identity()]
    if augmentations.get("miroir"):
        transformations.append(layers.RandomFlip("horizontal"))
    if augmentations.get("rotation"):
        transformations.append(layers.RandomRotation(0.03))          # environ 10 degrés au plus
    if augmentations.get("zoom"):
        transformations.append(layers.RandomZoom(0.1))
    if augmentations.get("luminosite"):
        transformations.append(layers.RandomBrightness(0.1, value_range=(0, 255)))
    if augmentations.get("couleur"):
        transformations.append(layers.RandomHue(0.1, value_range=(0, 255)))
    transformer = keras.Sequential(transformations, name="augmentation")
    projet.augmentations = augmentations
    # branchée sur l'ENTRAÎNEMENT seulement, à la volée
    projet.apprentissage_augmente = projet.apprentissage.map(
        lambda x, y: (transformer(x, training=True), y)).prefetch(tf.data.AUTOTUNE)
    print("Augmentations actives :", [n for n, oui in augmentations.items() if oui] or "aucune")
    print("Interdites :", [n for n, oui in augmentations.items() if not oui] or "aucune")

    photo = _lire_photo(_photos_de(projet.travail / projet.classes[0])[0], projet.taille)
    plt.figure(figsize=(12, 2.6))
    plt.suptitle("Six variantes de la même photo")
    for i in range(6):
        plt.subplot(1, 6, i + 1)
        plt.imshow(np.clip(transformer(photo[None].astype("float32"), training=True)[0], 0, 255).astype("uint8"))
        plt.axis("off")
    plt.show()
    for titre, partie in [("Ce que voit le modèle à l'entraînement : des photos augmentées", projet.apprentissage_augmente),
                          ("L'examen (validation) : des photos brutes", projet.validation)]:
        x, y = next(iter(partie))
        plt.figure(figsize=(12, 2.6))
        plt.suptitle(titre)
        for i in range(6):
            plt.subplot(1, 6, i + 1)
            plt.imshow(np.clip(x[i], 0, 255).astype("uint8"))
            plt.title(projet.classes[int(y[i])], fontsize=9)
            plt.axis("off")
        plt.show()


# ---------------------------------------------------------------- verbe 1 : importer et créer

def creer_modele(projet):
    """Importer et créer : MobileNetV2 pré-entraîné, poids gelés, et une dernière couche neuve à K sorties."""
    k = len(projet.classes)
    with contextlib.redirect_stdout(io.StringIO()):   # le téléchargement des poids affiche une barre en anglais
        projet.base = keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    projet.base.trainable = False                     # les poids appris sur ImageNet sont GELÉS
    entree = keras.Input(shape=(projet.taille, projet.taille, 3))
    x = layers.Resizing(224, 224, name="redimensionner")(entree)
    x = layers.Rescaling(1 / 127.5, offset=-1, name="normaliser")(x)
    x = projet.base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="moyenne")(x)
    x = layers.Dropout(0.2, name="oubli")(x)
    sortie = layers.Dense(k, activation="softmax", name="decision")(x)
    projet.model = keras.Model(entree, sortie)
    projet.fine_tuning_fait = False

    geles = sum(int(np.prod(w.shape)) for w in projet.model.non_trainable_weights)
    entraines = sum(int(np.prod(w.shape)) for w in projet.model.trainable_weights)
    print("Le modèle, de l'entrée à la sortie :")
    print(f"  1. redimensionner et normaliser : la photo passe en 224 x 224, ses pixels entre -1 et 1")
    print(f"  2. MobileNetV2, pré-entraîné sur ImageNet, GELÉ : il décrit la photo par 1 280 nombres")
    print(f"  3. moyenne puis oubli (Dropout) : résumer, et éteindre au hasard 20 % des nombres pendant l'entraînement")
    print(f"  4. decision : la dernière couche, NEUVE, {k} sorties = une probabilité par classe {projet.classes}")
    print()
    print("Poids gelés (appris sur ImageNet, on n'y touche pas) :", _en_clair(geles))
    print("Poids entraînés (la dernière couche, neuve)           :", _en_clair(entraines))
    print("Part des poids qui apprennent :", f"{entraines / (geles + entraines) * 100:.2f} %".replace(".", ","))


# ---------------------------------------------------------------- verbe 2 : fit

class _Journal(keras.callbacks.Callback):
    """Une ligne en français par époque."""

    def __init__(self, epoques):
        super().__init__()
        self.epoques, self.debut = epoques, None

    def on_epoch_begin(self, epoch, logs=None):
        self.debut = time.time()

    def on_epoch_end(self, epoch, logs=None):
        print(f"Époque {epoch + 1:>2}/{self.epoques} · exactitude entraînement {_pourcent(logs['accuracy']):>6} · "
              f"validation {_pourcent(logs['val_accuracy']):>6} · {time.time() - self.debut:.0f} s")


def _courbes(historique, titre=""):
    h = historique.history
    epoques = range(1, len(h["accuracy"]) + 1)
    meilleure = int(np.argmax(h["val_accuracy"])) + 1
    plt.figure(figsize=(12, 4))
    for i, (mesure, nom) in enumerate([("accuracy", "Exactitude"), ("loss", "Perte")]):
        plt.subplot(1, 2, i + 1)
        plt.plot(epoques, h[mesure], "o-", color=BLEU, label="entraînement")
        plt.plot(epoques, h["val_" + mesure], "o-", color=ORANGE, label="validation")
        plt.plot(meilleure, h["val_" + mesure][meilleure - 1], "o", color="red", markersize=12,
                 label=f"meilleure époque ({meilleure})")
        plt.xticks(list(epoques))
        plt.xlabel("époque"); plt.title(nom + titre); plt.legend()
    plt.show()
    return meilleure


def entrainer(projet, epoques=10):
    """fit : entraîne la dernière couche, s'arrête seul quand la validation ne progresse plus, trace les deux courbes."""
    _verifier(projet, "augmenter", "apprentissage_augmente")
    _verifier(projet, "créer le modèle", "model")
    projet.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                         loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    arret = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    debut = time.time()
    projet.historique = projet.model.fit(projet.apprentissage_augmente, validation_data=projet.validation,
                                         epochs=epoques, callbacks=[arret, _Journal(epoques)], verbose=0)
    projet.duree = time.time() - debut
    projet.epoques_max = epoques
    projet.epoques_faites = len(projet.historique.history["accuracy"])
    meilleure = int(np.argmax(projet.historique.history["val_accuracy"])) + 1
    if projet.epoques_faites < epoques:
        print(f"\nArrêt automatique : la validation ne progresse plus depuis 3 époques.")
    print(f"Le modèle reprend les poids de sa meilleure époque : l'époque {meilleure}, "
          f"exactitude de validation {_pourcent(projet.historique.history['val_accuracy'][meilleure - 1])}.")
    print(f"{projet.epoques_faites} époques en {projet.duree / 60:.1f} minutes, "
          f"soit {projet.duree / projet.epoques_faites:.0f} s par époque.".replace(".", ",", 1))
    _courbes(projet.historique)


# ---------------------------------------------------------------- verbe 3 : évaluer

def evaluer(projet):
    """Évaluer, avec les outils de la séance 6 : exactitude, matrice de confusion, rappel et précision, baseline."""
    _verifier(projet, "entraîner", "historique")
    images, vraies = [], []
    for x, y in projet.validation:
        images.append(x.numpy().astype("uint8"))
        vraies.append(y.numpy())
    projet.images_val, projet.vraies = np.concatenate(images), np.concatenate(vraies)
    probas = projet.model.predict(projet.images_val, verbose=0)
    projet.predites, confiances = probas.argmax(axis=1), probas.max(axis=1)
    projet.exactitude = (projet.predites == projet.vraies).mean()
    k, classes = len(projet.classes), projet.classes

    print(f"Exactitude sur la validation : {_pourcent(projet.exactitude)} "
          f"({(projet.predites == projet.vraies).sum()} photos justes sur {len(projet.vraies)})")
    majoritaire = _par_classe(projet.apprentissage, classes).idxmax()
    projet.exactitude_paresseux = (projet.vraies == classes.index(majoritaire)).mean()
    print(f"Modèle paresseux (répond toujours « {majoritaire} », la classe la plus fréquente) : "
          f"{_pourcent(projet.exactitude_paresseux)}")

    affichage = ConfusionMatrixDisplay(confusion_matrix(projet.vraies, projet.predites, labels=range(k)),
                                       display_labels=classes)
    affichage.plot(cmap="Blues", colorbar=False)
    affichage.ax_.set_xlabel("prédit")
    affichage.ax_.set_ylabel("vraiment")
    affichage.ax_.set_title("Matrice de confusion (validation)")
    plt.show()

    rapport = pd.DataFrame(classification_report(projet.vraies, projet.predites, labels=range(k), target_names=classes,
                                                 output_dict=True, zero_division=0)).T.loc[classes]
    projet.rapport = pd.DataFrame({"rappel": rapport["recall"].round(2), "précision": rapport["precision"].round(2),
                                   "photos": rapport["support"].astype(int)})
    print("Rappel : parmi les photos VRAIMENT de la classe, la part trouvée.")
    print("Précision : parmi les photos PRÉDITES de la classe, la part juste.")
    print(projet.rapport.to_string())

    erreurs = np.where(projet.predites != projet.vraies)[0]
    print(f"\n{len(erreurs)} photos mal classées sur {len(projet.vraies)} : vraie classe → classe prédite (confiance)")
    if len(erreurs):
        lignes = (min(len(erreurs), 12) + 5) // 6
        plt.figure(figsize=(15, 3 * lignes))
        for i, j in enumerate(erreurs[:12]):
            plt.subplot(lignes, 6, i + 1)
            plt.imshow(projet.images_val[j])
            plt.title(f"{classes[projet.vraies[j]]} → {classes[projet.predites[j]]} ({confiances[j]:.0%})", fontsize=9)
            plt.axis("off")
        plt.show()


# ---------------------------------------------------------------- verbe 4 : predict

def predire_une_photo(projet, source_nouvelles, choix=1, seuil=0.6):
    """predict sur une photo : ses K probabilités, la classe retenue, et le seuil de confiance."""
    _verifier(projet, "entraîner", "historique")
    projet.nouvelles = _recuperer(source_nouvelles, "nouvelles")
    if choix == "televerser":
        from google.colab import files
        chemin = Path(list(files.upload())[0])
    else:
        photos = _photos_de(projet.nouvelles)
        if not 1 <= choix <= len(photos):
            raise ValueError(f"Choisissez un numéro entre 1 et {len(photos)}, ou \"televerser\".")
        chemin = photos[choix - 1]
    photo = _lire_photo(chemin, projet.taille)
    p = projet.model.predict(photo[None], verbose=0)[0]
    classe, confiance = projet.classes[p.argmax()], p.max()
    decision = classe if confiance >= seuil else "à vérifier par un humain"
    plt.figure(figsize=(10, 3.5))
    plt.subplot(1, 2, 1); plt.imshow(photo); plt.axis("off"); plt.title(chemin.name)
    plt.subplot(1, 2, 2); plt.barh(projet.classes, p, color=BLEU); plt.xlim(0, 1)
    plt.axvline(seuil, color=ORANGE, linestyle="--", label=f"seuil {seuil}"); plt.legend()
    plt.title(f"{classe} ({confiance:.0%}) → {decision}")
    plt.show()
    print("Probabilités :", {c: round(float(v), 3) for c, v in zip(projet.classes, p)})
    print("Décision :", decision)


def predire_dossier(projet, source_nouvelles, seuil=0.6, seuils_a_comparer=(0.5, 0.6, 0.8)):
    """predict sur tout un dossier : un tableau fichier, classe, confiance, décision, enregistré en CSV."""
    _verifier(projet, "entraîner", "historique")
    projet.nouvelles = _recuperer(source_nouvelles, "nouvelles")
    fichiers = _photos_de(projet.nouvelles)
    p = projet.model.predict(np.stack([_lire_photo(f, projet.taille) for f in fichiers]), verbose=0)
    projet.seuil = seuil
    projet.resultats = pd.DataFrame({"fichier": [f.name for f in fichiers],
                                     "classe": [projet.classes[i] for i in p.argmax(axis=1)],
                                     "confiance": p.max(axis=1).round(3)})
    projet.resultats["decision"] = np.where(projet.resultats["confiance"] >= seuil, projet.resultats["classe"],
                                            "à vérifier par un humain")
    projet.resultats.to_csv("predictions.csv", index=False)
    print(f"{len(fichiers)} photos nouvelles. Les 10 premières lignes du tableau (enregistré dans predictions.csv) :")
    print(projet.resultats.head(10).to_string(index=False))
    print(f"\nCombien de photos par décision, avec le seuil {seuil} :")
    print(projet.resultats["decision"].value_counts().to_string())
    print("\nEt avec d'autres seuils :")
    for s in seuils_a_comparer:
        print(f"  seuil {s} : {(projet.resultats['confiance'] < s).sum():>3} photos à vérifier sur {len(fichiers)}")


# ---------------------------------------------------------------- où regarde le modèle ?

def _carte(projet, photo):
    m = projet.model
    x = m.get_layer("normaliser")(m.get_layer("redimensionner")(photo[None].astype("float32")))
    with tf.GradientTape() as bande:
        filtres = projet.base(x, training=False)          # la sortie de la dernière couche de filtres
        bande.watch(filtres)
        p = m.get_layer("decision")(m.get_layer("moyenne")(filtres))
        choisie = int(tf.argmax(p[0]))
        score = p[:, choisie]
    gradients = bande.gradient(score, filtres)
    poids = tf.reduce_mean(gradients, axis=(1, 2))      # l'importance de chaque filtre
    carte = tf.nn.relu(tf.reduce_sum(filtres * poids[:, None, None, :], axis=-1))[0]
    carte = carte / (tf.reduce_max(carte) + 1e-8)
    taille = projet.taille
    return tf.image.resize(carte[..., None], (taille, taille))[..., 0].numpy(), choisie, float(p[0, choisie])


def cartes_de_chaleur(projet, n=6):
    """Carte de chaleur (Grad-CAM) de n photos de validation : en rouge, ce qui a décidé."""
    _verifier(projet, "évaluer", "images_val")
    plt.figure(figsize=(15, 5.5))
    for i, j in enumerate(np.random.default_rng(42).choice(len(projet.images_val), n, replace=False)):
        carte, choisie, confiance = _carte(projet, projet.images_val[j])
        plt.subplot(2, n, i + 1); plt.imshow(projet.images_val[j]); plt.axis("off")
        plt.title(f"vraiment : {projet.classes[projet.vraies[j]]}", fontsize=9)
        plt.subplot(2, n, i + n + 1); plt.imshow(projet.images_val[j]); plt.imshow(carte, cmap="jet", alpha=0.45)
        plt.axis("off")
        plt.title(f"prédit : {projet.classes[choisie]} ({confiance:.0%})", fontsize=9)
    plt.show()


# ---------------------------------------------------------------- sauvegarder et documenter

def enregistrer(projet, dossier_sortie, limites=""):
    """Enregistre le modèle, les prédictions et la fiche du modèle (le journal de la séance 5, pour un modèle)."""
    _verifier(projet, "évaluer", "rapport")
    if EN_COLAB and str(dossier_sortie).startswith("/content/drive"):
        from google.colab import drive
        drive.mount("/content/drive")
        sortie = Path(dossier_sortie)
    else:
        sortie = Path("sorties")
    sortie.mkdir(parents=True, exist_ok=True)
    predites = projet.model.predict(projet.images_val, verbose=0).argmax(axis=1)   # le modèle tel qu'il est maintenant
    rapport = pd.DataFrame(classification_report(projet.vraies, predites, labels=range(len(projet.classes)),
                                                 target_names=projet.classes, output_dict=True, zero_division=0)).T
    lignes = "\n".join(f"| {c} | {projet.gardees[c]} | {rapport.loc[c, 'recall']:.2f} | {rapport.loc[c, 'precision']:.2f} |"
                       for c in projet.classes)
    augm = getattr(projet, "augmentations", {})
    h = projet.historique.history
    fiche = f"""# Fiche du modèle : {', '.join(projet.classes)}

- **Date :** {date.today():%d/%m/%Y}
- **Photos :** `{projet.source}`
- **Modèle :** MobileNetV2 pré-entraîné sur ImageNet, poids gelés{', dernières couches dégelées (fine-tuning)' if projet.fine_tuning_fait else ''} ; dernière couche neuve à {len(projet.classes)} sorties
- **Préparation :** photos recadrées en carré, {projet.taille} x {projet.taille} px ; {len(projet.ecartees)} photo(s) écartée(s) à la main, {len(projet.abimees)} illisible(s)
- **Partage :** {len(projet.apprentissage.file_paths)} photos pour l'entraînement, {len(projet.validation.file_paths)} pour la validation (photo par photo, graine 42)
- **Augmentations :** {', '.join(n for n, oui in augm.items() if oui) or 'aucune'} (interdites : {', '.join(n for n, oui in augm.items() if not oui) or 'aucune'})
- **Entraînement :** {projet.epoques_faites} époques sur {projet.epoques_max} au plus, meilleure époque {int(np.argmax(h['val_accuracy'])) + 1}, {f'{projet.duree / 60:.1f}'.replace('.', ',')} minutes
- **Exactitude sur la validation :** {_pourcent((predites == projet.vraies).mean())} (modèle paresseux : {_pourcent(projet.exactitude_paresseux)})
- **Seuil de confiance :** {getattr(projet, 'seuil', 'non choisi')} ; en dessous, la photo est à vérifier par un humain

| Classe | Photos | Rappel | Précision |
|---|---|---|---|
{lignes}

**Limites connues**

{limites or "À COMPLÉTER : les sites, appareils et conditions vus, et ceux qui ne l'ont pas été."}

Le modèle propose, un humain décide.
"""
    projet.model.save(sortie / "modele_photos.keras")
    if hasattr(projet, "resultats"):
        projet.resultats.to_csv(sortie / "predictions.csv", index=False)
    (sortie / "fiche_modele.md").write_text(fiche, encoding="utf-8")
    print("Enregistrés dans", sortie, ":", sorted(f.name for f in sortie.iterdir()))
    print()
    print(fiche)


# ---------------------------------------------------------------- bonus

def fine_tuning(projet, couches=20, epoques=5):
    """BONUS · Dégèle les dernières couches de MobileNetV2 et réentraîne avec un pas dix fois plus petit."""
    _verifier(projet, "entraîner", "historique")
    avant = max(projet.historique.history["val_accuracy"])
    projet.base.trainable = True
    for couche in projet.base.layers[:-couches]:
        couche.trainable = False                     # tout reste gelé, sauf les dernières couches
    projet.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                         loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print(f"{couches} dernières couches dégelées. Poids entraînés maintenant :",
          _en_clair(sum(int(np.prod(w.shape)) for w in projet.model.trainable_weights)))
    debut = time.time()
    historique = projet.model.fit(projet.apprentissage_augmente, validation_data=projet.validation, epochs=epoques,
                                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3,
                                                                           restore_best_weights=True),
                                             _Journal(epoques)], verbose=0)
    duree = time.time() - debut
    projet.fine_tuning_fait = True
    apres = max(historique.history["val_accuracy"])
    _courbes(historique, " (fine-tuning)")
    print(f"Validation : {_pourcent(avant)} avant, {_pourcent(apres)} après "
          f"({(apres - avant) * 100:+.1f} points), pour {duree / 60:.1f} minutes de plus.".replace(".", ",", 2))
