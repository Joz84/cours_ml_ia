"""Fabrique le jeu de photos d'exemple du TP « Classer des photos ».

Usage :
    python scripts/make_dataset_exemple.py
    python scripts/make_dataset_exemple.py --classes daisy,sunflowers --sortie /tmp/deux_classes   (tests)

Source : le jeu « flower_photos » de TensorFlow (aussi appelé tf_flowers), 3 670 photos
de fleurs prises par des photographes de Flickr, en 5 classes (daisy, dandelion, roses,
sunflowers, tulips). Licence : Creative Commons Attribution 2.0
(https://creativecommons.org/licenses/by/2.0/), avec un auteur par photo, listé dans
le fichier LICENSE.txt de l'archive. On ne garde QUE des photos qui y figurent, et on
recopie leur auteur dans LICENCE_ET_AUTEURS.txt.

Ce que fait le script (seed 42, résultat identique à chaque exécution) :
1. télécharge l'archive (218 Mo) dans data/raw/flower_photos/ si elle n'y est pas ;
2. tire 300 photos par classe pour 3 classes, renommées en français :
   marguerite (daisy), tournesol (sunflowers), tulipe (tulips) ;
3. recadre chaque photo en carré (au centre), la met en 256 x 256 px, et l'enregistre
   en JPEG dans data/images_exemple/<classe>/ ; puis compresse le tout dans
   data/images_exemple.zip ;
4. tire 32 autres photos, jamais vues par le modèle : 8 par classe apprise, plus 4 roses
   et 4 pissenlits, deux fleurs que le modèle ne connaît pas ; les enregistre sous des
   noms neutres (nouvelle_01.jpg…) dans data/photos_nouvelles.zip, avec un fichier
   etiquettes.csv qui donne la vraie fleur de chaque photo.

Les photos de l'archive font en général 320 x 240 px : le carré central fait donc
240 px de côté, légèrement agrandi à 256.
"""
import argparse
import csv
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

RACINE = Path(__file__).resolve().parent.parent
URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
SOURCE = RACINE / "data" / "raw" / "flower_photos"
SEED = 42
TAILLE = 256
PAR_CLASSE = 300
FRANCAIS = {"daisy": "marguerite", "sunflowers": "tournesol", "tulips": "tulipe",
            "roses": "rose", "dandelion": "pissenlit"}
NOUVELLES = {"daisy": 8, "sunflowers": 8, "tulips": 8, "roses": 4, "dandelion": 4}


def telecharger():
    if (SOURCE / "LICENSE.txt").exists():
        return
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    archive = SOURCE.parent / "flower_photos.tgz"
    print("Téléchargement de", URL)
    urllib.request.urlretrieve(URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(SOURCE.parent, filter="data")
    archive.unlink()


def lire_auteurs():
    """{"daisy/123_abc_n.jpg": "CC-BY by … - https://www.flickr.com/…"} d'après LICENSE.txt."""
    auteurs = {}
    for ligne in (SOURCE / "LICENSE.txt").read_text(encoding="utf-8").splitlines():
        if " CC-BY by " in ligne:
            fichier, credit = ligne.split(" ", 1)
            auteurs[fichier] = credit
    return auteurs


def carre(chemin):
    """Recadre au centre en carré, puis met en TAILLE x TAILLE px."""
    image = ImageOps.exif_transpose(Image.open(chemin)).convert("RGB")
    cote = min(image.size)
    gauche, haut = (image.width - cote) // 2, (image.height - cote) // 2
    image = image.crop((gauche, haut, gauche + cote, haut + cote))
    return image.resize((TAILLE, TAILLE), Image.LANCZOS)


def zipper(dossier, archive):
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:  # le JPEG est déjà compressé
        for f in sorted(dossier.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(dossier.parent))


def entete(texte):
    return (f"{texte}\n\nSource : jeu « flower_photos » de TensorFlow (tf_flowers),\n{URL}\n"
            "Licence : Creative Commons Attribution 2.0, https://creativecommons.org/licenses/by/2.0/\n"
            "Modifications : photos recadrées en carré et redimensionnées en 256 x 256 px.\n\n"
            "Photo, puis son auteur et l'adresse de l'original :\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", default="daisy,sunflowers,tulips",
                        help="classes du jeu source à garder, séparées par des virgules")
    parser.add_argument("--sortie", default=str(RACINE / "data" / "images_exemple"))
    args = parser.parse_args()
    classes = args.classes.split(",")
    sortie = Path(args.sortie)
    par_defaut = sortie == RACINE / "data" / "images_exemple"

    telecharger()
    auteurs = lire_auteurs()
    rng = np.random.default_rng(SEED)

    # chaque photo de la source n'est tirée qu'une fois : apprentissage OU nouvelle
    restantes = {}
    for c in FRANCAIS:
        dispo = sorted(f"{c}/{p.name}" for p in (SOURCE / c).glob("*.jpg") if f"{c}/{p.name}" in auteurs)
        restantes[c] = [dispo[i] for i in rng.permutation(len(dispo))]

    shutil.rmtree(sortie, ignore_errors=True)
    credits = []
    for c in classes:
        (sortie / FRANCAIS[c]).mkdir(parents=True)
        for rel in restantes[c][:PAR_CLASSE]:
            nom = Path(rel).name
            carre(SOURCE / rel).save(sortie / FRANCAIS[c] / nom, quality=90)
            credits.append(f"{FRANCAIS[c]}/{nom}  {auteurs[rel]}")
        restantes[c] = restantes[c][PAR_CLASSE:]
        print(f"{FRANCAIS[c]:<12} {len(list((sortie / FRANCAIS[c]).glob('*.jpg')))} photos")
    (sortie / "LICENCE_ET_AUTEURS.txt").write_text(
        entete("Jeu d'exemple du TP « Classer des photos » (UE 5.7, UPF).") + "\n".join(credits) + "\n",
        encoding="utf-8")
    if not par_defaut:
        return
    zipper(sortie, RACINE / "data" / "images_exemple.zip")

    # photos nouvelles, sous des noms neutres, mélangées
    nouvelles = RACINE / "data" / "photos_nouvelles"
    shutil.rmtree(nouvelles, ignore_errors=True)
    nouvelles.mkdir(parents=True)
    tirees = [rel for c, k in NOUVELLES.items() for rel in restantes[c][:k]]
    tirees = [tirees[i] for i in rng.permutation(len(tirees))]
    credits = []
    with open(nouvelles / "etiquettes.csv", "w", newline="", encoding="utf-8") as f:
        ecrire = csv.writer(f)
        ecrire.writerow(["fichier", "vraie_fleur", "classe_apprise"])
        for i, rel in enumerate(tirees, 1):
            nom, fleur = f"nouvelle_{i:02d}.jpg", Path(rel).parent.name
            carre(SOURCE / rel).save(nouvelles / nom, quality=90)
            ecrire.writerow([nom, FRANCAIS[fleur], "oui" if fleur in classes else "non"])
            credits.append(f"{nom}  {auteurs[rel]}")
    (nouvelles / "LICENCE_ET_AUTEURS.txt").write_text(
        entete("Photos nouvelles du TP « Classer des photos » (UE 5.7, UPF).") + "\n".join(credits) + "\n",
        encoding="utf-8")
    zipper(nouvelles, RACINE / "data" / "photos_nouvelles.zip")

    for z in ["images_exemple.zip", "photos_nouvelles.zip"]:
        print(f"data/{z} : {(RACINE / 'data' / z).stat().st_size / 1e6:.1f} Mo")


if __name__ == "__main__":
    main()
