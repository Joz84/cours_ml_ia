"""Corrigé du TP « Classer des photos », version simple (utilisé par scripts/build_tp_donnees.py)."""
import os
import re
import shutil
from pathlib import Path

NOTEBOOK = "TP_classer_photos"
FICHIERS = ["data/images_exemple.zip", "data/photos_nouvelles.zip", "tp/classer_photos/outils_photos.py"]
TIMEOUT = 3600   # l'entraînement sans GPU dépasse le quart d'heure par défaut

README = (Path(__file__).resolve().parent.parent.parent / "tp" / "classer_photos" / "README.md").read_text()


def bloc(ide, *remplacements):
    """Le code du bloc <!-- id: ide --> du README, avec les lignes de décision remplacées."""
    code = re.search(rf"<!-- id: {ide} -->\s*\n```python\n(.*?)```", README, re.S).group(1).rstrip()
    for avant, apres in remplacements:
        assert avant in code, (ide, avant)
        code = code.replace(avant, apres)
    return code


TROUS = {
    "ecarter": bloc("ecarter", ('A_ECARTER = {}   # DÉCISION : par ex. {"tulipe": [12, 187], "tournesol": [40]}',
                                '''A_ECARTER = {                   # DÉCISION
    "marguerite": [76, 214],     # 76 : des asters violets, mal rangés ; 214 : un bouquet d'œillets roses, hors sujet
    "tournesol": [293],          # une personne au premier plan, qui cache presque la fleur
    "tulipe": [17],              # quasi-doublon de la n° 18 : le même moulin, la même prise de vue
}''')),
    "fiche": bloc("fiche", ('LIMITES = ""   # DÉCISION : les sites, appareils et conditions vus, et ceux qui ne l\'ont pas été',
                            '''LIMITES = ("Photos de fleurs publiées sur Flickr, prises de jour, le plus souvent de près, par des appareils "
           "et des photographes très divers. Le modèle n'a vu ni les fleurs de Polynésie, ni des photos de nuit, "
           "ni d'autres fleurs que ces trois : une rose ou un pissenlit reçoivent quand même l'une des trois étiquettes.")   # DÉCISION''')),
}

REPONSES = {
    "Étape 0": "**Q1.** Oui : **3 classes de 300 photos**, parfaitement équilibrées. Si une classe avait moins de la moitié des photos des autres, le modèle aurait intérêt à répondre plus souvent les grosses classes, et l'exactitude cacherait son échec sur la petite. On peut **sous-échantillonner** les autres classes, mais le mieux est d'**aller chercher** des photos de la classe qui manque.",
    "Étape 2": "**Q2.** Sur les 60 photos regardées, le corrigé en écarte 4 : `marguerite` n° 76 montre des **asters violets**, une autre fleur (**mal rangée**) ; `marguerite` n° 214 est un bouquet d'œillets roses où la marguerite se voit à peine (**hors sujet**) ; `tournesol` n° 293 montre surtout une **personne**, qui cache presque la fleur (et une photo de personne pose une question de vie privée) ; `tulipe` n° 17 est un **quasi-doublon** de la n° 18 : le même moulin, la même prise de vue. Hésitations, photos gardées : `marguerite` n° 10 (une pensée jaune au milieu de vraies marguerites), `tournesol` n° 284 (de jeunes plants sans fleur). Une fois les cellules suivantes exécutées, on découvre d'autres intrus, par exemple une peintre devant son chevalet rangée dans `tulipe` (étape 6) : on ne voit pas tout en regardant 20 photos sur 300.",
    "Étape 1": "**Q3.** Deux photos presque identiques peuvent tomber l'une dans l'entraînement, l'autre dans l'examen : le modèle reconnaît la jumelle qu'il a déjà vue au lieu de classer une photo nouvelle. L'exactitude mesurée est alors trop belle, et elle s'effondre sur un nouveau site. En gardant des **sites entiers** pour l'examen, on mesure ce qui compte : le modèle marche-t-il sur un site qu'il n'a jamais vu ? Dans l'exemple, les tulipes n° 17 et n° 18 (le même moulin) sont ce genre de jumelles.",
    "Étape 3": "**Q4.** Oui, pour les fleurs aussi, la couleur doit rester interdite : une marguerite est blanche, un tournesol jaune, une tulipe souvent rouge ou rose. Changer les teintes fabriquerait des tournesols rouges ou des marguerites violettes, et brouillerait l'indice le plus utile. Le miroir, une rotation légère, un zoom léger et la luminosité ne changent pas l'espèce : ils sont permis. Pour le corail, c'est encore plus net : la couleur *est* la classe (sain, pâle, blanchi).",
    "Étape 4": "**Q5.** **2 257 984 poids gelés**, ceux de MobileNetV2, et **3 843 poids entraînés**, ceux de la dernière couche (1 280 nombres × 3 sorties, plus 3) : **0,17 %** des poids apprennent. Le réseau sait déjà voir des traits, des textures et des formes ; il ne reste qu'à apprendre comment combiner ses 1 280 nombres pour séparer 3 fleurs. Pour régler 3 843 poids, 300 photos par classe suffisent ; pour en régler 2,3 millions, il en faudrait des dizaines de milliers. C'est la diapo « on ne part jamais de zéro ».",
    "Étape 5": "**Q6.** La validation atteint son meilleur niveau, **93,9 %**, à l'**époque 6** : c'est là qu'il fallait s'arrêter, et c'est ce qu'a fait l'arrêt automatique (arrêt à l'époque 9, poids de l'époque 6). Ensuite, l'entraînement continue de monter (96 %) alors que la validation stagne autour de 93-94 % : c'est un **début de sur-apprentissage**, encore léger. Au début, la validation est au-dessus de l'entraînement (86,6 % contre 68,6 % à la 1ʳᵉ époque) : l'exactitude d'entraînement est mesurée sur des photos **augmentées**, avec l'oubli actif, et en moyenne pendant une époque où le modèle progresse encore ; la validation est mesurée sur des photos brutes, à la fin de l'époque.",
    "Étape 6": "**Q7.** Oui, nettement : **93,9 %** contre **31,3 %** pour le modèle paresseux, soit 63 points de plus. La confusion la plus fréquente : des **marguerites prises pour des tournesols** (4 photos), puis des **tulipes prises pour des tournesols** (3). Beaucoup d'erreurs sont difficiles pour un humain aussi : des marguerites jaunes, un gros plan de pétales orange, un massif de petites fleurs blanches vu de loin, une fleur à contre-jour. Certaines ne sont pas des erreurs du modèle : une photo d'une **peintre devant son chevalet** est rangée dans `tulipe`, c'est l'étiquette qui est fausse. Pour le corail, l'erreur la plus coûteuse serait de dire « sain » à un corail blanchi : on surveillerait le **rappel de `blanchi`**. Pour le miconia, le **rappel de `miconia`**.",
    "Étape 7": "**Q8.** Les 4 roses sont toutes classées **tulipe**, avec des confiances de **66 % à 99,7 %** : aucune ne passe sous le seuil de 0,6. Les 4 pissenlits sont classés marguerite ou tournesol, à 75 % et plus. Le modèle ne peut pas dire « je ne sais pas » : sa dernière couche donne des probabilités **dont la somme fait 1**, il répartit donc toujours toute la confiance entre les classes qu'il connaît. Le seuil attrape les hésitations (ici, l'unique erreur sur les fleurs apprises, à 48 %), pas les fleurs inconnues. La parade : une classe `autre`, avec des photos de ce qu'on risque de rencontrer. Pour l'association : avec 0,6, 3 photos sur 32 partent à vérifier ; avec 0,8, 9 photos, soit plus d'un quart. Avec deux bénévoles et des milliers de photos, un seuil de **0,6 à 0,7** est raisonnable, à condition de vérifier de temps en temps un échantillon des photos acceptées.",
    "Étape 8": "**Q9.** Sur ces six photos, **les zones rouges sont sur les fleurs** : les tulipes, la tête de tournesol séchée à gauche, la petite marguerite dans l'herbe. Sur la photo d'une femme dans un champ de tournesols, la chaleur est sur les tournesols à droite, **pas sur la personne** : bon signe. Si les zones rouges tombaient sur le fond (le ciel, la pelouse, une personne), ce serait un **raccourci** : on reprendrait l'**étape 0**, en collectant des photos plus variées (des fleurs sur fond sombre et sur fond clair, avec et sans personne), puis l'**étape 2**, en écartant les photos où le sujet n'est pas la fleur.",
    "Étape 9": "**Q10.** Ce modèle reconnaît, sur une photo, une marguerite, un tournesol ou une tulipe, avec à peu près 9 réponses justes sur 10 sur des photos qu'il n'avait jamais vues. Il ne sait pas dire « je ne sais pas » devant une autre fleur, et il n'a pas été testé sur des photos prises ailleurs, avec d'autres appareils : avant de s'en servir sur le terrain, il faut le tester là-bas. Il trie et propose ; les photos douteuses vont à un humain, et c'est un humain qui signe la décision.",
    "Bonus": "**Q11.** La validation passe de **93,9 % à 95,5 %**, soit **+1,7 point**, c'est-à-dire **3 photos** sur 179, pour environ une minute de calcul en plus sur ce portable. Un écart de 3 photos peut venir du hasard. Et l'entraînement monte à 99 % pendant que la validation redescend à la dernière époque : le sur-apprentissage guette. Ici, le fine-tuning ne valait pas vraiment le coup : mieux vaut d'abord retirer les étiquettes fausses (la peintre rangée dans `tulipe`) et ajouter de la variété.",
}


def APRES(racine):
    """Efface ce que l'exécution du corrigé a écrit à côté du carnet, et remet les adresses GitHub dans les sorties."""
    dossier = racine / "tp" / "classer_photos"
    if os.environ.get("GARDER"):
        return
    for nom in ["mes_photos", "nouvelles", "photos_travail", "sorties", "__pycache__"]:
        shutil.rmtree(dossier / nom, ignore_errors=True)
    for nom in ["predictions.csv", "mes_photos.zip", "nouvelles.zip"]:
        (dossier / nom).unlink(missing_ok=True)
    import nbformat as nbf
    carnet = dossier / f"{NOTEBOOK}_CORRIGE.ipynb"
    nb = nbf.read(carnet, as_version=4)
    brut = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/"
    for c in nb.cells:
        for o in c.get("outputs", []):
            for rel in FICHIERS:
                if "text" in o:
                    o["text"] = o["text"].replace(str(racine / rel), brut + rel)
                for cle in ["text/plain", "text/markdown"]:
                    if cle in o.get("data", {}):
                        o["data"][cle] = o["data"][cle].replace(str(racine / rel), brut + rel)
    nbf.write(nb, carnet)
