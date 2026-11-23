"""Corrigé du TP « Classer des photos » (utilisé par scripts/build_tp_donnees.py)."""
import os
import re
import shutil
from pathlib import Path

NOTEBOOK = "TP_classification_images"
FICHIERS = ["data/images_exemple.zip", "data/photos_nouvelles.zip"]
TIMEOUT = 3600   # l'entraînement sans GPU dépasse le quart d'heure par défaut

README = (Path(__file__).resolve().parent.parent.parent / "tp" / "classification_images" / "README.md").read_text()


def bloc(ide, *remplacements):
    """Le code du bloc <!-- id: ide --> du README, avec les lignes de décision remplacées."""
    code = re.search(rf"<!-- id: {ide} -->\s*\n```python\n(.*?)```", README, re.S).group(1).rstrip()
    for avant, apres in remplacements:
        assert avant in code, (ide, avant)
        code = code.replace(avant, apres)
    return code


TROUS = {
    "ecarter": bloc("ecarter",
        ('A_ECARTER = []        # DÉCISION : par ex. ["tulipe/1234_abcd_n.jpg", "tournesol/5678_efgh_m.jpg"]',
         '''A_ECARTER = [                                    # DÉCISION
    "tulipe/17202535346_ab828e779b.jpg",             # un arbre en noir et blanc : hors sujet
    "tulipe/14674388855_2da18e375a_m.jpg",           # une planche botanique dessinée : pas une photo
    "tulipe/14487762578_baba13d16a_m.jpg",           # une illustration peinte : pas une photo
    "tournesol/4933230161_12f3ee7587.jpg",           # un portrait : la personne occupe toute la photo
    "tournesol/15054750690_198b6ab0f2_n.jpg",        # floue : on ne reconnaît pas la fleur
    "marguerite/21805938544_bf6bb0e4bc.jpg",         # des fleurs roses d'une autre espèce : mal rangée
]'''),
        ('justification_ecarter = ""   # DÉCISION : une phrase',
         'justification_ecarter = "Sur les 60 photos regardées, 6 apprendraient au modèle autre chose que la fleur : on les écarte, sans les supprimer de l\'original."')),
    "une_photo": bloc("une_photo", ("CHOIX = 1   #", "CHOIX = 16  #")),
    "fine_tuning": bloc("fine_tuning", ("FINE_TUNING = False", "FINE_TUNING = True")),
    "fiche": bloc("fiche", ('LIMITES = ""', '''LIMITES = ("Photos de fleurs publiées sur Flickr, prises de jour, le plus souvent de près, par des appareils et des photographes "
           "très divers. Le modèle n'a vu ni les fleurs de Polynésie, ni des photos de nuit, ni d'autres fleurs que ces trois : "
           "une rose ou un pissenlit reçoivent quand même l'une des trois étiquettes, souvent avec une forte confiance.")''')),
}

REPONSES = {
    "Q1": "**3 classes** (`marguerite`, `tournesol`, `tulipe`), **300 photos chacune** : le jeu est parfaitement équilibré. Si une classe avait moins de la moitié des photos des autres, le modèle aurait intérêt à répondre plus souvent les grosses classes, et l'exactitude cacherait son échec sur la petite. Deux solutions du cours : **sous-échantillonner** les autres classes (en garder autant que la plus petite), ou **aller chercher** des photos de la classe qui manque, ce qui est toujours mieux.",
    "Q2": "Sur les 60 photos regardées, **6 posent un vrai problème** : `tulipe/17202535346_ab828e779b.jpg` est un arbre en noir et blanc (**hors sujet**) ; `tulipe/14674388855_2da18e375a_m.jpg` et `tulipe/14487762578_baba13d16a_m.jpg` sont des planches botaniques dessinées ou peintes (**pas des photos**) ; `tournesol/15054750690_198b6ab0f2_n.jpg` est **floue** ; `marguerite/21805938544_bf6bb0e4bc.jpg` montre des fleurs roses d'une autre espèce (**mal rangée**) ; `tournesol/4933230161_12f3ee7587.jpg` est un **portrait** où le visage occupe toute la photo, ce qui pose aussi une question de vie privée. Elles sont écartées dans la cellule suivante. Deux cas limites sont gardés : `tournesol/4933822052_2dfef02517.jpg` et `tournesol/5037790727_57c527494f.jpg` montrent une personne dans un champ de tournesols, mais les fleurs sont bien visibles. On repère aussi des **quasi-doublons** : les deux photos `4933…` montrent la même personne, dans la même séance photo. C'est exactement le piège des rafales de Q4.",
    "Q3": "80 % de 894 photos gardées, c'est 715,2, arrondi à 716 : **716 photos pour l'entraînement, 178 pour la validation**. Chaque classe a à peu près 20 % de ses photos dans la validation (61 sur 299 marguerites, 62 sur 298 tournesols, 55 sur 297 tulipes, soit 18,5 à 21 %). Le partage est tiré au hasard, pas classe par classe : avec 300 photos par classe, l'écart reste faible.",
    "Q4": "Dans le jeu d'exemple, les photos viennent de nombreux photographes, mais certaines sortent d'une même séance (les deux portraits `4933…` de Q2). Dans un vrai projet, par exemple sur le corail, on prend souvent 20 photos du même massif dans la même plongée. **Deux photos presque identiques peuvent alors tomber l'une dans l'entraînement, l'autre dans la validation : le modèle reconnaît la jumelle qu'il a déjà vue au lieu de classer une photo nouvelle, et l'exactitude mesurée est trop belle.** En gardant des sites entiers pour l'examen, on mesure ce qui compte : le modèle marche-t-il sur un site qu'il n'a jamais vu ?",
    "Q5": "Pour les fleurs de l'exemple, il faut interdire la **couleur** : une marguerite est blanche, un tournesol jaune, une tulipe souvent rouge ou rose. Changer les teintes fabriquerait des tournesols rouges ou des marguerites violettes qui n'existent pas, et brouillerait l'indice le plus utile. Le miroir, une rotation légère, un zoom léger et la luminosité ne changent pas l'espèce : ils sont permis. Pour le corail, la couleur est interdite pour la même raison, encore plus fortement : la couleur *est* la classe (sain, pâle, blanchi).",
    "Q6": "**2 257 984 poids gelés**, ceux de MobileNetV2, appris sur ImageNet ; **3 843 poids entraînés**, ceux de la dernière couche : 1 280 nombres en entrée × 3 sorties = 3 840, plus un poids par sortie (3), soit 3 843. **Seulement 0,17 % des poids apprennent.** C'est la diapo « on ne part jamais de zéro » : le réseau sait déjà voir des traits, des textures et des formes ; il ne reste qu'à apprendre comment combiner ces 1 280 descriptions pour séparer 3 fleurs. Pour ajuster 3 843 poids, 300 photos par classe suffisent ; pour en ajuster 2,3 millions, il en faudrait des dizaines de milliers.",
    "Q7": "Au début, la validation est **au-dessus** de l'entraînement (82 % contre 56 % à la première époque). Ce n'est pas une erreur : l'exactitude d'entraînement est mesurée sur des photos **augmentées**, avec l'`oubli` actif, et en moyenne pendant une époque où le modèle progresse encore ; la validation est mesurée sur des photos brutes, à la fin de l'époque, sans `oubli`. Les deux courbes se rejoignent vers la 5ᵉ époque. Ensuite, l'entraînement continue de monter (95 %) alors que la validation **plafonne à 93,3 % dès la 4ᵉ époque** : il fallait s'arrêter à l'**époque 4**, et c'est ce qu'a fait l'arrêt automatique (arrêt à l'époque 7, poids de l'époque 4). Le sur-apprentissage est **léger** : l'écart reste de 1 à 2 points, et la perte de validation baisse encore un peu.",
    "Q8": "Les confusions les plus fréquentes : **tulipe prise pour tournesol** (4 photos) et **tournesol pris pour marguerite** (3 photos). Les photos mal classées sont souvent difficiles pour un humain aussi : des gros plans de pétales jaunes et orange (une tulipe vue de très près ressemble à un tournesol), un parterre de jardin où les fleurs sont minuscules, une marguerite à contre-jour, un bouquet de tournesols tenu par une personne, des marguerites jaunes prises pour des tournesols. Une erreur n'est pas celle du modèle : une photo d'**arbres sans feuilles** est rangée dans `tulipe`, c'est une **étiquette fausse**, à écarter (étape 2).",
    "Q9": "Pour le **corail**, l'erreur qui coûte cher est de dire « sain » à un corail blanchi : on rate une alerte. On surveille le **rappel de la classe `blanchi`**. Pour le **miconia**, rater un plant (« autre végétation » au lieu de « miconia ») laisse l'invasion progresser : on surveille le **rappel de `miconia`**, quitte à accepter des fausses alertes vérifiées par un humain. Pour les **déchets**, les erreurs coûtent à peu près pareil : l'exactitude globale et le rappel de chaque classe suffisent. Dans l'exemple des fleurs, aucune erreur ne coûte plus qu'une autre.",
    "Q10": "Le modèle paresseux répond toujours `tulipe` et obtient **30,9 %** ; notre modèle obtient **93,3 %**, soit **62 points de plus** : il bat nettement la baseline. Avec K classes équilibrées, le modèle paresseux obtient environ **1/K** (33 % pour 3 classes, 50 % pour 2). Si une classe faisait 90 % des photos, il obtiendrait **90 %** sans rien regarder : c'est le piège de l'exactitude de la séance 6, et la raison de regarder le rappel classe par classe.",
    "Q11": "Trois photos : `nouvelle_01` est une marguerite, reconnue à 92 % : **juste**. `nouvelle_12` est un tournesol pris pour une marguerite à 57 % : **faux, mais sous le seuil**, la photo part « à vérifier ». `nouvelle_16` est une **rose**, une fleur que le modèle n'a jamais vue : il répond **tulipe à 97 %**, avec une très forte confiance. Sur les 8 roses et pissenlits, **7 reçoivent une classe avec une confiance d'au moins 0,6** ; une seule rose (`nouvelle_25`, 54 %) part « à vérifier ». Le modèle ne peut pas répondre « je ne sais pas » : sa dernière couche donne K probabilités **dont la somme fait 1**, il répartit donc toujours toute la confiance entre les classes qu'il connaît. Le seuil attrape les hésitations, pas les fleurs inconnues. La parade : ajouter une classe `autre`, avec des photos de ce qu'on risque de rencontrer.",
    "Q12": "**0 photo** à vérifier avec le seuil 0,5 ; **3** avec 0,6 ; **9** avec 0,8, sur 32. À 0,8, les 3 erreurs sur les fleurs apprises partent toutes chez un humain, ainsi que 3 des 8 fleurs inconnues, mais aussi 3 photos que le modèle avait justes : on gagne en sûreté, on perd du temps de bénévole. Pour l'association du cas 2, avec des milliers de photos et deux bénévoles, un seuil d'environ **0,6 à 0,7** est raisonnable : environ 10 à 20 % des photos à revoir, et les erreurs les plus probables dedans. On le monte si une erreur coûte cher (un déchet dangereux mal compté), on le baisse si les bénévoles sont débordés. Et on vérifie régulièrement un échantillon des photos acceptées, car une fleur (ou un déchet) inconnue passe le seuil sans bruit (Q11).",
    "Q13": "Sur les six photos, **les zones rouges sont sur les fleurs** : le cœur de la marguerite, les tournesols du premier plan, les tulipes. Sur le champ de tournesols, le ciel reste bleu : le modèle ne s'appuie pas sur le paysage. Sur la photo d'une femme (une image de synthèse) au milieu des tournesols, la chaleur est surtout sur les tournesols, pas sur la personne : bon signe. Une photo pose une autre question : la deuxième montre des **rudbeckies**, des fleurs jaunes au cœur noir, rangées dans `tournesol` ; le modèle les appelle tournesol à 96 %, en regardant les bonnes zones, mais l'étiquette est discutable. Si les zones rouges tombaient sur le fond (le ciel, une pelouse, une personne), ce serait un **raccourci** : on reprendrait **l'étape 0**, collecter des photos plus variées (des marguerites sur fond sombre et sur fond clair, des tournesols avec et sans personne), et **l'étape 2**, écarter les photos où le sujet n'est pas la fleur.",
    "Q14 (bonus)": "Le fine-tuning fait passer la validation de **93,3 % à 94,9 %**, soit **+1,7 point**, pour environ une minute de calcul en plus sur ce portable. Mais 1,7 point sur 178 photos, c'est **3 photos** : un écart de cette taille peut venir du hasard. Et la courbe montre l'entraînement à 99,9 % pendant que la validation reste vers 94-95 %, et la perte de validation remonte dans les deux dernières époques : c'est le **sur-apprentissage** qui commence. Ici, le fine-tuning ne valait pas vraiment les minutes en plus : mieux vaut d'abord écarter les étiquettes fausses (Q8) et ajouter de la variété (Q13).",
    "Q15": "Ce modèle sait reconnaître, sur une photo, une marguerite, un tournesol ou une tulipe, avec environ 9 réponses justes sur 10 sur des photos qu'il n'avait jamais vues. Il ne sait pas dire « je ne sais pas » devant une autre fleur, et il n'a pas été testé sur des photos prises ailleurs, avec d'autres appareils : avant de s'en servir sur le terrain, il faut le tester là-bas. Il trie et propose ; les photos douteuses vont à un humain, et c'est un humain qui signe la décision.",
}


def APRES(racine):
    """Efface ce que l'exécution du corrigé a écrit à côté du carnet."""
    dossier = racine / "tp" / "classification_images"
    if os.environ.get("GARDER"):
        return
    # les sorties citent les zip locaux utilisés pendant l'exécution : on remet leurs adresses GitHub
    import nbformat as nbf
    carnet = dossier / f"{NOTEBOOK}_CORRIGE.ipynb"
    nb = nbf.read(carnet, as_version=4)
    brut = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/"
    for c in nb.cells:
        for o in c.get("outputs", []):
            if "text" in o:
                for rel in FICHIERS:
                    o["text"] = o["text"].replace(str(racine / rel), brut + rel)
            for cle in ["text/plain", "text/markdown"]:
                if cle in o.get("data", {}):
                    for rel in FICHIERS:
                        o["data"][cle] = o["data"][cle].replace(str(racine / rel), brut + rel)
    nbf.write(nb, carnet)
    for nom in ["mes_photos", "nouvelles", "photos_travail", "photos_app", "photos_val", "sorties"]:
        shutil.rmtree(dossier / nom, ignore_errors=True)
    for nom in ["predictions.csv", "mes_photos.zip", "nouvelles.zip"]:
        (dossier / nom).unlink(missing_ok=True)
