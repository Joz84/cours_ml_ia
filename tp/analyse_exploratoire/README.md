# TP 2 · Analyser : lire un jeu de données avant de conclure

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Joz84/cours_ml_ia/blob/main/tp/analyse_exploratoire/TP2_analyse_ELEVE.ipynb)

UE 5.7 « Data Science et IA » · Licence STES · Université de la Polynésie française

Le tableau est propre (TP 1). Avant de tracer quoi que ce soit, il faut le **lire** : combien de lignes, pour quoi, quand, avec quels trous, quelles valeurs typiques, quels liens. Ce TP suit trois outils du cours :
1. **la check-list de Peng**, pour ne pas se tromper sur l'emballage ;
2. **ydata-profiling**, qui fait tout d'un coup, *mais pas à la place de vous* ;
3. **les 7 regards**, plus les **deux réflexes de terrain** de Zuur (2010).

À la fin, vous écrirez **cinq constats chiffrés** : ce seront les questions du TP 3.

- **Durée :** environ 1 h 30.
- **En binôme**, dans **Google Colab**. Vous n'avez rien à installer à la main : une cellule installe ydata-profiling.
- **Méthode :** ouvrez le carnet avec le bouton « Ouvrir dans Colab », ou copiez chaque cellule de code de cette page dans Colab, **dans l'ordre**.
- **Vous ne modifiez que la cellule RÉGLAGES** et la cellule « validation extérieure », où vous tapez une valeur trouvée ailleurs.
- **Si une cellule plante**, copiez-la avec le message d'erreur dans ChatGPT ou Claude et demandez : « Explique-moi cette erreur simplement et corrige la cellule. »

> **Pour réutiliser ce carnet sur votre propre jeu de données :** changez la cellule RÉGLAGES (le fichier, le séparateur, la colonne de date, la colonne de catégorie, la mesure principale), puis exécutez tout le carnet dans l'ordre. Chaque cellule fonctionne sur n'importe quel tableau. S'il n'y a pas de date, mettez `COLONNE_DATE = None` : les cellules « dans le temps » le diront et passeront.

## Le rituel, à chaque étape

> **Je prédis** : je réponds aux questions *avant* d'exécuter la cellule.
> **J'exécute** : je lance la cellule.
> **Je constate** : j'écris deux phrases sur ce que j'observe, et je corrige ma prédiction si besoin.

## Les données

Le fichier `releves_meteo_propre.csv` est le tableau produit par le corrigé du TP 1 : **1 646 relevés journaliers** de 2024, pour 5 stations fictives de Tahiti. Colonnes : `date`, `station`, `altitude_m`, `pluie_mm`, `temp_min_C` et `temp_max_C`. Les trous ont été laissés vides (geste 5, option C).

**Rappel : ces données sont simulées**, avec un climat qui imite celui de Faa'a (voir le TP 1).

La question de départ, la première étape de Peng, est : **où et quand pleut-il le plus dans nos 5 stations ?**

---

## Le carnet

### Réglages · La seule cellule à modifier

```python
# ===================== RÉGLAGES : la seule cellule à modifier =====================
URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/clean/releves_meteo_propre.csv"  # le fichier : une URL, ou un chemin dans votre Drive
SEPARATEUR = ","                                   # le caractère entre les colonnes : "," ou ";"
COLONNE_DATE = "date"                              # la colonne des dates, ou None s'il n'y en a pas
COLONNE_CATEGORIE = "station"                      # la colonne qui regroupe les lignes
VALEUR = "pluie_mm"                                # la mesure principale qu'on étudie
AGREGATION = "sum"                                 # comment cumuler VALEUR sur une année : "sum" (pluie) ou "mean" (température)
COLONNES_A_RELIER = ["altitude_m", "pluie_mm"]     # regard 6 : deux colonnes de chiffres dont on cherche le lien
TITRE_RAPPORT = "Relevés météo de 5 stations de Tahiti, 2024"   # le titre du rapport ydata-profiling
DOSSIER_SORTIE = "/content/drive/MyDrive/TP_analyse"            # où écrire le rapport, dans votre Drive
# ===================================================================================
```

## Partie 1 · La check-list de Peng

### Peng 1 · Lire et vérifier l'emballage

> **Je prédis → J'exécute → Je constate**

On charge le tableau, puis on vérifie l'**emballage** : le nombre de lignes et de colonnes, les types et les cases remplies.

```python
import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(URL, sep=SEPARATEUR)
if COLONNE_DATE:
    df[COLONNE_DATE] = pd.to_datetime(df[COLONNE_DATE])
colonnes_num = list(df.select_dtypes("number").columns)

print("Lignes, colonnes :", df.shape)
df.info()
```

**Résultat attendu**

```text
Lignes, colonnes : (1646, 6)
RangeIndex: 1646 entries, 0 to 1645
Data columns (total 6 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   date        1646 non-null   datetime64[ns]
 1   station     1646 non-null   object
 2   altitude_m  1646 non-null   int64
 3   pluie_mm    1580 non-null   float64
 4   temp_min_C  1579 non-null   float64
 5   temp_max_C  1643 non-null   float64
```

### Peng 2 · Le haut et le bas

> **Je prédis → J'exécute → Je constate**

Les premières et les dernières lignes : est-ce bien ce qu'on attend ?

```python
display(df.head())
df.tail()
```

**Résultat attendu**

```text
        date station  altitude_m  pluie_mm  temp_min_C  temp_max_C
0 2024-11-27    Port           5      10.6        25.1        31.3
1 2024-04-21     Col         900      10.7        17.4        25.2
2 2024-07-22    Port           5       0.0        22.4        28.5
...
           date  station  altitude_m  pluie_mm  temp_min_C  temp_max_C
1644 2024-12-27  Plateau         450       0.0        22.2        29.7
1645 2024-01-29  Plateau         450       0.0        21.9        31.4
```

### Peng 3 · Vérifier ses n

> **Je prédis → J'exécute → Je constate**

Combien de lignes par catégorie, puis par catégorie **et** par mois ? C'est l'étape qu'on oublie le plus souvent, et celle qui évite les plus grosses erreurs.

```python
print("Lignes par", COLONNE_CATEGORIE, ":")
print(df[COLONNE_CATEGORIE].value_counts())

if COLONNE_DATE:
    print("\nLignes par", COLONNE_CATEGORIE, "et par mois :")
    display(pd.crosstab(df[COLONNE_CATEGORIE], df[COLONNE_DATE].dt.month))
else:
    print("Pas de colonne de date : on ne peut pas compter par mois.")
```

**Résultat attendu**

```text
Lignes par station :
station
Port       366
Lagon      366
Plateau    366
Vallée     366
Col        182

Lignes par station et par mois :
date     1   2   3   4   5   6   7   8   9   10  11  12
station
Col      31  29  31  30   0   0   0   0   0   0  30  31
Lagon    31  29  31  30  31  30  31  31  30  31  30  31
Plateau  31  29  31  30  31  30  31  31  30  31  30  31
Port     31  29  31  30  31  30  31  31  30  31  30  31
Vallée   31  29  31  30  31  30  31  31  30  31  30  31
```

**Questions**
- Q1 · Combien de relevés par station ? Qu'a `Col` de particulier, et pourquoi est-ce important avant de comparer les stations ?

### Peng 4 · Valider avec une source extérieure

> **Je prédis → J'exécute → Je constate**

Un chiffre ne vaut que s'il ressemble à ce qu'on sait déjà. Cherchez sur Internet **la normale annuelle de pluie d'une station Météo-France de Tahiti**, par exemple Faa'a, avec sa source. Tapez-la ci-dessous. La cellule la compare au cumul annuel de `VALEUR` pour la catégorie la plus proche, ici `Port`, une station de bord de mer comme Faa'a.

<!-- id: reference -->
```python
REFERENCE = None              # À COMPLÉTER : la valeur trouvée, par exemple 1500 (un nombre, sans unité)
SOURCE_REFERENCE = ""         # À COMPLÉTER : d'où vient-elle ? (site, station, période)
CATEGORIE_A_COMPARER = "Port" # la catégorie du tableau à comparer à cette référence

if COLONNE_DATE:
    par_an = df.groupby([COLONNE_CATEGORIE, df[COLONNE_DATE].dt.year])[VALEUR].agg(AGREGATION)
    cumul = par_an.groupby(level=0).mean()
else:
    cumul = df.groupby(COLONNE_CATEGORIE)[VALEUR].agg(AGREGATION)
print("Cumul annuel de", VALEUR, "par", COLONNE_CATEGORIE, ":")
print(cumul.round(1))

if REFERENCE is None:
    print("\nTapez une valeur de référence dans REFERENCE, puis relancez.")
else:
    ecart = cumul[CATEGORIE_A_COMPARER] / REFERENCE
    print(f"\n{CATEGORIE_A_COMPARER} : {cumul[CATEGORIE_A_COMPARER]:.0f} contre {REFERENCE} pour la référence ({SOURCE_REFERENCE})")
    print(f"Rapport : {ecart:.2f}, soit un écart de {(ecart - 1) * 100:+.0f} %")
```

**Résultat attendu** (avec la référence du corrigé)

```text
Cumul annuel de pluie_mm par station :
station
Col        2543.9
Lagon      1600.0
Plateau    2898.5
Port       1989.0
Vallée     4203.3

Port : 1989 contre 1579 pour la référence (Météo-France, station de Faa'a, moyenne 1991-2020 ...)
Rapport : 1.26, soit un écart de +26 %
```

Sans référence tapée, la cellule affiche seulement les cumuls et vous demande d'en taper une.

**Questions**
- Quelle référence avez-vous trouvée, et où ? L'écart avec `Port` vous paraît-il acceptable ? Qu'est-ce qui pourrait l'expliquer ?

## Partie 2 · ydata-profiling : tout d'un coup, mais pas à la place de vous

### Profil 1 · Installer et lancer le rapport

> **J'exécute**

ydata-profiling fabrique en une commande un rapport complet : vue d'ensemble, chaque colonne, les alertes, les corrélations et les interactions. La première cellule l'installe : cela prend environ une minute. Si Colab propose de **redémarrer la session**, acceptez, puis relancez le carnet depuis RÉGLAGES. Un avertissement « deprecated » peut s'afficher : il est sans conséquence pour ce TP.

```python
%pip install -q ydata-profiling
```

Le rapport s'affiche dans le carnet. Pour un gros tableau (des centaines de milliers de lignes), écrivez `ProfileReport(df, title=TITRE_RAPPORT, minimal=True)` : le rapport est bien plus rapide, mais il ne calcule ni les corrélations ni les interactions.

```python
from ydata_profiling import ProfileReport

rapport = ProfileReport(df, title=TITRE_RAPPORT)
rapport.to_notebook_iframe()
```

Enfin, on enregistre le rapport en HTML dans le Drive, pour pouvoir l'ouvrir et le partager.

```python
if DOSSIER_SORTIE.startswith("/content/drive"):
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except ImportError:
        DOSSIER_SORTIE = "."   # en dehors de Colab : on écrit à côté du carnet
os.makedirs(DOSSIER_SORTIE, exist_ok=True)
chemin_rapport = os.path.join(DOSSIER_SORTIE, "rapport_profil.html")
rapport.to_file(chemin_rapport)
print("Rapport enregistré :", chemin_rapport)
```

Répondez maintenant aux questions Q2 à Q6 **en lisant le rapport**, sans écrire de code.

### Q2 · La vue d'ensemble

Onglet **Overview**.

**Questions**
- Combien de lignes, de colonnes, de cases vides et de doublons le rapport annonce-t-il ? Est-ce cohérent avec le journal du TP 1 ?

### Q3 · Les alertes

Onglet **Alerts**, dans la vue d'ensemble.

**Questions**
- Quelles alertes le rapport affiche-t-il ? Pour chacune, dites si c'est **un problème** ou **une propriété des données**. Par exemple, « pluie_mm has … zeros » : c'est la pluie, pas une erreur.

### Q4 · La pluie, colonne par colonne

Section **Variables**, colonne `pluie_mm` (cliquez sur « More details »).

**Questions**
- Quelles sont la moyenne et la médiane de `pluie_mm` ? Quelle est la forme de l'histogramme ?
- Laquelle des deux, la moyenne ou la médiane, raconte la journée typique ? Pourquoi ?

### Q5 · Les corrélations

Section **Correlations**.

**Questions**
- Quelle paire de colonnes est la plus corrélée, en dehors de `station` et `altitude_m` qui disent la même chose ? Le signe est-il celui qu'on attend ?

### Q6 · Les interactions

Section **Interactions** : choisissez `altitude_m` d'un côté et `pluie_mm` de l'autre.

**Questions**
- Que montre la pluie en fonction de l'altitude ? Pourquoi les points forment-ils des colonnes ?

## Partie 3 · Les 7 regards

### Regard 1 · Décrire

> **Je prédis → J'exécute → Je constate**

`describe()` résume chaque colonne de chiffres : nombre de valeurs, moyenne, écart-type, minimum, quartiles et maximum. La première ligne remet l'affichage des graphiques dans le carnet : ydata-profiling l'a changé.

```python
%matplotlib inline
df.describe().round(1)
```

**Résultat attendu**

```text
       altitude_m  pluie_mm  temp_min_C  temp_max_C
count      1646.0    1580.0      1579.0      1643.0
mean        236.3       8.4        22.0        28.7
min           5.0       0.0        15.4        20.2
25%          10.0       0.0        20.7        27.1
50%         150.0       0.0        22.4        28.9
75%         450.0       7.6        23.5        30.4
max         900.0     231.2        26.4        36.3
std         289.6      20.7         2.0         2.4
```

La ligne `date` du tableau peut aussi apparaître, selon la version de pandas.

### Regard 2 · La répartition

> **Je prédis → J'exécute → Je constate**

L'histogramme de la mesure principale, avec sa moyenne et sa médiane.

**Questions**
- La moyenne et la médiane sont-elles proches ? Qu'est-ce que cela dit de la forme de la répartition ?

```python
moyenne, mediane = df[VALEUR].mean(), df[VALEUR].median()
print(f"Moyenne : {moyenne:.1f}   Médiane : {mediane:.1f}")

plt.figure(figsize=(8, 4))
plt.hist(df[VALEUR].dropna(), bins=40, color="#2a78d6")
plt.axvline(moyenne, color="#eb6834", linewidth=2, label=f"moyenne = {moyenne:.1f}")
plt.axvline(mediane, color="#555555", linewidth=2, linestyle="--", label=f"médiane = {mediane:.1f}")
plt.title("Répartition de " + VALEUR)
plt.xlabel(VALEUR)
plt.ylabel("Nombre de lignes")
plt.legend()
plt.show()
```

**Résultat attendu**

```text
Moyenne : 8.4   Médiane : 0.0
```

puis un histogramme très asymétrique : une énorme barre à 0 et une longue traîne jusqu'à plus de 200 mm.

### Regard 3 · Compter

> **Je prédis → J'exécute → Je constate**

`value_counts()` compte les lignes de chaque catégorie, en nombre puis en pourcentage.

```python
print(df[COLONNE_CATEGORIE].value_counts())
print()
print((df[COLONNE_CATEGORIE].value_counts(normalize=True) * 100).round(1))
```

**Résultat attendu**

```text
station
Port       366
Lagon      366
Plateau    366
Vallée     366
Col        182

station
Port       22.2
Lagon      22.2
Plateau    22.2
Vallée     22.2
Col        11.1
```

### Regard 4 · Grouper

> **Je prédis → J'exécute → Je constate**

`groupby` calcule la moyenne de chaque colonne de chiffres, catégorie par catégorie, avec en plus le nombre de lignes : on n'oublie jamais ses n.

**Questions**
- Quelle station reçoit le plus de pluie par jour ? Combien de fois plus que `Port` ?

```python
groupes = df.groupby(COLONNE_CATEGORIE)[colonnes_num].mean().round(1)
groupes["n"] = df[COLONNE_CATEGORIE].value_counts()
groupes.sort_values(VALEUR, ascending=False)
```

**Résultat attendu**

```text
         altitude_m  pluie_mm  temp_min_C  temp_max_C    n
station
Col           900.0      14.7        18.4        25.4  182
Vallée        150.0      12.0        22.5        29.1  366
Plateau       450.0       8.4        20.6        27.2  366
Port            5.0       5.6        23.5        30.1  366
Lagon          10.0       4.5        23.4        30.0  366
```

### Regard 5 · Suivre dans le temps

> **Je prédis → J'exécute → Je constate**

La moyenne de la mesure principale mois par mois, toutes catégories confondues.

**Questions**
- Quels mois sont les plus pluvieux ? Retrouvez-vous la saison des pluies (novembre à avril) ?

```python
if COLONNE_DATE:
    par_mois = df.groupby(df[COLONNE_DATE].dt.month)[VALEUR].mean()
    print(par_mois.round(1))
    plt.figure(figsize=(8, 4))
    plt.plot(par_mois.index, par_mois.values, marker="o", color="#2a78d6", linewidth=2)
    plt.title(VALEUR + " moyen par jour, mois par mois")
    plt.xlabel("Mois")
    plt.ylabel(VALEUR + " (moyenne par jour)")
    plt.xticks(range(1, 13))
    plt.grid(alpha=0.3)
    plt.show()
else:
    print("Pas de colonne de date : ce regard ne s'applique pas.")
```

**Résultat attendu**

```text
date
1     16.5
2     16.7
3     11.4
4      5.9
5      4.0
6      3.0
7      6.6
8      1.5
9      2.2
10     8.4
11     6.6
12    13.0
```

puis une courbe en « U » : haute de décembre à mars, basse de mai à septembre.

### Regard 6 · Relier

> **Je prédis → J'exécute → Je constate**

`corr()` donne la corrélation entre chaque paire de colonnes de chiffres : de -1 à 1, et 0 quand il n'y a pas de lien en ligne droite. Le nuage de points montre les deux colonnes de `COLONNES_A_RELIER`.

**Questions**
- Quelle paire est la plus corrélée ? Le lien entre altitude et pluie est-il fort ou faible, jour par jour ?

```python
display(df[colonnes_num].corr().round(2))

x, y = COLONNES_A_RELIER
r = df[x].corr(df[y])
plt.figure(figsize=(6, 4))
plt.scatter(df[x], df[y], s=10, alpha=0.4, color="#2a78d6")
plt.title(f"{y} selon {x} (r = {r:.2f})")
plt.xlabel(x)
plt.ylabel(y)
plt.grid(alpha=0.3)
plt.show()
```

**Résultat attendu**

```text
            altitude_m  pluie_mm  temp_min_C  temp_max_C
altitude_m        1.00      0.12       -0.83       -0.67
pluie_mm          0.12      1.00       -0.02       -0.09
temp_min_C       -0.83     -0.02        1.00        0.89
temp_max_C       -0.67     -0.09        0.89        1.00
```

puis un nuage en 5 colonnes verticales, une par altitude, intitulé « pluie_mm selon altitude_m (r = 0.12) ».

### Regard 7 · Croiser

> **Je prédis → J'exécute → Je constate**

`pivot_table` croise deux dimensions : ici la mesure principale moyenne par catégorie **et** par mois. Une case vide veut dire : aucune ligne.

**Questions**
- Le mois le plus pluvieux est-il le même pour toutes les stations ? Que remarquez-vous pour `Col` ?

```python
if COLONNE_DATE:
    croise = pd.pivot_table(df, index=COLONNE_CATEGORIE, columns=df[COLONNE_DATE].dt.month,
                            values=VALEUR, aggfunc="mean").round(1)
else:
    croise = pd.crosstab(df[COLONNE_CATEGORIE], df[VALEUR] > df[VALEUR].median())
croise
```

**Résultat attendu**

```text
date       1     2     3    4    5    6     7    8    9     10    11    12
station
Col      15.1  25.6  12.8  9.3  NaN  NaN   NaN  NaN  NaN   NaN   7.9  17.4
Lagon     7.9   5.0   5.2  3.4  4.7  1.8   2.3  1.3  2.5   4.2   1.8  13.6
Plateau  28.6   9.0  12.9  4.9  7.1  4.1   3.4  1.6  3.7   7.5  10.0   5.6
Port      7.9  10.4  10.7  2.5  1.8  2.8  14.0  1.3  1.1   1.0   4.0   9.6
Vallée   23.2  34.0  15.0  9.4  2.4  3.1   6.7  1.7  1.6  20.9   9.6  18.3
```

### Zuur 1 · Excès de zéros

> **Je prédis → J'exécute → Je constate**

Premier réflexe de terrain : beaucoup de zéros changent tout. Une moyenne, une corrélation ou une droite de régression se comportent mal quand la moitié des valeurs vaut 0.

**Questions**
- Quelle colonne a beaucoup de zéros ? Est-ce une erreur ?

```python
zeros = (df[colonnes_num] == 0).mean() * 100
print("Part de zéros, colonne par colonne (%) :")
print(zeros.round(1).sort_values(ascending=False))
```

**Résultat attendu**

```text
Part de zéros, colonne par colonne (%) :
pluie_mm      56.3
altitude_m     0.0
temp_min_C     0.0
temp_max_C     0.0
```

### Zuur 2 · Dépendance dans le temps et l'espace

> **Je prédis → J'exécute → Je constate**

Second réflexe : deux jours qui se suivent, ou deux stations voisines, se ressemblent. Leurs relevés ne sont pas indépendants. Une courbe par mois **et** par catégorie le montre.

**Questions**
- Les stations montent-elles et descendent-elles ensemble, au fil des mois ? Qu'est-ce que cela veut dire pour leur indépendance ?

```python
if COLONNE_DATE:
    courbes = df.groupby([df[COLONNE_DATE].dt.month, COLONNE_CATEGORIE])[VALEUR].mean().unstack()
    courbes.plot(figsize=(9, 4), marker="o", linewidth=1.5)
    plt.title(VALEUR + " moyen par jour, par mois et par " + COLONNE_CATEGORIE)
    plt.xlabel("Mois")
    plt.ylabel(VALEUR + " (moyenne par jour)")
    plt.xticks(range(1, 13))
    plt.grid(alpha=0.3)
    plt.show()
else:
    print("Pas de colonne de date : ce réflexe ne s'applique pas.")
```

**Résultat attendu**

Un graphique à 5 courbes, une par station : toutes hautes de décembre à mars et basses d'avril à septembre. La courbe de `Col` s'interrompt de mai à octobre.

### Q7 · Cinq constats chiffrés

À partir de tout ce qui précède, écrivez **cinq constats chiffrés**, au format imposé, par exemple : « Vallée reçoit 2,1 fois plus de pluie par jour que Port ». Ce seront les questions du TP 3, à transformer en graphiques.

**Questions**
- Vos cinq constats, chacun en une phrase avec un chiffre.

---

## Ce qu'il faut rendre

Un notebook Colab par binôme, avec les deux noms en haut. Il doit contenir :
- toutes les cellules, **exécutées**, avec leurs résultats visibles ;
- votre valeur de référence et sa source, dans la cellule « validation extérieure » ;
- vos réponses aux questions Q1 à Q7 et aux questions des regards, dans les cellules « Votre réponse » ;
- le rapport `rapport_profil.html`, enregistré dans votre Drive.

## Testé sur d'autres fichiers

Le carnet a été exécuté sur deux autres fichiers, **en ne changeant que la cellule RÉGLAGES** :
- **`rayonnement_pf.csv`** (TP de régression) avec `COLONNE_CATEGORIE = "ile"`, `VALEUR = "rayonnement"`, `AGREGATION = "mean"` : toutes les cellules tournent, ydata-profiling compris.
- **le jeu « iris » de scikit-learn**, sans date (`COLONNE_DATE = None`) : toutes les cellules tournent ; celles qui ont besoin du temps l'annoncent et passent.

Ce qui a dû être adapté, en plus des réglages : la catégorie à comparer dans la cellule « validation extérieure » (`CATEGORIE_A_COMPARER`), qui doit être une catégorie de votre tableau.

## Glossaire

| Mot | En une phrase |
|---|---|
| **Check-list de Peng** | Formuler la question, lire, vérifier l'emballage, regarder le haut et le bas, vérifier ses n, valider avec une source extérieure, tracer, essayer une solution simple, se poser les questions de suite. |
| **Vérifier ses n** | Compter les lignes de chaque groupe avant de comparer les groupes. |
| **Validation extérieure** | Comparer un chiffre du tableau à une valeur connue ailleurs. |
| **ydata-profiling** | Un rapport automatique complet : tout d'un coup, mais pas à la place de vous. |
| **Moyenne / médiane** | La moyenne est tirée par les valeurs extrêmes ; la médiane coupe les données en deux moitiés. |
| **Corrélation** | Un nombre de -1 à 1 qui mesure un lien en ligne droite entre deux colonnes. |
| **Excès de zéros** | Beaucoup de valeurs à 0 : une propriété des données qui fausse moyennes et corrélations. |
| **Dépendance** | Des mesures proches dans le temps ou l'espace se ressemblent : elles ne sont pas indépendantes. |
