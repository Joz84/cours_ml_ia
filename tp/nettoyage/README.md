# TP 1 · Nettoyer un jeu de données en 9 gestes

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Joz84/cours_ml_ia/blob/main/tp/nettoyage/TP1_nettoyage_ELEVE.ipynb)

UE 5.7 « Data Science et IA » · Licence STES · Université de la Polynésie française

Un tableau de données n'arrive presque jamais propre : dates écrites de trois façons, virgules à la place des points, cases vides, lignes en double, températures impossibles… Avant toute analyse, il faut **nettoyer**. Dans ce TP, vous allez nettoyer les relevés météo de 5 stations de Tahiti en suivant **les 9 gestes du cours, dans l'ordre**, et garder la trace de chaque décision dans un **journal de nettoyage**.

- **Durée :** environ 1 h 30.
- **En binôme**, dans **Google Colab**. Vous n'avez rien à installer.
- **Méthode :** ouvrez le carnet avec le bouton « Ouvrir dans Colab » ci-dessus, ou copiez chaque cellule de code de cette page dans une nouvelle cellule Colab. Exécutez-les **dans l'ordre**, avec Maj + Entrée.
- **Vous ne modifiez que ce qu'on vous dit de modifier** : la cellule RÉGLAGES et les cellules DÉCISION.
- **Si une cellule plante**, copiez-la avec le message d'erreur dans ChatGPT ou Claude et demandez : « Explique-moi cette erreur simplement et corrige la cellule. »

> **Pour réutiliser ce carnet sur votre propre jeu de données :** changez la cellule RÉGLAGES (le fichier, le séparateur, les noms de colonnes, les bornes et les règles), puis exécutez tout le carnet dans l'ordre. Toutes les autres cellules fonctionnent sur n'importe quel tableau : elles parcourent les colonnes et détectent elles-mêmes les types. Seules les cellules DÉCISION vous demandent de choisir.

## Les 9 gestes du nettoyage

Le cadre est celui de Van den Broeck et al. (2005) : **dépister, diagnostiquer, traiter, documenter**. Les 9 gestes se répartissent en trois sortes.

| Geste | En une ligne | Sorte | Où sont les cellules |
|---|---|---|---|
| **1 Regarder** | Travailler sur une copie, et se poser trois questions avant de toucher à quoi que ce soit. | c'est un jugement | 2 cellules + 3 questions |
| **2 Typer** | Que les dates soient des dates, les nombres des nombres, le texte du texte. | la machine le fait | dépistage, puis action |
| **3 Trier** | Ne garder que les colonnes utiles. | la machine le fait | dépistage, puis action (liste dans RÉGLAGES) |
| **4 Structurer** | Une ligne par observation, une unité par colonne : passer du large au long. | c'est un jugement | détection, puis démonstration de `melt` |
| **5 Trous** | Vide ne veut pas dire zéro : décider quoi faire des cases vides. | la machine dépiste, vous décidez | dépistage, puis **DÉCISION** (options A, B, C) |
| **6 Doublons** | Une même observation ne compte qu'une fois. | la machine le fait | dépistage, puis action |
| **7 Harmoniser** | Une chose, une écriture (`Port`, pas `PORT ` ni ` port`). | la machine dépiste, vous décidez | dépistage, normalisation, puis **DÉCISION** (correspondances) |
| **8 Impossibles et incohérentes** | D'abord les min et max, puis les règles entre colonnes. | la machine dépiste, vous décidez | dépistage, puis **2 DÉCISIONS** (options A, B, C) |
| **9 Tracer** | Écrire le journal de nettoyage, pour que quelqu'un d'autre puisse refaire la même chose. | c'est un jugement | tout au long du carnet, puis une cellule finale |

Les cellules **DÉCISION** sont les seules, avec RÉGLAGES, où vous tapez quelque chose : la lettre de l'option choisie et **une phrase de justification**. Chaque décision s'ajoute automatiquement au journal.

## Le rituel, à chaque étape

> **Je prédis** : je réponds aux questions *avant* d'exécuter la cellule.
> **J'exécute** : je lance la cellule.
> **Je constate** : j'écris deux phrases sur ce que j'observe, et je corrige ma prédiction si besoin.

## Les données

Le fichier `releves_meteo_brut.csv` contient les relevés journaliers de l'année 2024 de **5 stations de Tahiti** : `Port`, `Lagon`, `Plateau`, `Vallée` et `Col`. Colonnes : `date`, `station`, `altitude_m`, `pluie_mm`, `temp_min_C`, `temp_max_C`, plus deux colonnes `code_agent` et `remarque`. Le séparateur est le point-virgule.

**Attention : ces données sont simulées, et volontairement salies.** Les stations sont fictives. Leur climat imite celui de Faa'a (Météo-France) : la station `Port` reçoit à peu près la pluie de Faa'a. Le script [`scripts/make_releves_brut.py`](../../scripts/make_releves_brut.py) les fabrique et liste tous les défauts cachés dedans. Ne le lisez pas avant la fin du TP : il dévoile les réponses.

---

## Le carnet

### Réglages · La seule cellule à modifier

Pour ce TP, ne changez rien : les réglages sont déjà ceux des relevés météo. Pour un autre fichier, c'est ici, et seulement ici, que vous changez les noms.

```python
# ===================== RÉGLAGES : la seule cellule à modifier =====================
URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/raw/releves_meteo_brut.csv"  # le fichier : une URL, ou un chemin dans votre Drive
SEPARATEUR = ";"                                   # le caractère entre les colonnes : "," ou ";"
COLONNE_DATE = "date"                              # la colonne des dates, ou None s'il n'y en a pas
COLONNE_CATEGORIE = "station"                      # la colonne qui regroupe les lignes, ou None
COLONNES_A_RETIRER = ["code_agent", "remarque"]    # geste 3 : les colonnes inutiles
CODES_TROU = [-999, -9999]                         # geste 5 : les nombres qui veulent dire « pas de mesure »
BORNES = {"temp_min_C": (5, 35), "temp_max_C": (10, 40), "pluie_mm": (0, 500)}  # geste 8 : (minimum, maximum) possibles
REGLES = ["temp_min_C <= temp_max_C", "pluie_mm >= 0"]   # geste 8 : règles entre colonnes
NOM_SORTIE = "releves_meteo_propre"                # geste 9 : nom du fichier propre
DOSSIER_SORTIE = "/content/drive/MyDrive/TP_nettoyage"   # geste 9 : le dossier de votre Drive où tout sera écrit
# ===================================================================================
```

### Outils · Préparer le carnet

Cette cellule importe les bibliothèques et crée le **journal de nettoyage** (geste 9) : une liste vide, qui se remplira au fil du carnet grâce à la fonction `noter`. Exécutez-la sans la modifier.

```python
import os
import re
import unicodedata
import numpy as np
import pandas as pd

journal = []   # geste 9 : le journal de nettoyage

def noter(geste, constat, decision, justification):
    """Ajoute une ligne au journal de nettoyage."""
    journal.append({"geste": geste, "constat": constat, "décision": decision, "justification": justification})
    print("Noté dans le journal :", geste, "->", decision)

print("Outils prêts.")
```

**Résultat attendu**

```text
Outils prêts.
```

### Geste 1 · Regarder

> **Je prédis → J'exécute → Je constate**

**On ne travaille jamais sur l'original.** La cellule charge le fichier dans `df_brut`, qu'on ne touchera plus, et crée une copie `df` sur laquelle on travaille. Tout est lu comme du texte : aucune transformation ne se fait sans qu'on la voie.

```python
df_brut = pd.read_csv(URL, sep=SEPARATEUR, dtype=str, keep_default_na=False, na_values=[""])
df = df_brut.copy()   # on ne travaille JAMAIS sur l'original : df_brut reste intact

print("Lignes, colonnes :", df.shape)
noter("1 Regarder", f"{df.shape[0]} lignes, {df.shape[1]} colonnes", "travail sur une copie", "L'original doit rester intact.")
df.info()
```

**Résultat attendu**

```text
Lignes, colonnes : (1686, 8)
Noté dans le journal : 1 Regarder -> travail sur une copie
RangeIndex: 1686 entries, 0 to 1685
Data columns (total 8 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   date        1686 non-null   object
 1   station     1686 non-null   object
 2   altitude_m  1686 non-null   object
 3   pluie_mm    1653 non-null   object
 4   temp_min_C  1653 non-null   object
 5   temp_max_C  1686 non-null   object
 6   code_agent  1686 non-null   object
 7   remarque    604 non-null    object
```

Toutes les colonnes sont du texte (`object`) : c'est voulu, on les typera au geste 2.

Puis le début et la fin du tableau :

```python
display(df.head())
df.tail()
```

**Résultat attendu**

```text
          date station altitude_m pluie_mm temp_min_C temp_max_C code_agent       remarque
0   27 nov. 24    Port          5     10.6       25.1       31.3       AG04            RAS
1   21 avr. 24     Col        900     10.7       17,4       25,2       AG08            NaN
2   2024-07-22    Port          5      0.0       22.4       28,5       AG01  relevé tardif
...
            date  station altitude_m pluie_mm temp_min_C temp_max_C code_agent         remarque
1682  2024-07-15  Plateau        450      0,0         NA       25.4       AG03              NaN
1683  08/05/2024   Vallée        150      0.0       22.5       29.4       AG06              NaN
```

**Questions**
- Que représente une ligne du tableau ?
- Qu'est-ce qui vous paraît déjà bizarre ? Citez au moins trois choses.
- Quelle question voudriez-vous poser à ces données une fois propres ?

### Geste 2 · Typer

> **Je prédis → J'exécute → Je constate**

*La machine le fait.* **Dépistage** : on regarde quelques valeurs de chaque colonne.

```python
print("Exemples de valeurs, colonne par colonne :")
for colonne in df.columns:
    print("-", colonne, ":", list(df[colonne].dropna().unique()[:5]))
```

**Résultat attendu**

```text
Exemples de valeurs, colonne par colonne :
- date : ['27 nov. 24', '21 avr. 24', '2024-07-22', '22 sept. 24', '1 juin 24']
- station : ['Port', 'Col', 'Lagon', 'Plateau', 'Vallée']
- altitude_m : ['5', '900', '10', '450', '150']
- pluie_mm : ['10.6', '10.7', '0.0', '11.0', '-999']
- temp_min_C : ['25.1', '17,4', '22.4', '22.0', '22,7']
- temp_max_C : ['31.3', '25,2', '28,5', '28,1', '28.7']
- code_agent : ['AG04', 'AG08', 'AG01', 'AG02', 'AG05']
- remarque : ['RAS', 'relevé tardif', 'capteur nettoyé']
```

**Action** : la cellule convertit la colonne de dates, quel que soit le format, et chaque colonne qui contient au moins 80 % de nombres. Elle remplace la virgule par un point et retire les unités (`48 mm` devient `48`). Avant de les passer en vide, elle **affiche les cases qui n'ont pas pu être converties**.

```python
MOIS_FR = {"janv.": "01", "févr.": "02", "mars": "03", "avr.": "04", "mai": "05", "juin": "06",
           "juil.": "07", "août": "08", "sept.": "09", "oct.": "10", "nov.": "11", "déc.": "12"}

def en_date(texte):
    """Lit les dates écrites « 2024-01-03 », « 03/01/2024 » ou « 3 janv. 24 »."""
    texte = texte.str.strip()
    for mois, numero in MOIS_FR.items():
        texte = texte.str.replace(" " + mois + " ", "/" + numero + "/", regex=False)
    iso = pd.to_datetime(texte, format="%Y-%m-%d", errors="coerce")
    jour_d_abord = pd.to_datetime(texte, dayfirst=True, format="mixed", errors="coerce")
    return iso.fillna(jour_d_abord)

def en_nombre(texte):
    """Lit « 12,5 » (virgule) et « 48 mm » (unité). Ce qui reste illisible devient vide."""
    texte = texte.str.strip().str.replace(",", ".", regex=False)
    texte = texte.str.replace(r"\s*[a-zA-Z°%]+\.?$", "", regex=True)
    return pd.to_numeric(texte, errors="coerce")

non_convertis = []
for colonne in df.columns:
    avant = df[colonne]
    if colonne == COLONNE_DATE:
        apres = en_date(avant)
    else:
        apres = en_nombre(avant)
        if apres.notna().sum() < 0.8 * avant.notna().sum():
            continue   # moins de 80 % de nombres : c'est une colonne de texte, on n'y touche pas
    rates = avant[avant.notna() & apres.isna()]
    for valeur, nombre in rates.value_counts().items():
        non_convertis.append({"colonne": colonne, "valeur illisible": valeur, "nombre": nombre})
    df[colonne] = apres

print("Cases qui n'ont pas pu être converties (elles deviennent vides) :")
display(pd.DataFrame(non_convertis))
noter("2 Typer", f"{sum(x['nombre'] for x in non_convertis)} cases illisibles : "
      + ", ".join(f"{x['valeur illisible']} ({x['colonne']})" for x in non_convertis),
      "passées en vide", "Ce ne sont pas des valeurs lisibles : on ne devine pas ce qu'elles valaient.")
df.dtypes
```

**Résultat attendu**

```text
Cases qui n'ont pas pu être converties (elles deviennent vides) :
      colonne valeur illisible  nombre
0    pluie_mm               NA      16
1  temp_min_C               NA      18
2  temp_max_C       vingt-deux       1
Noté dans le journal : 2 Typer -> passées en vide
date          datetime64[ns]
station               object
altitude_m             int64
pluie_mm             float64
temp_min_C           float64
temp_max_C           float64
code_agent            object
remarque              object
```

Les dates sont devenues des dates (`datetime64`), les mesures des nombres (`float64`). `station`, `code_agent` et `remarque` restent du texte.

**Questions**
- Quelles cases n'ont pas pu être converties ? Pour chacune, est-ce une vraie valeur perdue ou une absence de mesure ?
- Le `vingt-deux` aurait-il pu être converti par un humain ? Pourquoi la machine ne le fait-elle pas ?

### Geste 3 · Trier

> **Je prédis → J'exécute → Je constate**

*La machine le fait.* **Dépistage** : pour chaque colonne, son type, son taux de remplissage et son nombre de valeurs différentes.

```python
pd.DataFrame({"type": df.dtypes.astype(str),
              "remplissage_%": (df.notna().mean() * 100).round(1),
              "valeurs_differentes": df.nunique()})
```

**Résultat attendu**

```text
                      type  remplissage_%  valeurs_differentes
date        datetime64[ns]          100.0                  366
station             object          100.0                    9
altitude_m           int64          100.0                    5
pluie_mm           float64           97.1                  327
temp_min_C         float64           97.0                  108
temp_max_C         float64           99.9                  124
code_agent          object          100.0                    8
remarque            object           35.8                    3
```

**Action** : on retire les colonnes listées dans `COLONNES_A_RETIRER` (RÉGLAGES).

```python
df = df.drop(columns=COLONNES_A_RETIRER)
noter("3 Trier", "colonnes retirées : " + ", ".join(COLONNES_A_RETIRER), "retirées",
      "Elles ne servent à aucune question sur la météo.")
print("Colonnes gardées :", list(df.columns))
```

**Résultat attendu**

```text
Noté dans le journal : 3 Trier -> retirées
Colonnes gardées : ['date', 'station', 'altitude_m', 'pluie_mm', 'temp_min_C', 'temp_max_C']
```

**Questions**
- Pourquoi `remarque` est-elle peu utile ici, d'après son taux de remplissage ?

### Geste 4 · Structurer

> **Je prédis → J'exécute → Je constate**

*C'est un jugement.* Un tableau **long** a une ligne par observation. Un tableau **large** étale une même mesure sur plusieurs colonnes, par exemple une colonne par mois. La cellule **détecte** un tableau large : elle cherche les colonnes dont le nom ressemble à un mois ou à une année.

```python
MOTS_DE_TEMPS = r"^(janv|jan|fev|fév|feb|mars|mar|avr|apr|mai|may|juin|jun|juil|jul|aout|août|aug|sept|sep|oct|nov|dec|déc|(19|20)\d\d)"

def colonnes_de_temps(tableau):
    """Les colonnes dont le nom ressemble à un mois ou à une année : le signe d'un tableau LARGE."""
    return [c for c in tableau.columns if re.match(MOTS_DE_TEMPS, str(c).strip().lower())]

if len(colonnes_de_temps(df)) >= 3:
    print("Tableau LARGE probable : ces colonnes sont des mois ou des années :", colonnes_de_temps(df))
    noter("4 Structurer", "tableau large", "à passer en long avec melt", "Une ligne doit être une observation.")
else:
    print("Tableau déjà au format LONG : aucune colonne ne ressemble à un mois ou à une année.")
    noter("4 Structurer", "tableau déjà long", "rien à faire", "Une ligne est déjà un relevé (un jour, une station).")
```

**Résultat attendu**

```text
Tableau déjà au format LONG : aucune colonne ne ressemble à un mois ou à une année.
Noté dans le journal : 4 Structurer -> rien à faire
```

**Démonstration sur un tableau large** : la pluie de chaque station, avec une colonne par mois.

```python
URL_EXEMPLE_LARGE = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/raw/pluie_mensuelle_large.csv"
df_large = pd.read_csv(URL_EXEMPLE_LARGE)

print("Colonnes de temps trouvées :", colonnes_de_temps(df_large))
df_large
```

**Résultat attendu**

```text
Colonnes de temps trouvées : ['janvier', 'fevrier', 'mars', 'avril', 'mai', 'juin', 'juillet', 'aout', 'septembre', 'octobre', 'novembre', 'decembre']
   station  janvier  fevrier   mars  avril    mai   juin  ...  novembre  decembre
0     Port    237.5    282.0  336.6   72.0   66.0   83.9  ...     120.8     297.6
1    Lagon    245.8    151.4  159.5  105.0  144.9   54.6  ...      53.7     420.3
2   Vallée    729.3    884.4  450.3  282.0   73.4   88.7  ...     291.2     568.8
3  Plateau    886.9    260.0  380.7  137.0  219.2  120.6  ...     297.8     156.6
4      Col    497.3    725.1  397.4  281.8    NaN    NaN  ...     225.0     521.8
```

`melt` fait passer le tableau du large au long : une ligne par station et par mois. **Si votre propre tableau est large**, remplacez `df_large` par `df` et adaptez les trois noms en haut de la cellule.

```python
COLONNES_FIXES = ["station"]   # À ADAPTER : les colonnes qui ne sont PAS des mois
NOM_VARIABLE = "mois"          # À ADAPTER : le nom de la colonne qui recevra les noms des mois
NOM_VALEUR = "pluie_mm"        # À ADAPTER : le nom de la colonne qui recevra les valeurs

df_long = df_large.melt(id_vars=COLONNES_FIXES, var_name=NOM_VARIABLE, value_name=NOM_VALEUR)
print("Large :", df_large.shape, "-> long :", df_long.shape)
df_long.head(8)
```

**Résultat attendu**

```text
Large : (5, 13) -> long : (60, 3)
   station     mois  pluie_mm
0     Port  janvier     237.5
1    Lagon  janvier     245.8
2   Vallée  janvier     729.3
3  Plateau  janvier     886.9
4      Col  janvier     497.3
5     Port  fevrier     282.0
...
```

**Questions**
- Le tableau des relevés est-il large ou long ?
- Combien de lignes compte la version longue du tableau mensuel ? Pourquoi ce nombre ?
- Pourquoi les mois de mai à octobre de `Col` sont-ils vides, et pas égaux à zéro ?

### Geste 5 · Trous

> **Je prédis → J'exécute → Je constate**

*La machine dépiste, vous décidez.* **Vide ne veut pas dire zéro.** D'abord, les **codes de trou** : des nombres comme `-999` qui veulent dire « pas de mesure » (liste dans RÉGLAGES).

```python
colonnes_num = list(df.select_dtypes("number").columns)
for colonne in colonnes_num:
    n = df[colonne].isin(CODES_TROU).sum()
    if n > 0:
        print(colonne, ":", n, "valeurs codées", CODES_TROU, "-> ce ne sont pas des mesures")
```

**Résultat attendu**

```text
pluie_mm : 18 valeurs codées [-999, -9999] -> ce ne sont pas des mesures
temp_min_C : 20 valeurs codées [-999, -9999] -> ce ne sont pas des mesures
```

Ces codes deviennent de vrais trous, sinon ils fausseraient toutes les moyennes.

```python
n_codes = int(df[colonnes_num].isin(CODES_TROU).sum().sum())
df[colonnes_num] = df[colonnes_num].replace(CODES_TROU, np.nan)
noter("5 Trous", f"{n_codes} valeurs codées {CODES_TROU}", "transformées en trous",
      "Un code -999 n'est pas une mesure : il fausserait les moyennes.")
```

**Dépistage** : les trous de chaque colonne, en nombre et en pourcentage.

```python
trous = pd.DataFrame({"trous": df.isna().sum(), "trous_%": (df.isna().mean() * 100).round(1)})
trous = trous[trous["trous"] > 0]
print(len(trous), "colonnes contiennent des trous :", list(trous.index))
trous
```

**Résultat attendu**

```text
3 colonnes contiennent des trous : ['pluie_mm', 'temp_min_C', 'temp_max_C']
            trous  trous_%
pluie_mm       67      4.0
temp_min_C     71      4.2
temp_max_C      1      0.1
```

**DÉCISION** · Que faire des cases vides ?
- **Option A : supprimer les lignes** qui ont au moins un trou. C'est simple, mais on perd aussi les autres mesures de ces lignes.
- **Option B : remplir par la médiane de la station**. Le tableau est complet, mais on invente des valeurs.
- **Option C : laisser vide et le noter**. On ne perd rien et on n'invente rien, mais il faudra en tenir compte dans l'analyse.

Tapez la lettre choisie et votre justification, puis exécutez **les quatre cellules** : seule l'option choisie agit.

<!-- id: decision_trous -->
```python
decision_trous = ""        # DÉCISION : "A", "B" ou "C"
justification_trous = ""   # DÉCISION : une phrase, pourquoi ce choix ?
print("Option choisie :", decision_trous or "AUCUNE : tapez A, B ou C")
```

```python
# Option A : supprimer les lignes qui ont un trou
if decision_trous == "A":
    avant = len(df)
    df = df.dropna()
    noter("5 Trous", f"{avant - len(df)} lignes avec au moins un trou", "lignes supprimées", justification_trous)
else:
    print("Option A non choisie : rien n'est fait.")
```

```python
# Option B : remplir par la médiane de la catégorie (ou de la colonne)
if decision_trous == "B":
    n = int(df[colonnes_num].isna().sum().sum())
    for colonne in colonnes_num:
        if COLONNE_CATEGORIE:
            mediane = df.groupby(COLONNE_CATEGORIE)[colonne].transform("median")
        else:
            mediane = df[colonne].median()
        df[colonne] = df[colonne].fillna(mediane)
    noter("5 Trous", f"{n} trous dans les colonnes de chiffres", "remplis par la médiane", justification_trous)
else:
    print("Option B non choisie : rien n'est fait.")
```

```python
# Option C : laisser vide, et le noter
if decision_trous == "C":
    noter("5 Trous", f"{int(df.isna().sum().sum())} cases vides", "laissées vides", justification_trous)
else:
    print("Option C non choisie : rien n'est fait.")
```

**Questions**
- Pourquoi un trou dans `pluie_mm` ne doit-il surtout pas être remplacé par 0 ?
- Quelle option avez-vous choisie, et que faudra-t-il garder en tête pendant l'analyse ?

### Geste 6 · Doublons

> **Je prédis → J'exécute → Je constate**

*La machine le fait.* **Dépistage** : les lignes recopiées à l'identique.

```python
print(df.duplicated().sum(), "lignes sont des copies exactes d'une ligne déjà vue.")
df[df.duplicated(keep=False)].sort_values(list(df.columns[:2])).head(6)
```

**Résultat attendu**

```text
40 lignes sont des copies exactes d'une ligne déjà vue.
           date station  altitude_m  pluie_mm  temp_min_C  temp_max_C
45   2024-01-02     Col         900       0.0        18.7        26.8
1140 2024-01-02     Col         900       0.0        18.7        26.8
85   2024-01-25  Vallée         150       0.0        25.2        32.5
504  2024-01-25  Vallée         150       0.0        25.2        32.5
...
```

**Action** : on garde une seule fois chaque ligne.

```python
avant = len(df)
df = df.drop_duplicates()
noter("6 Doublons", f"{avant - len(df)} lignes en double", "supprimées", "Une même observation ne doit compter qu'une fois.")
print("Lignes :", avant, "->", len(df))
```

**Résultat attendu**

```text
Noté dans le journal : 6 Doublons -> supprimées
Lignes : 1686 -> 1646
```

**Questions**
- Que se passerait-il pour la pluie totale d'une station si on gardait les doublons ?

### Geste 7 · Harmoniser

> **Je prédis → J'exécute → Je constate**

*La machine dépiste, vous décidez.* **Une chose, une écriture.** **Dépistage** : toutes les écritures de chaque colonne de texte, avec leur nombre.

```python
colonnes_texte = list(df.select_dtypes(exclude=["number", "datetime"]).columns)
for colonne in colonnes_texte:
    print(colonne, ":", df[colonne].value_counts().to_dict())
```

**Résultat attendu**

```text
station : {'Lagon': 366, 'Plateau': 366, 'Vallée': 354, 'Port': 306, 'Col': 182, 'port': 23, 'PORT ': 19, ' Port': 18, 'Vallee': 12}
```

**Normalisation** : la machine retire les espaces autour du texte et unifie les majuscules (`PORT ` et ` port` deviennent `Port`). Elle signale ensuite les écritures qui ne diffèrent que par un accent.

```python
def sans_accent(texte):
    return "".join(c for c in unicodedata.normalize("NFD", texte) if unicodedata.category(c) != "Mn").lower()

for colonne in colonnes_texte:
    propre = df[colonne].str.strip()
    ecriture = propre.groupby(propre.str.lower()).agg(lambda s: s.value_counts().index[0])
    df[colonne] = propre.str.lower().map(ecriture)   # l'écriture la plus fréquente gagne
    valeurs = pd.Series(df[colonne].dropna().unique())
    proches = valeurs.groupby(valeurs.map(sans_accent)).apply(list)
    for groupe in proches[proches.str.len() > 1]:
        print("À VÉRIFIER dans", colonne, ": ces écritures se ressemblent :", groupe)
    print(colonne, ":", df[colonne].value_counts().to_dict())
```

**Résultat attendu**

```text
À VÉRIFIER dans station : ces écritures se ressemblent : ['Vallée', 'Vallee']
station : {'Port': 366, 'Lagon': 366, 'Plateau': 366, 'Vallée': 354, 'Col': 182, 'Vallee': 12}
```

Les quatre écritures de `Port` sont devenues une seule. Reste `Vallee`, que la machine signale sans le corriger : c'est à vous de décider.

**DÉCISION** · Quelles écritures faut-il encore corriger ? Complétez le dictionnaire `CORRESPONDANCES` avec des paires `"écriture à corriger": "bonne écriture"` (par exemple `{"PORT": "Port"}`), écrivez votre justification, puis exécutez.

<!-- id: decision_harmoniser -->
```python
CORRESPONDANCES = {}               # DÉCISION : par exemple {"PORT": "Port"}
justification_harmoniser = ""      # DÉCISION : une phrase, pourquoi ?

for colonne in colonnes_texte:
    df[colonne] = df[colonne].replace(CORRESPONDANCES)
noter("7 Harmoniser", f"correspondances : {CORRESPONDANCES}", "écritures unifiées", justification_harmoniser)
for colonne in colonnes_texte:
    print(colonne, ":", df[colonne].value_counts().to_dict())
```

**Questions**
- Combien d'écritures différentes de `Port` y avait-il au départ ? Et après ?
- Pourquoi la machine ne corrige-t-elle pas seule `Vallee` en `Vallée` ?

### Geste 8 · Impossibles et incohérentes

> **Je prédis → J'exécute → Je constate**

*La machine dépiste, vous décidez.* **Dépistage**, en trois temps :
- d'abord le minimum et le maximum de chaque colonne de chiffres ;
- puis les valeurs **impossibles**, hors des `BORNES` des RÉGLAGES ;
- enfin les lignes **incohérentes**, qui ne respectent pas une des `REGLES` entre colonnes.

```python
display(df[colonnes_num].agg(["min", "max"]))

impossibles = pd.Series(False, index=df.index)
for colonne, (mini, maxi) in BORNES.items():
    hors = (df[colonne] < mini) | (df[colonne] > maxi)
    impossibles = impossibles | hors
    print("IMPOSSIBLE -", colonne, ":", hors.sum(), "valeurs hors de", (mini, maxi), ":", df.loc[hors, colonne].tolist())

incoherentes = pd.Series(False, index=df.index)
for regle in REGLES:
    colonnes_regle = [c for c in df.columns if re.search(r"\b" + re.escape(c) + r"\b", regle)]
    viole = df[colonnes_regle].notna().all(axis=1) & ~df.eval(regle)
    incoherentes = incoherentes | viole
    print("INCOHÉRENT - règle", regle, ":", viole.sum(), "lignes ne la respectent pas")
```

**Résultat attendu**

```text
     altitude_m  pluie_mm  temp_min_C  temp_max_C
min           5       0.0        -5.0        19.0
max         900     231.2        33.2       270.0
IMPOSSIBLE - temp_min_C : 1 valeurs hors de (5, 35) : [-5.0]
IMPOSSIBLE - temp_max_C : 2 valeurs hors de (10, 40) : [270.0, 99.0]
IMPOSSIBLE - pluie_mm : 0 valeurs hors de (0, 500) : []
INCOHÉRENT - règle temp_min_C <= temp_max_C : 5 lignes ne la respectent pas
INCOHÉRENT - règle pluie_mm >= 0 : 0 lignes ne la respectent pas
```

Les valeurs **étonnantes** sont à plus de 3 écarts-types de la moyenne de leur catégorie. Elles sont rares, mais pas forcément fausses : on les vérifie, et on les **garde par défaut**.

```python
for colonne in colonnes_num:
    valeurs = df[colonne]
    if colonne in BORNES:
        valeurs = valeurs.where(valeurs.between(*BORNES[colonne]))   # sans les impossibles
    groupes = df[COLONNE_CATEGORIE] if COLONNE_CATEGORIE else pd.Series(0, index=df.index)
    ecart = (valeurs - valeurs.groupby(groupes).transform("mean")) / valeurs.groupby(groupes).transform("std")
    etonnants = ecart.abs() > 3
    if etonnants.sum() > 0:
        print("ÉTONNANT -", colonne, ":", etonnants.sum(), "valeurs ; les plus fortes :",
              sorted(df.loc[etonnants, colonne].tolist())[-5:])
noter("8 Étonnantes", "valeurs à plus de 3 écarts-types", "gardées", "Rares mais plausibles : on vérifie, on ne supprime pas.")
df[impossibles | incoherentes]
```

**Résultat attendu**

```text
ÉTONNANT - pluie_mm : 34 valeurs ; les plus fortes : [180.0, 190.4, 210.0, 212.9, 231.2]
ÉTONNANT - temp_min_C : 6 valeurs ; les plus fortes : [24.3, 28.8, 29.8, 30.5, 33.2]
ÉTONNANT - temp_max_C : 6 valeurs ; les plus fortes : [20.2, 21.9, 23.0, 23.6, 36.3]
Noté dans le journal : 8 Étonnantes -> gardées
```

puis le tableau des 8 lignes impossibles ou incohérentes : 3 lignes avec une température impossible (-5, 270 et 99 °C), et 5 lignes où la température minimale dépasse la maximale.

**DÉCISION 1** · Que faire des valeurs **impossibles** ?
- **Option A : supprimer les lignes**. On perd aussi les autres mesures de la ligne.
- **Option B : remplacer la valeur impossible par vide**. On ne garde que les mesures plausibles.
- **Option C : laisser et le noter**. Les moyennes seront fausses.

<!-- id: decision_impossibles -->
```python
decision_impossibles = ""        # DÉCISION : "A", "B" ou "C"
justification_impossibles = ""   # DÉCISION : une phrase, pourquoi ce choix ?
print("Option choisie :", decision_impossibles or "AUCUNE : tapez A, B ou C")
```

```python
# Option A : supprimer les lignes qui ont une valeur impossible
if decision_impossibles == "A":
    df = df[~impossibles]
    noter("8 Impossibles", f"{impossibles.sum()} lignes", "lignes supprimées", justification_impossibles)
else:
    print("Option A non choisie : rien n'est fait.")
```

```python
# Option B : remplacer chaque valeur impossible par vide
if decision_impossibles == "B":
    for colonne, (mini, maxi) in BORNES.items():
        df[colonne] = df[colonne].where(df[colonne].between(mini, maxi) | df[colonne].isna())
    noter("8 Impossibles", f"{impossibles.sum()} valeurs hors bornes", "remplacées par vide", justification_impossibles)
else:
    print("Option B non choisie : rien n'est fait.")
```

```python
# Option C : laisser, et le noter
if decision_impossibles == "C":
    noter("8 Impossibles", f"{impossibles.sum()} valeurs hors bornes", "laissées", justification_impossibles)
else:
    print("Option C non choisie : rien n'est fait.")
```

**DÉCISION 2** · Que faire des lignes **incohérentes** ?
- **Option A : supprimer les lignes**.
- **Option B : échanger les deux valeurs**, si c'est sans doute une inversion de saisie (par exemple le minimum écrit à la place du maximum).
- **Option C : vider les valeurs concernées**.

<!-- id: decision_incoherentes -->
```python
decision_incoherentes = ""        # DÉCISION : "A", "B" ou "C"
justification_incoherentes = ""   # DÉCISION : une phrase, pourquoi ce choix ?
print("Option choisie :", decision_incoherentes or "AUCUNE : tapez A, B ou C")
```

```python
# Option A : supprimer les lignes incohérentes
if decision_incoherentes == "A":
    df = df[~incoherentes.reindex(df.index, fill_value=False)]
    noter("8 Incohérentes", f"{incoherentes.sum()} lignes", "lignes supprimées", justification_incoherentes)
else:
    print("Option A non choisie : rien n'est fait.")
```

```python
# Option B : échanger les deux valeurs d'une règle « a <= b » qui n'est pas respectée
if decision_incoherentes == "B":
    for regle in REGLES:
        morceaux = re.match(r"^\s*(\w+)\s*<=?\s*(\w+)\s*$", regle)
        if morceaux and morceaux.group(2) in df.columns:
            a, b = morceaux.groups()
            viole = df[[a, b]].notna().all(axis=1) & ~df.eval(regle)
            df.loc[viole, [a, b]] = df.loc[viole, [b, a]].values
    noter("8 Incohérentes", f"{incoherentes.sum()} lignes", "valeurs échangées", justification_incoherentes)
else:
    print("Option B non choisie : rien n'est fait.")
```

```python
# Option C : vider les valeurs des colonnes de la règle
if decision_incoherentes == "C":
    for regle in REGLES:
        colonnes_regle = [c for c in df.columns if re.search(r"\b" + re.escape(c) + r"\b", regle)]
        viole = df[colonnes_regle].notna().all(axis=1) & ~df.eval(regle)
        df.loc[viole, colonnes_regle] = np.nan
    noter("8 Incohérentes", f"{incoherentes.sum()} lignes", "valeurs vidées", justification_incoherentes)
else:
    print("Option C non choisie : rien n'est fait.")
```

**Questions**
- Pourquoi 270 °C est-il impossible, alors que 210 mm de pluie en un jour est seulement étonnant ?
- Pour les lignes incohérentes, pourquoi l'option B est-elle raisonnable ici ?

### Geste 9 · Tracer

> **Je prédis → J'exécute → Je constate**

*C'est un jugement.* Le journal s'est rempli tout au long du carnet. La dernière cellule l'affiche, puis écrit **côte à côte**, dans votre Drive, le tableau propre (`releves_meteo_propre.csv`) et le journal (`journal_nettoyage.md`). Colab vous demandera d'autoriser l'accès à votre Drive.

```python
display(pd.DataFrame(journal))

if DOSSIER_SORTIE.startswith("/content/drive"):
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except ImportError:
        DOSSIER_SORTIE = "."   # en dehors de Colab : on écrit à côté du carnet
os.makedirs(DOSSIER_SORTIE, exist_ok=True)

chemin_csv = os.path.join(DOSSIER_SORTIE, NOM_SORTIE + ".csv")
chemin_journal = os.path.join(DOSSIER_SORTIE, "journal_nettoyage.md")
df.to_csv(chemin_csv, index=False)
with open(chemin_journal, "w", encoding="utf-8") as f:
    f.write("# Journal de nettoyage\n\n")
    f.write(f"Fichier d'origine : {URL}\n\n")
    f.write(f"Avant : {df_brut.shape[0]} lignes et {df_brut.shape[1]} colonnes. "
            f"Après : {df.shape[0]} lignes et {df.shape[1]} colonnes.\n\n")
    f.write("| geste | constat | décision | justification |\n|---|---|---|---|\n")
    for ligne in journal:
        f.write(f"| {ligne['geste']} | {ligne['constat']} | {ligne['décision']} | {ligne['justification']} |\n")

print("Tableau propre :", df.shape, "->", chemin_csv)
print("Journal        :", len(journal), "lignes ->", chemin_journal)
```

**Résultat attendu**

Le journal (11 lignes, du geste 1 au geste 8), puis, dans Colab :

```text
Mounted at /content/drive
Tableau propre : (1646, 6) -> /content/drive/MyDrive/TP_nettoyage/releves_meteo_propre.csv
Journal        : 11 lignes -> /content/drive/MyDrive/TP_nettoyage/journal_nettoyage.md
```

Les deux fichiers apparaissent côte à côte dans le dossier `TP_nettoyage` de votre Drive. Le nombre de lignes du journal dépend de vos décisions.

**Questions**
- Relisez le journal : quelqu'un qui n'était pas là pourrait-il refaire exactement le même nettoyage ? Que manquerait-il ?
- Combien de lignes avez-vous perdues entre le fichier brut et le fichier propre ? Pourquoi ?

---

## Ce qu'il faut rendre

Un notebook Colab par binôme, avec les deux noms en haut. Il doit contenir :
- toutes les cellules, **exécutées**, avec leurs résultats visibles ;
- vos décisions et leurs justifications dans les cellules DÉCISION ;
- sous chaque geste, votre réponse écrite dans la cellule « Votre réponse » ;
- le fichier `journal_nettoyage.md` produit à la fin.

Le tableau propre produit par le corrigé est publié dans [`data/clean/releves_meteo_propre.csv`](../../data/clean/releves_meteo_propre.csv). C'est lui que lisent le TP 2 et le TP 3, pour que tout le monde parte du même tableau.

## Testé sur d'autres fichiers

Pour vérifier que le carnet est réutilisable, il a été exécuté sur deux autres fichiers, **en ne changeant que la cellule RÉGLAGES** :
- **`rayonnement_pf.csv`** (TP de régression : 3 000 journées Météo-France, avec dates et stations), avec `COLONNES_A_RETIRER = ["mois", "longitude"]`, des bornes pour `humidite`, `heures_soleil` et `rayonnement`, et la règle `heures_soleil >= 0` : toutes les cellules tournent. Le tableau étant déjà propre, chaque dépistage répond « 0 ».
- **le jeu « iris » de scikit-learn** (150 fleurs, sans date) : `COLONNE_DATE = None`, `COLONNE_CATEGORIE = "espece"`. Toutes les cellules tournent, et le geste 6 trouve un vrai doublon dans ce jeu célèbre.

Ce qui a dû être adapté : uniquement les réglages. Pensez à vider `BORNES` et `REGLES` (`{}` et `[]`) si vous n'avez pas encore de bornes ; et si votre tableau est large, adaptez la cellule `melt` du geste 4.

## Glossaire

| Mot | En une phrase |
|---|---|
| **Copie de travail** | `df`, la copie qu'on nettoie ; `df_brut`, l'original, n'est jamais modifié. |
| **Typer** | Donner à chaque colonne le bon type : date, nombre ou texte. |
| **Trou** | Une case vide. Vide ne veut pas dire zéro. |
| **Code de trou** | Un nombre comme `-999` qui veut dire « pas de mesure ». |
| **Doublon** | Une ligne recopiée à l'identique. |
| **Harmoniser** | Une chose, une écriture. |
| **Impossible / étonnant / incohérent** | Hors des bornes possibles / rare mais plausible / qui contredit une règle entre colonnes. |
| **Journal de nettoyage** | La liste des constats, décisions et justifications, pour que le nettoyage soit refaisable. |
