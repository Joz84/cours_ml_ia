# TP · La plage est-elle baignable ?

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Joz84/cours_ml_ia/blob/main/tp/classification/TP_classification_ELEVE.ipynb)

UE 5.7 « Data Science et IA » · Licence STES · Université de la Polynésie française

Dans le cours, vous avez vu la **classification** : le modèle ne répond plus « combien ? » mais « **laquelle ?** ». Ici, la question est simple et elle compte pour la santé : *aujourd'hui, l'eau de cette plage est-elle conforme pour la baignade, oui ou non ?* Vous allez entraîner une **régression logistique** à y répondre à partir de la pluie, de la température et du débit de la rivière.

- **Durée :** 2 × 45 minutes. Le TP 1 prépare les données, le TP 2 entraîne, évalue et utilise le modèle.
- **En binôme**, dans **Google Colab**. Vous n'avez rien à installer.
- **Méthode :** copiez chaque cellule de code dans une nouvelle cellule Colab, dans l'ordre, puis exécutez-la avec Maj + Entrée.
- **Quand une cellule est à compléter**, un encadré « Ce que vous dites à l'assistant » vous donne la phrase à dire à ChatGPT ou à Claude.

## Un code réutilisable pour toute classification

Ce TP est écrit pour servir **à n'importe quel problème de classification**, pas seulement à la baignade. Tout ce qui est propre au projet est rassemblé à deux endroits, signalés par `À CHANGER` :

| Où ? | Ce qui change d'un projet à l'autre |
|---|---|
| **La cellule « Réglages »** (TP 1, étape 0) | le fichier, la colonne cible, les colonnes à écarter, la classe à ne pas rater |
| **Le cas inventé** (TP 2, question Q6) | les valeurs d'un nouveau cas, avec les colonnes de votre fichier |

**Toutes les autres cellules se copient telles quelles**, quel que soit le projet : elles n'utilisent que les réglages.

## Le rituel, à chaque étape

> **Je prédis** : je réponds aux questions *avant* d'exécuter la cellule.
> **J'exécute** : je lance la cellule.
> **Je constate** : j'écris deux phrases sur ce que j'observe, et je corrige ma prédiction si besoin.

Se tromper dans sa prédiction n'est pas grave : c'est même comme ça qu'on apprend le plus.

## Les données

Le fichier `baignade_taharuu.csv` suit **3 plages de Tahiti** (`taharuu`, `vaiava` et `pk18`) chaque jour pendant 2 ans, de 2023 à 2024 : 2 193 lignes. **Une ligne = un jour sur une plage.**

| Colonne | Contenu |
|---|---|
| `date` | Jour de l'observation |
| `plage` | `taharuu`, `vaiava` ou `pk18` (texte) |
| `pluie_24h_mm` | Pluie tombée dans la journée, en mm |
| `pluie_72h_mm` | Pluie tombée sur les 3 derniers jours, en mm |
| `temperature_C` | Température moyenne de l'air, en °C |
| `debit_m3s` | Débit de la rivière qui débouche près de la plage, en m³/s |
| `saison` | `ete_austral` (novembre à avril) ou `hiver_austral` (mai à octobre) |
| `conforme` | `oui` si l'eau est conforme pour la baignade, `non` sinon |

**Attention : ces données sont en partie simulées.** La pluie et la température sont **réelles** : ce sont les mesures Météo-France des pluviomètres les plus proches (Papara, Faa'a, Paea). En revanche, le **débit** et la **conformité** sont **simulés** à partir de règles réalistes : l'eau est plus souvent non conforme après de fortes pluies, quand la rivière est en crue, et à Taharu'u, à l'embouchure d'une grande rivière. Ce ne sont pas de vrais résultats d'analyse. La règle complète est décrite dans [`scripts/prepare_baignade.py`](../../scripts/prepare_baignade.py).

*Source de la pluie et de la température : Météo-France, « Données climatologiques de base - quotidiennes », département 987, sur data.gouv.fr, Licence Ouverte 2.0, téléchargé le 18/11/2026.*

---

## TP 1 · Le preprocessing de classification (45 min)

Objectif : transformer le tableau en un tableau **que le modèle sait lire**, en suivant les 5 étapes du cours : **séparer**, **trier**, **normaliser**, **encoder** les features texte et **encoder la cible**.

### Étape 0 · Les réglages du projet

> **Je prédis → J'exécute → Je constate**

C'est **la seule cellule propre à ce projet**. À vous de la compléter, en vous aidant du tableau « Les données » ci-dessus. Il faut trois choses :
- la **cible**, c'est-à-dire la colonne à prédire ;
- les **colonnes à écarter**, celles qui ne servent pas à prédire ;
- la **classe à détecter**, celle qu'on ne veut surtout pas rater.

> **Ce que vous dites à l'assistant**
>
> « Je fais une classification dans Google Colab. Mon fichier a les colonnes `date`, `plage`, `pluie_24h_mm`, `pluie_72h_mm`, `temperature_C`, `debit_m3s`, `saison` et `conforme` (`oui` / `non`). Je veux savoir si l'eau de baignade est conforme. Complète ces trois lignes Python : `cible = ...` (la colonne à prédire), `colonnes_a_exclure = [...]` (les colonnes qui ne servent pas à prédire) et `classe_a_detecter = ...` (la valeur de la cible qu'il ne faut pas rater), et explique tes choix. »

Collez les trois lignes à la place des `___`.

**Questions**
- Quelle colonne est la cible ? Pourquoi la date ne sert-elle à rien pour prédire un jour futur ?
- Quelle classe ne faut-il surtout pas rater : `oui` ou `non` ? Pourquoi ?

```python
# =============== RÉGLAGES DU PROJET : les seules lignes à changer ===============
URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/clean/baignade_taharuu.csv"  # À CHANGER : le fichier
cible = ___                # À CHANGER : la colonne à prédire, entre guillemets
colonnes_a_exclure = ___   # À CHANGER : la liste des colonnes qui ne servent pas à prédire
classe_a_detecter = ___    # À CHANGER : la classe qu'on ne veut pas rater, entre guillemets
# ================================================================================
```

**Ce que vous devez obtenir**

Rien ne s'affiche : la cellule enregistre seulement vos réglages. Si une erreur `NameError` apparaît, c'est qu'il reste un `___`.

### Étape 1 · Charger les données

> **Je prédis → J'exécute → Je constate**

Cette cellule importe les outils et charge le fichier. Elle affiche la taille du tableau et ses 5 premières lignes.

**Questions**
- Combien de lignes et de colonnes le tableau contient-il ?

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

df = pd.read_csv(URL)

print("Nombre de lignes et de colonnes :", df.shape)
df.head()
```

**Ce que vous devez obtenir**

`Nombre de lignes et de colonnes : (2193, 8)`, puis un tableau qui commence ainsi :

```text
         date    plage  pluie_24h_mm  pluie_72h_mm  temperature_C  debit_m3s       saison conforme
0  2023-01-01     pk18          13.9          33.3           27.8       4.02  ete_austral      oui
1  2023-01-01  taharuu           3.2           9.6           25.8       5.67  ete_austral      oui
2  2023-01-01   vaiava          14.3          14.3           27.8       1.85  ete_austral      oui
```

### Étape 2 · Les types de colonnes

> **Je prédis → J'exécute → Je constate**

`df.info()` donne, pour chaque colonne, le nombre de cases remplies et le type : nombre (`float64`, `int64`) ou texte (`object`, ou `str` selon la version de Colab).

**Questions**
- Y a-t-il des cases vides ?
- Quelles colonnes sont du texte ?

```python
df.info()
```

**Ce que vous devez obtenir**

```text
RangeIndex: 2193 entries, 0 to 2192
Data columns (total 8 columns):
 #   Column         Non-Null Count  Dtype
---  ------         --------------  -----
 0   date           2193 non-null   object
 1   plage          2193 non-null   object
 2   pluie_24h_mm   2193 non-null   float64
 3   pluie_72h_mm   2193 non-null   float64
 4   temperature_C  2193 non-null   float64
 5   debit_m3s      2193 non-null   float64
 6   saison         2193 non-null   object
 7   conforme       2193 non-null   object
```

Toutes les colonnes ont 2 193 cases remplies : il n'y a aucune case vide.

### Étape 3 · Combien de cas dans chaque classe ?

> **Je prédis → J'exécute → Je constate**

On compte les lignes de chaque classe de la cible, en nombre puis en pourcentage.

**Questions**
- Les deux classes sont-elles aussi fréquentes l'une que l'autre ?
- Un modèle « paresseux » qui répondrait toujours la classe la plus fréquente aurait-il souvent raison ?

```python
print(df[cible].value_counts())
print()
print((df[cible].value_counts(normalize=True) * 100).round(1))
```

**Ce que vous devez obtenir**

```text
conforme
oui    1835
non     358
Name: count, dtype: int64

conforme
oui    83.7
non    16.3
Name: proportion, dtype: float64
```

Les classes sont très **déséquilibrées** : seulement 16,3 % de jours non conformes.

### Étape 4 · Les features (X) et la cible (y)

> **Je prédis → J'exécute → Je constate**

**X** contient les features : toutes les colonnes, sauf la cible et les colonnes écartées. **y** contient la cible.

**Questions**
- Combien de colonnes aura X ?
- Pourquoi la cible ne doit-elle surtout pas être dans X ?

```python
X = df.drop(columns=colonnes_a_exclure + [cible])
y = df[cible]

print("X :", X.shape, "  y :", y.shape)
X.dtypes
```

**Ce que vous devez obtenir**

`X : (2193, 6)   y : (2193,)`, puis le type de chaque feature :

```text
plage             object
pluie_24h_mm     float64
pluie_72h_mm     float64
temperature_C    float64
debit_m3s        float64
saison            object
```

### Étape 5 · Séparer entraînement et test

> **Je prédis → J'exécute → Je constate**

On met de côté 20 % des lignes pour le **test**. `stratify=y` garde **la même proportion de chaque classe** dans l'entraînement et dans le test. C'est important quand une classe est rare : sinon, le hasard pourrait en mettre presque aucune dans le test.

**Questions**
- Combien de lignes dans chaque jeu ? Quelle part de la classe à détecter dans chacun ?

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Entraînement :", X_train.shape[0], "lignes")
print("Test         :", X_test.shape[0], "lignes")
print("Part de", classe_a_detecter, "dans l'entraînement :", round((y_train == classe_a_detecter).mean() * 100, 1), "%")
print("Part de", classe_a_detecter, "dans le test        :", round((y_test == classe_a_detecter).mean() * 100, 1), "%")
```

**Ce que vous devez obtenir**

```text
Entraînement : 1754 lignes
Test         : 439 lignes
Part de non dans l'entraînement : 16.3 %
Part de non dans le test        : 16.4 %
```

### Étape 6 · Trier les colonnes numériques et non numériques

> **Je prédis → J'exécute → Je constate**

Le preprocessing n'est pas le même pour les nombres et pour le texte : on range les colonnes de X en deux groupes.

**Questions**
- Quelles colonnes iront dans chaque groupe ?

```python
colonnes_num = X_train.select_dtypes(include="number").columns
colonnes_texte = X_train.select_dtypes(exclude="number").columns

print("Colonnes numériques     :", list(colonnes_num))
print("Colonnes non numériques :", list(colonnes_texte))
```

**Ce que vous devez obtenir**

```text
Colonnes numériques     : ['pluie_24h_mm', 'pluie_72h_mm', 'temperature_C', 'debit_m3s']
Colonnes non numériques : ['plage', 'saison']
```

### Étape 7 · Normaliser les colonnes numériques

> **Je prédis → J'exécute → Je constate**

Le `StandardScaler` met chaque colonne numérique à la même échelle (moyenne 0, écart-type 1). Il apprend **sur l'entraînement seulement** (`fit_transform`), puis applique la même transformation au test (`transform`).

**Questions**
- Avant normalisation, quelle colonne a les plus grands nombres ? Laquelle varie le moins ?

```python
scaler = StandardScaler()
X_train_num = pd.DataFrame(scaler.fit_transform(X_train[colonnes_num]),
                           columns=colonnes_num, index=X_train.index)
X_test_num = pd.DataFrame(scaler.transform(X_test[colonnes_num]),
                          columns=colonnes_num, index=X_test.index)

print("AVANT la normalisation")
print(X_train[colonnes_num].describe().round(1))
print("\nAPRÈS la normalisation")
print(X_train_num.describe().round(1))
```

**Ce que vous devez obtenir**

```text
AVANT la normalisation
       pluie_24h_mm  pluie_72h_mm  temperature_C  debit_m3s
count        1754.0        1754.0         1754.0     1754.0
mean            4.6          13.7           26.3        3.3
std            12.2          24.7            1.6        4.4
min             0.0           0.0           20.8        0.5
...
max           127.9         242.9           29.7       42.3

APRÈS la normalisation
       pluie_24h_mm  pluie_72h_mm  temperature_C  debit_m3s
count        1754.0        1754.0         1754.0     1754.0
mean           -0.0           0.0            0.0       -0.0
std             1.0           1.0            1.0        1.0
...
```

Après normalisation, toutes les colonnes ont une moyenne de 0 et un écart-type de 1. `-0.0` est simplement un 0 arrondi.

### Étape 8 · Encoder les features texte

> **Je prédis → J'exécute → Je constate**

Le `OneHotEncoder` transforme chaque catégorie des colonnes texte en une colonne de 0 et de 1.

**Questions**
- Combien de colonnes vont apparaître ?

```python
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_texte = pd.DataFrame(encoder.fit_transform(X_train[colonnes_texte]),
                             columns=encoder.get_feature_names_out(), index=X_train.index)
X_test_texte = pd.DataFrame(encoder.transform(X_test[colonnes_texte]),
                            columns=encoder.get_feature_names_out(), index=X_test.index)

print("Nouvelles colonnes :", list(X_train_texte.columns))
X_train_texte.head()
```

**Ce que vous devez obtenir**

```text
Nouvelles colonnes : ['plage_pk18', 'plage_taharuu', 'plage_vaiava', 'saison_ete_austral', 'saison_hiver_austral']
```

puis un tableau de 0 et de 1 qui commence ainsi :

```text
      plage_pk18  plage_taharuu  plage_vaiava  saison_ete_austral  saison_hiver_austral
1597         0.0            1.0           0.0                 0.0                   1.0
897          1.0            0.0           0.0                 0.0                   1.0
31           0.0            1.0           0.0                 1.0                   0.0
```

### Étape 9 · Encoder la cible

> **Je prédis → J'exécute → Je constate**

La cible aussi est du texte. Le `LabelEncoder` donne un numéro à chaque classe, par ordre alphabétique. La cellule retient aussi le numéro de la classe à détecter dans `code_positif` : le TP 2 en aura besoin.

**Questions**
- Quel numéro recevra chaque classe ? Quel sera le numéro de la classe à détecter ?

```python
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)
code_positif = label_encoder.transform([classe_a_detecter])[0]

for numero, classe in enumerate(label_encoder.classes_):
    print(classe, "->", numero)
print("Classe à détecter :", classe_a_detecter, "-> numéro", code_positif)
```

**Ce que vous devez obtenir**

```text
non -> 0
oui -> 1
Classe à détecter : non -> numéro 0
```

### Étape 10 · Tout réunir

> **Je prédis → J'exécute → Je constate**

On recolle les colonnes normalisées et les colonnes encodées : c'est le tableau final que le modèle va lire.

**Questions**
- Combien de colonnes aura `X_train_prep` ?

```python
X_train_prep = pd.concat([X_train_num, X_train_texte], axis=1)
X_test_prep = pd.concat([X_test_num, X_test_texte], axis=1)

print("X_train_prep :", X_train_prep.shape, "  X_test_prep :", X_test_prep.shape)
X_train_prep.head().round(2)
```

**Ce que vous devez obtenir**

`X_train_prep : (1754, 9)   X_test_prep : (439, 9)`, puis un tableau qui ne contient que des nombres :

```text
      pluie_24h_mm  pluie_72h_mm  temperature_C  debit_m3s  plage_pk18  plage_taharuu  plage_vaiava  saison_ete_austral  saison_hiver_austral
1597         -0.23         -0.47          -0.81       0.24         0.0            1.0           0.0                 0.0                   1.0
897          -0.06          1.65          -1.32       0.40         1.0            0.0           0.0                 0.0                   1.0
31            0.68          1.14          -1.07       2.02         0.0            1.0           0.0                 1.0                   0.0
```

**Fin du TP 1.** Vos données sont prêtes. Vous n'avez rien à sauvegarder : le TP 2 commence par une cellule qui refait tout le TP 1 d'un coup.

---

## TP 2 · Entraîner, évaluer, prédire (45 min)

Objectif : suivre les quatre verbes du cours, **importer et créer → `fit` → `score` → `predict`**, puis répondre aux questions Q1 à Q7 par écrit, dans le notebook, sous chaque question.

### Étape 0 · Tout le TP 1 en une fois

> **J'exécute**

Ouvrez un nouveau notebook Colab, ou continuez dans celui du TP 1, et exécutez cette cellule. Elle refait tout le TP 1, réglages compris, et importe les outils d'évaluation. Pour un autre projet, seul le bloc « RÉGLAGES » change.

```python
# =============== RÉGLAGES DU PROJET : les seules lignes à changer ===============
URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/clean/baignade_taharuu.csv"  # À CHANGER : le fichier
cible = "conforme"            # À CHANGER : la colonne à prédire
colonnes_a_exclure = ["date"]  # À CHANGER : les colonnes qui ne servent pas à prédire
classe_a_detecter = "non"     # À CHANGER : la classe qu'on ne veut pas rater
# ================================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix, recall_score, precision_score

df = pd.read_csv(URL)
X = df.drop(columns=colonnes_a_exclure + [cible])
y = df[cible]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

colonnes_num = X_train.select_dtypes(include="number").columns
colonnes_texte = X_train.select_dtypes(exclude="number").columns

scaler = StandardScaler()
X_train_num = pd.DataFrame(scaler.fit_transform(X_train[colonnes_num]),
                           columns=colonnes_num, index=X_train.index)
X_test_num = pd.DataFrame(scaler.transform(X_test[colonnes_num]),
                          columns=colonnes_num, index=X_test.index)

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_texte = pd.DataFrame(encoder.fit_transform(X_train[colonnes_texte]),
                             columns=encoder.get_feature_names_out(), index=X_train.index)
X_test_texte = pd.DataFrame(encoder.transform(X_test[colonnes_texte]),
                            columns=encoder.get_feature_names_out(), index=X_test.index)

label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)
code_positif = label_encoder.transform([classe_a_detecter])[0]

X_train_prep = pd.concat([X_train_num, X_train_texte], axis=1)
X_test_prep = pd.concat([X_test_num, X_test_texte], axis=1)
print("TP 1 refait :", X_train_prep.shape, X_test_prep.shape)
```

**Ce que vous devez obtenir**

`TP 1 refait : (1754, 9) (439, 9)`

### Étape 1 · Importer et créer le modèle

> **Je prédis → J'exécute → Je constate**

On importe la régression logistique, puis on crée un modèle vide. Malgré son nom, c'est un modèle de **classification** : il calcule une **probabilité** pour chaque classe.

**Questions**
- À ce stade, le modèle sait-il déjà reconnaître une eau non conforme ?

```python
from sklearn.linear_model import LogisticRegression

modele = LogisticRegression(random_state=42)
modele
```

**Ce que vous devez obtenir**

Une petite boîte qui affiche `LogisticRegression(random_state=42)`.

### Étape 2 · Entraîner : `fit`

> **Je prédis → J'exécute → Je constate**

Le modèle apprend, sur les lignes d'entraînement, le lien entre les features et la cible encodée.

**Questions**
- Pourquoi donne-t-on `y_train_enc` et pas `y_train` ?

```python
modele.fit(X_train_prep, y_train_enc)
```

**Ce que vous devez obtenir**

La même petite boîte `LogisticRegression(random_state=42)` : le modèle est entraîné.

### Étape 3 · Évaluer : `score`

> **Je prédis → J'exécute → Je constate**

Pour un classifieur, `score` donne l'**exactitude** sur le jeu de test.

**Questions**
- Le score sera-t-il plutôt de 50 %, de 70 % ou de 90 % ?

```python
exactitude = modele.score(X_test_prep, y_test_enc)
print("Exactitude sur le test :", round(exactitude * 100, 1), "%")
```

**Ce que vous devez obtenir**

```text
Exactitude sur le test : 91.1 %
```

### Étape 4 · Prédire : `predict`

> **Je prédis → J'exécute → Je constate**

`predict` donne la classe prédite pour chaque ligne du test : un numéro, qu'on retraduit en texte avec `inverse_transform`. On compare avec la vraie classe sur les 10 premières lignes.

**Questions**
- Sur ces 10 lignes, le modèle se trompera-t-il ?

```python
y_pred = modele.predict(X_test_prep)

pd.DataFrame({"vraiment": y_test[:10].values,
              "predit": label_encoder.inverse_transform(y_pred[:10])})
```

**Ce que vous devez obtenir**

```text
  vraiment predit
0      oui    oui
1      non    non
2      oui    oui
3      non    non
4      non    non
5      non    oui
6      oui    oui
7      oui    oui
8      oui    oui
9      oui    oui
```

Sur ces 10 jours, le modèle se trompe une fois (ligne 5) : il annonce `oui` pour un jour vraiment non conforme.

### Q1 · Que vaut le score ?

Relisez le résultat de l'étape 3.

**Questions**
- Quel est le `score` du modèle ? Que mesure-t-il exactement pour un classifieur ?

### Q2 · Le modèle paresseux

> **Je prédis → J'exécute → Je constate**

Un modèle « paresseux » répond **toujours la classe la plus fréquente**, sans rien regarder. C'est notre **baseline**. On trouve cette classe avec `value_counts`, puis on calcule son exactitude sur le test.

**Questions**
- Quelle est l'exactitude de ce modèle paresseux ? Comparez-la au score de votre modèle.
- Un score de 90 % est-il impressionnant ici ?

```python
classe_frequente = y_train.value_counts().index[0]
exactitude_paresseux = (y_test == classe_frequente).mean()

print("Le modèle paresseux répond toujours :", classe_frequente)
print("Exactitude du modèle paresseux :", round(exactitude_paresseux * 100, 1), "%")
print("Exactitude de notre modèle     :", round(exactitude * 100, 1), "%")
```

**Ce que vous devez obtenir**

```text
Le modèle paresseux répond toujours : oui
Exactitude du modèle paresseux : 83.6 %
Exactitude de notre modèle     : 91.1 %
```

### Q3 · La matrice de confusion

> **Je prédis → J'exécute → Je constate**

La **matrice de confusion** croise la vérité (lignes) et la prédiction (colonnes). La classe « positive », celle qu'on cherche à repérer, est la classe à détecter : ici `non`, une eau non conforme. On la place en premier :

| | prédit non (non conforme) | prédit oui (conforme) |
|---|---|---|
| **vraiment non** | VP (vrais positifs) | FN (faux négatifs : ratés) |
| **vraiment oui** | FP (faux positifs : fausses alertes) | VN (vrais négatifs) |

**Questions**
- Combien de jours non conformes le modèle a-t-il **ratés** ?
- Combien de fausses alertes a-t-il données ?

```python
ordre = [code_positif, 1 - code_positif]       # la classe à détecter d'abord, puis l'autre
noms = label_encoder.inverse_transform(ordre)  # leurs noms, dans le même ordre

matrice = confusion_matrix(y_test_enc, y_pred, labels=ordre)
pd.DataFrame(matrice,
             index=["vraiment " + noms[0], "vraiment " + noms[1]],
             columns=["prédit " + noms[0], "prédit " + noms[1]])
```

**Ce que vous devez obtenir**

```text
              prédit non  prédit oui
vraiment non          42          30
vraiment oui           9         358
```

### Q4 · Rappel et précision de la classe à détecter

> **Je prédis → J'exécute → Je constate**

- **Rappel** : parmi les lignes **vraiment** de la classe à détecter (la ligne du haut), quelle part le modèle a-t-il repérée ?
- **Précision** : parmi les lignes **prédites** de la classe à détecter (la colonne de gauche), quelle part l'était vraiment ?

`pos_label=code_positif` dit à scikit-learn quelle classe compte.

**Questions**
- Calculez-les d'abord à la main avec la matrice de Q3, puis vérifiez.
- Ici, rater une plage polluée coûte cher (des baigneurs malades). Faut-il privilégier le rappel ou la précision ? Pourquoi ?

```python
rappel = recall_score(y_test_enc, y_pred, pos_label=code_positif)
precision = precision_score(y_test_enc, y_pred, pos_label=code_positif)

print("Rappel    de", classe_a_detecter, ":", round(rappel * 100, 1), "%")
print("Précision de", classe_a_detecter, ":", round(precision * 100, 1), "%")
```

**Ce que vous devez obtenir**

```text
Rappel    de non : 58.3 %
Précision de non : 82.4 %
```

### Q5 · Déplacer le seuil

> **Je prédis → J'exécute → Je constate**

`predict` déclare la classe à détecter quand sa **probabilité** dépasse **0,5**. Mais ce **seuil** est un choix. Avec `predict_proba`, on récupère cette probabilité pour chaque ligne du test, et on choisit nous-mêmes le seuil.

Exécutez la cellule avec `seuil = 0.3`, puis changez la première ligne en `seuil = 0.5`, puis en `seuil = 0.7`, et relancez à chaque fois.

**Questions**
- Si on abaisse le seuil à 0,3, le rappel va-t-il monter ou baisser ? Et la précision ?
- Même question avec un seuil de 0,7.
- Conclusion en deux phrases : quel seuil choisiriez-vous pour protéger les baigneurs ?

```python
seuil = 0.3   # À ESSAYER : 0.3, puis 0.5, puis 0.7

proba = modele.predict_proba(X_test_prep)[:, code_positif]  # probabilité de la classe à détecter
alerte = proba >= seuil                       # vrai si le modèle déclare la classe à détecter
vraiment = y_test == classe_a_detecter        # vrai si la ligne est vraiment de cette classe

print("Seuil", seuil, ": rappel =", round(recall_score(vraiment, alerte) * 100, 1), "%",
      "  précision =", round(precision_score(vraiment, alerte) * 100, 1), "%")
```

**Ce que vous devez obtenir**

Une ligne par essai :

```text
Seuil 0.3 : rappel = 69.4 %   précision = 72.5 %
Seuil 0.5 : rappel = 58.3 %   précision = 82.4 %
Seuil 0.7 : rappel = 44.4 %   précision = 91.4 %
```

### Q6 · Prédire pour un cas que vous inventez

> **Je prédis → J'exécute → Je constate**

Inventez un jour : changez les valeurs ci-dessous. C'est la **seule autre cellule propre au projet** : elle doit contenir exactement les colonnes de X. Le cas passe par **le même preprocessing** que l'entraînement (`transform`, jamais `fit`), puis par `predict` et `predict_proba`.

**Questions**
- Pour le jour que vous avez inventé, que va répondre le modèle ? Avec quelle probabilité ?
- Le modèle a-t-il raison, selon vous ? Essayez un jour sec, puis un jour de grosse pluie.

```python
# À CHANGER : un cas inventé, avec exactement les colonnes de X
X_nouveau = pd.DataFrame({
    "plage": ["taharuu"],
    "pluie_24h_mm": [35.0],
    "pluie_72h_mm": [80.0],
    "temperature_C": [26.5],
    "debit_m3s": [8.0],
    "saison": ["ete_austral"],
})

X_nouveau_prep = pd.concat([
    pd.DataFrame(scaler.transform(X_nouveau[colonnes_num]), columns=colonnes_num),
    pd.DataFrame(encoder.transform(X_nouveau[colonnes_texte]), columns=encoder.get_feature_names_out()),
], axis=1)

print("Prédiction :", label_encoder.inverse_transform(modele.predict(X_nouveau_prep))[0])
print("Probabilité de", classe_a_detecter, ":", round(modele.predict_proba(X_nouveau_prep)[0, code_positif] * 100), "%")
```

**Ce que vous devez obtenir**

Avec le jour proposé (forte pluie à Taharu'u) :

```text
Prédiction : non
Probabilité de non : 99 %
```

Avec vos propres valeurs, le résultat change, bien sûr.

### Q7 (bonus) · Quelles features pèsent le plus ?

> **Je prédis → J'exécute → Je constate**

`coef_` donne un coefficient par feature. La cellule les remet dans le bon sens : un effet **positif** pousse vers la classe à détecter, un effet négatif pousse vers l'autre classe. Les features numériques ont été normalisées : on peut comparer leurs effets entre eux, mais avec prudence.

**Questions**
- Quelle feature pousse le plus vers la classe à détecter ? Est-ce logique ?

```python
sens = 1 if code_positif == 1 else -1   # scikit-learn calcule pour la classe n° 1 : on remet dans le bon sens
effets = pd.DataFrame({"feature": X_train_prep.columns,
                       "effet": (sens * modele.coef_[0]).round(2)})
effets.sort_values("effet", ascending=False)
```

**Ce que vous devez obtenir**

```text
                feature  effet
1          pluie_72h_mm   1.59
5         plage_taharuu   0.88
0          pluie_24h_mm   0.54
3             debit_m3s   0.54
7    saison_ete_austral   0.07
8  saison_hiver_austral  -0.07
2         temperature_C  -0.20
6          plage_vaiava  -0.38
4            plage_pk18  -0.49
```

---

## Réutiliser ce code pour n'importe quelle classification

Le chemin est **toujours le même** :

> réglages → charger → X et y → séparer (stratifié) → trier → normaliser → encoder les features texte → encoder la cible → réunir → `fit` → `score` → `predict` / `predict_proba`

Pour un autre problème de classification, par exemple « ce récif est-il en bonne santé ? » ou « cette parcelle est-elle touchée par un ravageur ? », **seules 5 choses changent**, toutes marquées `À CHANGER` :

| | Ce qui change | Dans ce TP | Pour votre problème |
|---|---|---|---|
| **1** | **Le fichier** (`URL`) | `baignade_taharuu.csv` | l'adresse de votre fichier CSV |
| **2** | **La cible** (`cible`) | `"conforme"` | la colonne à prédire, une catégorie |
| **3** | **Les colonnes à écarter** (`colonnes_a_exclure`) | `["date"]` | identifiants, dates, ou toute colonne qui donnerait la réponse |
| **4** | **La classe à détecter** (`classe_a_detecter`) | `"non"` | la classe qu'on ne veut pas rater |
| **5** | **Le cas inventé** (`X_nouveau`) | un jour de pluie à Taharu'u | un cas avec exactement les colonnes de X |

**Tout le reste ne change pas** : toutes les autres cellules du TP 1 et du TP 2 se copient telles quelles. Deux conditions : le fichier doit être propre (sans cases vides), et la cible doit avoir **deux classes**, comme ici.

Voici la recette complète en une seule cellule. Les blocs `À CHANGER` sont en haut, et rien ne change sous la ligne `RIEN NE CHANGE`. Telle quelle, elle refait ce TP.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score

# =============== RÉGLAGES DU PROJET : les seules lignes à changer ===============
URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/clean/baignade_taharuu.csv"  # À CHANGER
cible = "conforme"            # À CHANGER : la colonne à prédire
colonnes_a_exclure = ["date"]  # À CHANGER : les colonnes qui ne servent pas à prédire
classe_a_detecter = "non"     # À CHANGER : la classe qu'on ne veut pas rater
X_nouveau = pd.DataFrame({    # À CHANGER : un cas inventé, avec exactement les colonnes de X
    "plage": ["taharuu"], "pluie_24h_mm": [35.0], "pluie_72h_mm": [80.0],
    "temperature_C": [26.5], "debit_m3s": [8.0], "saison": ["ete_austral"],
})

# ===================== À PARTIR D'ICI, RIEN NE CHANGE =====================
df = pd.read_csv(URL)
X = df.drop(columns=colonnes_a_exclure + [cible])
y = df[cible]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

colonnes_num = X_train.select_dtypes(include="number").columns
colonnes_texte = X_train.select_dtypes(exclude="number").columns
scaler = StandardScaler()
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
label_encoder = LabelEncoder()

X_train_prep = pd.concat([
    pd.DataFrame(scaler.fit_transform(X_train[colonnes_num]), columns=colonnes_num, index=X_train.index),
    pd.DataFrame(encoder.fit_transform(X_train[colonnes_texte]), columns=encoder.get_feature_names_out(), index=X_train.index),
], axis=1)
X_test_prep = pd.concat([
    pd.DataFrame(scaler.transform(X_test[colonnes_num]), columns=colonnes_num, index=X_test.index),
    pd.DataFrame(encoder.transform(X_test[colonnes_texte]), columns=encoder.get_feature_names_out(), index=X_test.index),
], axis=1)
X_nouveau_prep = pd.concat([
    pd.DataFrame(scaler.transform(X_nouveau[colonnes_num]), columns=colonnes_num),
    pd.DataFrame(encoder.transform(X_nouveau[colonnes_texte]), columns=encoder.get_feature_names_out()),
], axis=1)
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)
code_positif = label_encoder.transform([classe_a_detecter])[0]

modele = LogisticRegression(random_state=42)
modele.fit(X_train_prep, y_train_enc)
y_pred = modele.predict(X_test_prep)

classe_frequente = y_train.value_counts().index[0]
print("Exactitude du modèle          :", round(modele.score(X_test_prep, y_test_enc) * 100, 1), "%")
print("Exactitude du modèle paresseux :", round((y_test == classe_frequente).mean() * 100, 1), "%")
print("Rappel    de", classe_a_detecter, ":", round(recall_score(y_test_enc, y_pred, pos_label=code_positif) * 100, 1), "%")
print("Précision de", classe_a_detecter, ":", round(precision_score(y_test_enc, y_pred, pos_label=code_positif) * 100, 1), "%")
print("Prédiction pour le cas inventé :", label_encoder.inverse_transform(modele.predict(X_nouveau_prep))[0])
```

Telle quelle, la cellule affiche une exactitude de 91,1 %, 83,6 % pour le modèle paresseux, un rappel de `non` de 58,3 %, une précision de 82,4 %, et `non` pour le cas inventé.

## Ce qu'il faut rendre

Un notebook Colab par binôme, avec les deux noms en haut. Il doit contenir :
- toutes les cellules du TP 1 et du TP 2, **exécutées**, avec leurs résultats visibles ;
- sous chaque étape et chaque question Q1 à Q7, votre réponse écrite dans la cellule « Votre réponse ».

Partagez-le avec votre enseignant via « Partager » dans Colab, ou téléchargez-le avec Fichier > Télécharger > .ipynb.

## Glossaire

| Mot | En une phrase |
|---|---|
| **Classification** | Prédire une catégorie (« laquelle ? ») plutôt qu'un nombre. |
| **Régression logistique** | Un modèle de classification qui calcule la **probabilité** de chaque classe. |
| **Seuil** | La probabilité à partir de laquelle on déclare une classe : 0,5 par défaut, mais c'est un choix. |
| **Encoder la cible (LabelEncoder)** | Donner un numéro à chaque classe de la cible (ici `non` → 0, `oui` → 1). |
| **Matrice de confusion** | Le tableau qui croise la vérité et la prédiction : VP, FN, FP, VN. |
| **Exactitude** | La part de prédictions justes. Piège : avec une classe rare, répondre toujours la classe fréquente donne déjà une bonne exactitude. |
| **Rappel** | Parmi les vrais positifs à trouver, la part repérée : à privilégier quand **rater coûte cher**. |
| **Précision** | Parmi les alertes données, la part justifiée : à privilégier quand **une fausse alerte coûte cher**. |
