# TP · La plage est-elle baignable ?

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Joz84/cours_ml_ia/blob/main/tp/classification/TP_classification_ELEVE.ipynb)

UE 5.7 « Data Science et IA » · Licence STES · Université de la Polynésie française

Dans le cours, vous avez vu la **classification** : le modèle ne répond plus « combien ? » mais « **laquelle ?** ». Ici, la question est simple et elle compte pour la santé : *aujourd'hui, l'eau de cette plage est-elle conforme pour la baignade, oui ou non ?* Vous allez entraîner une **régression logistique** à y répondre à partir de la pluie, de la température et du débit de la rivière.

- **Durée :** 2 × 45 minutes. Le TP 1 prépare les données, le TP 2 entraîne, évalue et utilise le modèle.
- **En binôme**, dans **Google Colab**. Vous n'avez rien à installer.
- **Méthode :** copiez chaque cellule de code dans une nouvelle cellule Colab, dans l'ordre, puis exécutez-la avec Maj + Entrée.
- **Quand une cellule est à compléter**, un encadré « Ce que vous dites à l'assistant » vous donne la phrase à dire à ChatGPT ou à Claude.

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
| `conforme` | **La cible** : `oui` si l'eau est conforme pour la baignade, `non` sinon |

**Attention : ces données sont en partie simulées.** La pluie et la température sont **réelles** : ce sont les mesures Météo-France des pluviomètres les plus proches (Papara, Faa'a, Paea). En revanche, le **débit** et la **conformité** sont **simulés** à partir de règles réalistes : l'eau est plus souvent non conforme après de fortes pluies, quand la rivière est en crue, et à Taharu'u, à l'embouchure d'une grande rivière. Ce ne sont pas de vrais résultats d'analyse. La règle complète est décrite dans [`scripts/prepare_baignade.py`](../../scripts/prepare_baignade.py).

*Source de la pluie et de la température : Météo-France, « Données climatologiques de base - quotidiennes », département 987, sur data.gouv.fr, Licence Ouverte 2.0, téléchargé le 18/11/2026.*

---

## TP 1 · Le preprocessing de classification (45 min)

Objectif : transformer le tableau en un tableau **que le modèle sait lire**, en suivant les 5 étapes du cours : **séparer**, **trier**, **normaliser**, **encoder** les features texte et **encoder la cible**.

### Étape 0 · Charger les données

> **Je prédis → J'exécute → Je constate**

Cette cellule importe les outils et charge le fichier directement depuis GitHub. Elle affiche la taille du tableau et ses 5 premières lignes.

**Questions**
- Combien de lignes et de colonnes le tableau contient-il ?
- Quelle colonne est la cible ? Est-ce un nombre ou une catégorie ?

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/clean/baignade_taharuu.csv"
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

### Étape 1 · Les types de colonnes

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

Toutes les colonnes ont 2 193 cases remplies : il n'y a aucune case vide. Selon la version de Colab, les colonnes de texte s'affichent `object` ou `str`.

### Étape 2 · Combien de « oui », combien de « non » ?

> **Je prédis → J'exécute → Je constate**

On compte les jours conformes et non conformes, en nombre puis en pourcentage.

**Questions**
- Les deux classes sont-elles aussi fréquentes l'une que l'autre ?
- Un modèle « paresseux » qui répondrait toujours `oui` aurait-il souvent raison ?

```python
print(df["conforme"].value_counts())
print()
print((df["conforme"].value_counts(normalize=True) * 100).round(1))
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

Les deux classes sont très **déséquilibrées** : seulement 16,3 % de jours non conformes.

### Étape 3 · Choisir les features et la cible

> **Je prédis → J'exécute → Je constate**

À vous de choisir : quelles colonnes donne-t-on au modèle (**X**), et quelle colonne doit-il apprendre à prédire (**y**) ? Une colonne ne doit pas aller dans X : la date, qui ne se répétera jamais.

> **Ce que vous dites à l'assistant**
>
> « Dans Google Colab, j'ai un DataFrame pandas `df` avec les colonnes `date`, `plage`, `pluie_24h_mm`, `pluie_72h_mm`, `temperature_C`, `debit_m3s`, `saison` et `conforme`. Je veux prédire `conforme`. Écris-moi deux lignes de code : `X` avec toutes les colonnes sauf `date` et `conforme`, et `y` avec la colonne `conforme`. »

Collez ces deux lignes à la place des `___`.

**Questions**
- Pourquoi `conforme` ne doit-il surtout pas être dans X ?
- Pourquoi la date ne sert-elle à rien pour prédire un jour futur ?

```python
# À compléter : collez ici les deux lignes données par l'assistant
X = ___
y = ___

print("X :", X.shape, "  y :", y.shape)
for colonne in X.columns:
    print(f"  {colonne:15} ->", "nombre" if pd.api.types.is_numeric_dtype(X[colonne]) else "texte")
```

**Ce que vous devez obtenir**

```text
X : (2193, 6)   y : (2193,)
  plage           -> texte
  pluie_24h_mm    -> nombre
  pluie_72h_mm    -> nombre
  temperature_C   -> nombre
  debit_m3s       -> nombre
  saison          -> texte
```

Si vous obtenez une erreur `KeyError`, vérifiez l'orthographe des noms de colonnes.

### Étape 4 · Séparer entraînement et test

> **Je prédis → J'exécute → Je constate**

On met de côté 20 % des jours pour le **test**. `stratify=y` garde **la même proportion de `non`** dans l'entraînement et dans le test. C'est important quand une classe est rare : sinon, le hasard pourrait en mettre presque aucun dans le test.

**Questions**
- Combien de lignes dans chaque jeu ?
- Quelle part de `non` dans chacun ?

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Entraînement :", X_train.shape[0], "lignes, dont", round((y_train == "non").mean() * 100, 1), "% de non")
print("Test         :", X_test.shape[0], "lignes, dont", round((y_test == "non").mean() * 100, 1), "% de non")
```

**Ce que vous devez obtenir**

```text
Entraînement : 1754 lignes, dont 16.3 % de non
Test         : 439 lignes, dont 16.4 % de non
```

### Étape 5 · Trier les colonnes numériques et non numériques

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

### Étape 6 · Normaliser les colonnes numériques

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

### Étape 7 · Encoder les features texte

> **Je prédis → J'exécute → Je constate**

Le `OneHotEncoder` transforme chaque catégorie de `plage` et de `saison` en une colonne de 0 et de 1.

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

### Étape 8 · Encoder la cible

> **Je prédis → J'exécute → Je constate**

La cible aussi est du texte (`oui` / `non`). Le `LabelEncoder` donne un numéro à chaque classe, par ordre alphabétique. Retenez bien quel numéro reçoit `non` : vous en aurez besoin au TP 2.

**Questions**
- Quel numéro recevra `non`, et lequel recevra `oui` ?

```python
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

for numero, classe in enumerate(label_encoder.classes_):
    print(f"{classe} -> {numero}")
print("\nLes 5 premières cibles d'entraînement :", list(y_train[:5]), "->", y_train_enc[:5].tolist())
```

**Ce que vous devez obtenir**

```text
non -> 0
oui -> 1

Les 5 premières cibles d'entraînement : ['oui', 'non', 'non', 'oui', 'non'] -> [1, 0, 0, 1, 0]
```

**`non` reçoit le numéro 0 et `oui` le numéro 1**, par ordre alphabétique.

### Étape 9 · Tout réunir

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

Ouvrez un nouveau notebook Colab, ou continuez dans celui du TP 1, et exécutez cette cellule. Elle refait tout le TP 1, réponse de l'étape 3 comprise, et importe les outils d'évaluation.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix, recall_score, precision_score

URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/clean/baignade_taharuu.csv"
df = pd.read_csv(URL)

X = df.drop(columns=["date", "conforme"])
y = df["conforme"]
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

Le modèle apprend, sur les jours d'entraînement, le lien entre les features et la cible encodée.

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

`predict` donne la classe prédite pour chaque jour du test : un numéro, qu'on retraduit en `oui` / `non` avec `inverse_transform`.

**Questions**
- Sur les 10 premiers jours du test, le modèle se trompera-t-il ?

```python
y_pred = modele.predict(X_test_prep)

comparaison = pd.DataFrame({"vraiment": y_test[:10].values,
                            "predit": label_encoder.inverse_transform(y_pred[:10])})
comparaison
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

Un modèle « paresseux » répond **toujours `oui`**, sans rien regarder. C'est notre **baseline**. Son exactitude se lit directement dans `value_counts` : c'est la part de `oui` dans le test.

**Questions**
- Quelle est l'exactitude de ce modèle paresseux ? Comparez-la au score de votre modèle.
- Un score de 90 % est-il impressionnant ici ?

```python
part_oui = y_test.value_counts(normalize=True)["oui"]
print("Exactitude du modèle paresseux (toujours oui) :", round(part_oui * 100, 1), "%")
print("Exactitude de notre modèle                    :", round(exactitude * 100, 1), "%")
```

**Ce que vous devez obtenir**

```text
Exactitude du modèle paresseux (toujours oui) : 83.6 %
Exactitude de notre modèle                    : 91.1 %
```

### Q3 · La matrice de confusion

> **Je prédis → J'exécute → Je constate**

La **matrice de confusion** croise la vérité (lignes) et la prédiction (colonnes). Ici, la classe qui compte, la classe « positive », est `non` : c'est elle qu'on cherche à repérer. On place donc `non` en premier :

| | prédit non conforme | prédit conforme |
|---|---|---|
| **vraiment non conforme** | VP (vrais positifs) | FN (faux négatifs : ratés) |
| **vraiment conforme** | FP (faux positifs : fausses alertes) | VN (vrais négatifs) |

**Questions**
- Combien de jours non conformes le modèle a-t-il **ratés** ?
- Combien de fausses alertes a-t-il données ?

```python
code_non = label_encoder.transform(["non"])[0]
code_oui = label_encoder.transform(["oui"])[0]

matrice = confusion_matrix(y_test_enc, y_pred, labels=[code_non, code_oui])
print(matrice)

pd.DataFrame(matrice,
             index=["vraiment non conforme", "vraiment conforme"],
             columns=["prédit non conforme", "prédit conforme"])
```

**Ce que vous devez obtenir**

```text
[[ 42  30]
 [  9 358]]
```

puis le même tableau, lisible :

```text
                       prédit non conforme  prédit conforme
vraiment non conforme                   42               30
vraiment conforme                        9              358
```

### Q4 · Rappel et précision de la classe `non`

> **Je prédis → J'exécute → Je constate**

- **Rappel** de `non` : parmi les jours **vraiment** non conformes (la ligne du haut), quelle part le modèle a-t-il repérée ?
- **Précision** de `non` : parmi les jours **prédits** non conformes (la colonne de gauche), quelle part l'était vraiment ?

`pos_label=code_non` dit à scikit-learn que la classe qui compte est `non`.

**Questions**
- Calculez-les d'abord à la main avec la matrice de Q3, puis vérifiez.
- Ici, rater une plage polluée coûte cher (des baigneurs malades). Faut-il privilégier le rappel ou la précision ? Pourquoi ?

```python
rappel_non = recall_score(y_test_enc, y_pred, pos_label=code_non)
precision_non = precision_score(y_test_enc, y_pred, pos_label=code_non)

print("Rappel de non    :", round(rappel_non * 100, 1), "%")
print("Précision de non :", round(precision_non * 100, 1), "%")
```

**Ce que vous devez obtenir**

```text
Rappel de non    : 58.3 %
Précision de non : 82.4 %
```

### Q5 · Déplacer le seuil

> **Je prédis → J'exécute → Je constate**

`predict` déclare `non` quand la **probabilité** de `non` dépasse **0,5**. Mais ce **seuil** est un choix. Avec `predict_proba`, on récupère la probabilité de `non` pour chaque jour, puis on essaie trois seuils : 0,3, 0,5 et 0,7.

**Questions**
- Si on abaisse le seuil à 0,3, le rappel va-t-il monter ou baisser ? Et la précision ?
- Même question avec un seuil de 0,7.
- Conclusion en deux phrases : quel seuil choisiriez-vous pour protéger les baigneurs ?

```python
proba_non = modele.predict_proba(X_test_prep)[:, code_non]

for seuil in [0.3, 0.5, 0.7]:
    # on déclare « non » quand la probabilité de non atteint le seuil
    y_pred_seuil = [code_non if p >= seuil else code_oui for p in proba_non]
    rappel = recall_score(y_test_enc, y_pred_seuil, pos_label=code_non)
    precision = precision_score(y_test_enc, y_pred_seuil, pos_label=code_non)
    print(f"Seuil {seuil} : rappel de non = {rappel * 100:.1f} %   précision de non = {precision * 100:.1f} %")
```

**Ce que vous devez obtenir**

```text
Seuil 0.3 : rappel de non = 69.4 %   précision de non = 72.5 %
Seuil 0.5 : rappel de non = 58.3 %   précision de non = 82.4 %
Seuil 0.7 : rappel de non = 44.4 %   précision de non = 91.4 %
```

### Q6 · Prédire pour un jour que vous inventez

> **Je prédis → J'exécute → Je constate**

Inventez un jour : changez les valeurs ci-dessous (pluie, débit, plage, saison…). Le jour passe par **exactement le même preprocessing** que l'entraînement (`transform`, jamais `fit`), puis par `predict` et `predict_proba`.

**Questions**
- Pour le jour que vous avez inventé, que va répondre le modèle ? Avec quelle probabilité ?
- Le modèle a-t-il raison, selon vous ? Essayez un jour sec, puis un jour de grosse pluie.

```python
X_nouveau = pd.DataFrame({
    "plage": ["taharuu"],
    "pluie_24h_mm": [35.0],
    "pluie_72h_mm": [80.0],
    "temperature_C": [26.5],
    "debit_m3s": [8.0],
    "saison": ["ete_austral"],
})

X_nouveau_num = pd.DataFrame(scaler.transform(X_nouveau[colonnes_num]), columns=colonnes_num)
X_nouveau_texte = pd.DataFrame(encoder.transform(X_nouveau[colonnes_texte]),
                               columns=encoder.get_feature_names_out())
X_nouveau_prep = pd.concat([X_nouveau_num, X_nouveau_texte], axis=1)

prediction = label_encoder.inverse_transform(modele.predict(X_nouveau_prep))[0]
probabilite_non = modele.predict_proba(X_nouveau_prep)[0, code_non]
print("Conforme ?", prediction, "  (probabilité de non :", round(probabilite_non * 100), "%)")
```

**Ce que vous devez obtenir**

Avec le jour proposé (forte pluie à Taharu'u) :

```text
Conforme ? non   (probabilité de non : 99 %)
```

Avec vos propres valeurs, le résultat change, bien sûr.

### Q7 (bonus) · Quelles features pèsent le plus ?

> **Je prédis → J'exécute → Je constate**

`coef_` donne un coefficient par feature. Attention au sens : scikit-learn calcule la probabilité de la classe numéro 1, c'est-à-dire `oui`. Un coefficient **négatif** pousse donc vers `non`. Les features numériques ont été normalisées : on peut comparer leurs coefficients entre eux, mais avec prudence.

**Questions**
- Quelle feature pousse le plus vers « non conforme » ? Est-ce logique ?

```python
coefficients = pd.DataFrame({"feature": X_train_prep.columns,
                             "coefficient": modele.coef_[0].round(2)})
print(coefficients.sort_values("coefficient").to_string(index=False))
```

**Ce que vous devez obtenir**

```text
             feature  coefficient
        pluie_72h_mm        -1.59
       plage_taharuu        -0.88
        pluie_24h_mm        -0.54
           debit_m3s        -0.54
  saison_ete_austral        -0.07
saison_hiver_austral         0.07
       temperature_C         0.20
        plage_vaiava         0.38
          plage_pk18         0.49
```

---

## Réutiliser ce code pour n'importe quelle classification

Le chemin est **toujours le même** :

> charger → choisir X et y → séparer (stratifié) → trier → normaliser → encoder les features texte → encoder la cible → réunir → `fit` → `score` → `predict` / `predict_proba`

Pour un autre problème de classification, par exemple « ce récif est-il en bonne santé ? » ou « cet abonné va-t-il résilier ? », **seules 4 choses changent** :

| | Ce qui change | Dans ce TP | Pour votre problème |
|---|---|---|---|
| **1** | **Le fichier** (`URL`) | `baignade_taharuu.csv` | l'adresse de votre fichier CSV |
| **2** | **Les features** (X) | toutes les colonnes sauf `date` et `conforme` | les colonnes qui servent à prédire |
| **3** | **La cible** (y) | `conforme` | la colonne à prédire, une catégorie |
| **4** | **La classe qui compte** (`code_non`) | `non` | la classe qu'on ne veut pas rater |

Tout le reste ne change pas : les cellules de l'étape 0 du TP 2 et des questions Q2 à Q6 fonctionnent telles quelles.

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
