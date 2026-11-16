# TP · Régression linéaire : prédire le rayonnement solaire en Polynésie

UE 5.7 « Data Science et IA » · Licence STES · Université de la Polynésie française

Dans le cours, vous avez vu comment prédire le **rayonnement solaire** reçu au sol à partir de la **durée d'ensoleillement**. Dans ce TP, c'est vous qui le faites, sur de vraies mesures de Météo-France.

- **Durée :** 2 × 45 minutes. Le TP 1 prépare les données, le TP 2 entraîne, évalue et utilise le modèle.
- **En binôme**, dans **Google Colab**. Vous n'avez rien à installer.
- **Méthode :** copiez chaque cellule de code dans une nouvelle cellule Colab, dans l'ordre, puis exécutez-la avec Maj + Entrée.

## Le rituel, à chaque étape

> **Je prédis** : je réponds aux questions *avant* d'exécuter la cellule.
> **J'exécute** : je lance la cellule.
> **Je constate** : j'écris deux phrases sur ce que j'observe, et je corrige ma prédiction si besoin.

Se tromper dans sa prédiction n'est pas grave : c'est même comme ça qu'on apprend le plus.

## Les données

Le fichier `rayonnement_pf.csv` contient **3 000 journées réelles** mesurées entre 2000 et 2024 par 10 stations Météo-France de 9 îles : Tahiti, Moorea, Hiva Oa, Takaroa, Anaa, Hao, Tubuai, Mangareva et Rapa. Chaque station compte 300 journées tirées au hasard. **Une ligne = une journée dans une station.**

| Colonne | Contenu |
|---|---|
| `date` | Jour de la mesure |
| `station` | Nom de la station Météo-France |
| `ile` | Île de la station (texte) |
| `latitude`, `longitude` | Position de l'île, en degrés |
| `heures_soleil` | Durée d'ensoleillement de la journée, en heures |
| `humidite` | Humidité moyenne de l'air, en % |
| `pression` | Pression atmosphérique moyenne, en hPa (autour de 1 014) |
| `mois` | Mois, de 1 à 12 |
| `saison` | `été austral` (novembre à avril) ou `hiver austral` (mai à octobre) |
| `rayonnement` | **La cible** : rayonnement solaire reçu au sol dans la journée, en J/cm² |

*Source : Météo-France, « Données climatologiques de base - quotidiennes », département 987, sur data.gouv.fr, Licence Ouverte 2.0, téléchargé le 16/11/2026. Préparation : [`scripts/prepare_rayonnement.py`](../../scripts/prepare_rayonnement.py). Pour la latitude et la longitude, chaque île reçoit une seule valeur : la moyenne de ses stations, car Tubuai en a deux.*

---

## TP 1 · Préparer les données (45 min)

Objectif : transformer le tableau en un tableau **que le modèle sait lire**, c'est-à-dire uniquement des nombres, à la même échelle.

### Étape 0 · Charger les données

> **Je prédis → J'exécute → Je constate**

Cette cellule importe les outils et charge le fichier directement depuis GitHub. Elle affiche la taille du tableau et ses 5 premières lignes.

**Questions**
- Combien de lignes et combien de colonnes le tableau contient-il ?
- Quelle colonne est la cible, c'est-à-dire ce qu'on veut prédire ?

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/clean/rayonnement_pf.csv"
df = pd.read_csv(URL)

print("Nombre de lignes et de colonnes :", df.shape)
df.head()
```

**Ce que vous devez obtenir**

`Nombre de lignes et de colonnes : (3000, 11)`, puis un tableau qui commence ainsi :

```text
         date    station        ile  latitude  longitude  heures_soleil  humidite  pression  mois       saison  rayonnement
0  2000-01-02  MANGAREVA  Mangareva    -23.13    -134.97            5.0        87    1014.0     1  été austral         2129
1  2000-01-03    HIVA-OA    Hiva Oa     -9.81    -139.04            9.1        58    1012.0     1  été austral         2206
2  2000-01-03  MATAURA 1     Tubuai    -23.36    -149.50            7.7        80    1009.6     1  été austral         2598
```

### Étape 1 · Choisir les features et la cible

> **Je prédis → J'exécute → Je constate**

À vous de compléter : remplacez les `___` par les noms de colonnes, entre guillemets et séparés par des virgules. **X** contient les features, c'est-à-dire les informations qu'on donne au modèle. **y** contient la cible.

Les features à utiliser : la durée d'ensoleillement, l'humidité, la pression, la latitude, l'île et la saison.

**Questions**
- Pourquoi X s'écrit-il avec deux paires de crochets `[[ ]]` et y avec une seule ?
- Pourquoi ne met-on surtout pas `rayonnement` dans X ?

```python
# Complétez : les noms de colonnes entre guillemets, séparés par des virgules
X = df[[ ___ ]]
y = df[ ___ ]

print("X :", X.shape, "  y :", y.shape)
display(X.head())
y.head()
```

**Ce que vous devez obtenir**

`X : (3000, 6)   y : (3000,)`, puis :

```text
   heures_soleil  humidite  pression  latitude        ile       saison
0            5.0        87    1014.0    -23.13  Mangareva  été austral
1            9.1        58    1012.0     -9.81    Hiva Oa  été austral
2            7.7        80    1009.6    -23.36     Tubuai  été austral
3            7.0        84    1012.0    -23.36     Tubuai  été austral
4            8.2        80    1011.8    -23.36     Tubuai  été austral
```

```text
0    2129
1    2206
2    2598
3    2373
4    2403
Name: rayonnement, dtype: int64
```

Si vous obtenez une erreur `KeyError`, vérifiez l'orthographe des noms de colonnes : sans accent, en minuscules, avec le `_`.

### Étape 2 · Séparer entraînement et test

> **Je prédis → J'exécute → Je constate**

On met de côté 20 % des journées pour le **test**. Le modèle ne les verra pas pendant l'**entraînement** : elles serviront à vérifier qu'il sait prédire des journées qu'il ne connaît pas. `random_state=42` permet d'obtenir le même tirage que tout le monde.

**Questions**
- Combien de lignes y aura-t-il dans le jeu d'entraînement, et combien dans le jeu de test ?
- Pourquoi ne pas évaluer le modèle sur les journées qui ont servi à l'entraîner ?

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Entraînement :", X_train.shape[0], "lignes")
print("Test         :", X_test.shape[0], "lignes")
```

**Ce que vous devez obtenir**

```text
Entraînement : 2400 lignes
Test         : 600 lignes
```

### Étape 3 · Séparer les colonnes numériques et non numériques

> **Je prédis → J'exécute → Je constate**

Le **preprocessing** n'est pas le même pour les nombres et pour le texte. On range donc les colonnes en deux groupes.

**Questions**
- Quelles colonnes de X ne sont pas numériques ?
- Pourquoi le modèle ne peut-il pas les lire telles quelles ?

```python
colonnes_num = X_train.select_dtypes(include="number").columns
colonnes_texte = X_train.select_dtypes(exclude="number").columns

print("Colonnes numériques     :", list(colonnes_num))
print("Colonnes non numériques :", list(colonnes_texte))
```

**Ce que vous devez obtenir**

```text
Colonnes numériques     : ['heures_soleil', 'humidite', 'pression', 'latitude']
Colonnes non numériques : ['ile', 'saison']
```

### Étape 4 · Normaliser les colonnes numériques

> **Je prédis → J'exécute → Je constate**

Le `StandardScaler` met chaque colonne numérique à la même échelle : moyenne 0 et écart-type 1. Il apprend la moyenne et l'écart-type **sur l'entraînement seulement** (`fit_transform`), puis applique la même transformation au test (`transform`).

**Questions**
- Avant normalisation, autour de quelle valeur se trouve la pression, et autour de quelle valeur les heures de soleil ?
- Pourquoi la pression risque-t-elle d'« écraser » les heures de soleil si on ne normalise pas ?

```python
scaler = StandardScaler()
X_train_num = pd.DataFrame(scaler.fit_transform(X_train[colonnes_num]),
                           columns=colonnes_num, index=X_train.index)
X_test_num = pd.DataFrame(scaler.transform(X_test[colonnes_num]),
                          columns=colonnes_num, index=X_test.index)

print("AVANT la normalisation")
print(X_train[["pression", "heures_soleil"]].describe().round(1))
print("\nAPRÈS la normalisation")
print(X_train_num[["pression", "heures_soleil"]].describe().round(1))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for colonne, couleur in [("heures_soleil", "#eda100"), ("pression", "#2a78d6")]:
    axes[0].hist(X_train[colonne], bins=40, color=couleur, alpha=0.7, label=colonne)
    axes[1].hist(X_train_num[colonne], bins=40, color=couleur, alpha=0.7, label=colonne)
axes[0].set_title("Avant : des échelles très différentes")
axes[1].set_title("Après : même échelle, centrée sur 0")
for ax in axes:
    ax.set_ylabel("Nombre de journées")
    ax.legend()
plt.show()
```

**Ce que vous devez obtenir**

```text
AVANT la normalisation
       pression  heures_soleil
count    2400.0         2400.0
mean     1014.5            6.4
std         3.4            3.3
min       998.6            0.0
...
max      1031.5           12.6

APRÈS la normalisation
       pression  heures_soleil
count    2400.0         2400.0
mean        0.0            0.0
std         1.0            1.0
min        -4.6           -1.9
...
max         5.0            1.9
```

Sur le graphique de gauche, les deux histogrammes sont très loin l'un de l'autre : les heures de soleil sont collées à 0 et la pression autour de 1 000. Sur celui de droite, les deux se superposent autour de 0.

### Étape 5 · Encoder les colonnes non numériques

> **Je prédis → J'exécute → Je constate**

Le `OneHotEncoder` transforme chaque catégorie en une colonne de 0 et de 1. Par exemple, la colonne `ile_Rapa` vaut 1 si la journée a été mesurée à Rapa, et 0 sinon. `handle_unknown="ignore"` évite une erreur si une île inconnue se présente plus tard.

**Questions**
- En combien de colonnes la colonne `ile` va-t-elle se transformer ? Et la colonne `saison` ?

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
Nouvelles colonnes : ['ile_Anaa', 'ile_Hao', 'ile_Hiva Oa', 'ile_Mangareva', 'ile_Moorea', 'ile_Rapa', 'ile_Tahiti', 'ile_Takaroa', 'ile_Tubuai', 'saison_hiver austral', 'saison_été austral']
```

Vous obtenez donc 9 colonnes `ile_…` et 2 colonnes `saison_…`, suivies d'un tableau de 0 et de 1. Sur chaque ligne, il y a exactement un 1 parmi les colonnes `ile_…` et un 1 parmi les colonnes `saison_…`.

### Étape 6 · Tout réunir

> **Je prédis → J'exécute → Je constate**

On recolle les colonnes normalisées et les colonnes encodées, côte à côte, pour obtenir le tableau final que le modèle va lire.

**Questions**
- Combien de colonnes aura `X_train_prepare` ? Pourquoi plus que les 6 features de départ ?

```python
X_train_prepare = pd.concat([X_train_num, X_train_texte], axis=1)
X_test_prepare = pd.concat([X_test_num, X_test_texte], axis=1)

print("X_train_prepare :", X_train_prepare.shape, "  X_test_prepare :", X_test_prepare.shape)
X_train_prepare.head().round(2)
```

**Ce que vous devez obtenir**

`X_train_prepare : (2400, 15)   X_test_prepare : (600, 15)`, puis un tableau qui ne contient que des nombres :

```text
      heures_soleil  humidite  pression  latitude  ile_Anaa  ile_Hao  ...  ile_Tubuai  saison_hiver austral  saison_été austral
642           -0.18      0.98      1.73     -0.84       0.0      0.0  ...         1.0                   1.0                 0.0
700            0.99     -0.00     -0.07     -0.84       0.0      0.0  ...         1.0                   0.0                 1.0
226           -0.30     -0.71      0.68     -0.80       0.0      0.0  ...         0.0                   0.0                 1.0
```

**Fin du TP 1.** Vos données sont prêtes. Vous n'avez rien à sauvegarder : le TP 2 commence par une cellule qui refait tout le TP 1 d'un coup.

---

## TP 2 · Entraîner, évaluer, prédire (45 min)

Objectif : suivre les quatre verbes du cours, **importer-créer → entraîner (`fit`) → évaluer (`score`) → prédire (`predict`)**.

### Étape 0 · Tout le TP 1 en une fois

> **J'exécute**

Ouvrez un nouveau notebook Colab, ou continuez dans celui du TP 1, et exécutez cette cellule. Elle refait tout le TP 1, réponse de l'étape 1 comprise.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error

URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/clean/rayonnement_pf.csv"
df = pd.read_csv(URL)

X = df[["heures_soleil", "humidite", "pression", "latitude", "ile", "saison"]]
y = df["rayonnement"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

X_train_prepare = pd.concat([X_train_num, X_train_texte], axis=1)
X_test_prepare = pd.concat([X_test_num, X_test_texte], axis=1)
print("TP 1 refait :", X_train_prepare.shape, X_test_prepare.shape)
```

**Ce que vous devez obtenir**

`TP 1 refait : (2400, 15) (600, 15)`

### Étape 1 · Importer et créer le modèle

> **Je prédis → J'exécute → Je constate**

On importe la régression linéaire de scikit-learn, puis on crée un modèle vide.

**Questions**
- À ce stade, le modèle sait-il déjà prédire le rayonnement ?

```python
from sklearn.linear_model import LinearRegression

modele = LinearRegression()
modele
```

**Ce que vous devez obtenir**

Une petite boîte qui affiche `LinearRegression()`, et rien d'autre.

### Étape 2 · Entraîner le modèle

> **Je prédis → J'exécute → Je constate**

`fit` : le modèle cherche, sur le jeu d'entraînement, le coefficient de chaque feature qui donne les meilleures prédictions. On affiche ensuite ces coefficients, triés du plus grand au plus petit.

**Questions**
- Quel sera le coefficient le plus grand, à votre avis ?
- Que veut dire la phrase « +1 heure de soleil = + … J/cm² » ? Comparez avec le cours, qui donnait environ 145.
- Le coefficient de l'humidité est-il positif ou négatif ? Est-ce logique ?

```python
modele.fit(X_train_prepare, y_train)

coefficients = pd.DataFrame({"feature": X_train_prepare.columns,
                             "coefficient": modele.coef_.round(1)})
coefficients = coefficients.sort_values("coefficient", ascending=False)
print("Intercept :", round(modele.intercept_, 1))
print(coefficients.to_string(index=False))

# heures_soleil a été normalisée : son coefficient vaut pour « 1 écart-type » de soleil.
# On le ramène à « 1 heure de soleil en plus » :
coef_soleil = modele.coef_[X_train_prepare.columns.get_loc("heures_soleil")]
ecart_type_soleil = scaler.scale_[list(colonnes_num).index("heures_soleil")]
print("\n+1 heure de soleil = +", round(coef_soleil / ecart_type_soleil), "J/cm²")
```

**Ce que vous devez obtenir**

```text
Intercept : 1838.3
             feature  coefficient
       heures_soleil        485.9
  saison_été austral        194.3
         ile_Hiva Oa        126.3
            ile_Rapa         78.2
            latitude         33.6
            pression         20.7
            ile_Anaa          9.1
       ile_Mangareva          0.8
          ile_Tubuai         -0.1
         ile_Takaroa        -18.5
            humidite        -56.5
          ile_Tahiti        -62.1
          ile_Moorea        -63.6
             ile_Hao        -70.1
saison_hiver austral       -194.3

+1 heure de soleil = + 146 J/cm²
```

### Étape 3 · Évaluer le modèle

> **Je prédis → J'exécute → Je constate**

`score` donne le **R²**, c'est-à-dire la part des variations du rayonnement que le modèle explique. La **MAE** (erreur absolue moyenne) donne l'erreur moyenne, en J/cm². On compare avec une **baseline** qui prédit toujours la moyenne, sans rien regarder.

**Questions**
- Entre quelles valeurs le R² peut-il être ? Que voudrait dire un R² de 0,65 ?
- La MAE du modèle sera-t-elle plus petite ou plus grande que celle de la baseline ?
- Au vu des résultats, le modèle a-t-il appris quelque chose ?

```python
r2 = modele.score(X_test_prepare, y_test)
y_pred = modele.predict(X_test_prepare)
mae = mean_absolute_error(y_test, y_pred)

baseline = [y_train.mean()] * len(y_test)  # on prédit toujours la moyenne
mae_baseline = mean_absolute_error(y_test, baseline)

print("R² (part des variations expliquée) :", round(r2, 2))
print("MAE du modèle      :", round(mae), "J/cm²")
print("MAE de la baseline :", round(mae_baseline), "J/cm²")
```

**Ce que vous devez obtenir**

```text
R² (part des variations expliquée) : 0.74
MAE du modèle      : 255 J/cm²
MAE de la baseline : 538 J/cm²
```

### Étape 4 · Comparer prédictions et réalité

> **Je prédis → J'exécute → Je constate**

Chaque point est une journée du jeu de test. Elle est placée selon son rayonnement réel (horizontalement) et le rayonnement prédit par le modèle (verticalement).

**Questions**
- Où seraient les points si le modèle était parfait ?

```python
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, s=10, alpha=0.5, color="#2a78d6")
plt.plot([0, 4000], [0, 4000], color="#eb6834", linewidth=2, label="Prédiction parfaite")
plt.xlabel("Rayonnement réel (J/cm²)")
plt.ylabel("Rayonnement prédit (J/cm²)")
plt.title("Jeu de test : rayonnement prédit et réel")
plt.legend()
plt.show()
```

**Ce que vous devez obtenir**

Un nuage de points allongé le long de la diagonale orange, assez épais autour d'elle. Aucune prédiction ne dépasse environ 2 900 J/cm², alors que certaines journées réelles du test approchent 3 500. Aux deux extrémités, les points s'écartent de la diagonale : au-dessus pour les journées les plus faibles, en dessous pour les plus fortes.

### Étape 5 · Prédire pour de nouvelles journées

> **Je prédis → J'exécute → Je constate**

On invente trois journées :
- Bora Bora en janvier, avec 9 h de soleil ;
- Rapa en juillet, avec 4 h de soleil ;
- Tahiti en décembre, avec 11 h de soleil.

Elles doivent passer par **exactement le même preprocessing** que l'entraînement : `transform` seulement, jamais `fit`. Ensuite, `predict`. On convertit aussi en kWh/m² (1 J/cm² = 1/360 kWh/m²), l'unité des panneaux solaires.

**Questions**
- Laquelle des trois journées recevra le plus de rayonnement ? Laquelle en recevra le moins ? Pourquoi ?
- Bora Bora n'est pas dans les données : que va faire le modèle avec la colonne `ile` ?

```python
X_nouveau = pd.DataFrame({
    "heures_soleil": [9, 4, 11],
    "humidite": [76, 76, 75],
    "pression": [1012, 1020, 1012],
    "latitude": [-16.50, -27.62, -17.56],
    "ile": ["Bora Bora", "Rapa", "Tahiti"],
    "saison": ["été austral", "hiver austral", "été austral"],
})

# EXACTEMENT le même preprocessing qu'à l'entraînement : transform, jamais fit
X_nouveau_num = pd.DataFrame(scaler.transform(X_nouveau[colonnes_num]), columns=colonnes_num)
X_nouveau_texte = pd.DataFrame(encoder.transform(X_nouveau[colonnes_texte]),
                               columns=encoder.get_feature_names_out())
X_nouveau_prepare = pd.concat([X_nouveau_num, X_nouveau_texte], axis=1)

X_nouveau["rayonnement_predit"] = modele.predict(X_nouveau_prepare).round()
X_nouveau["kWh_par_m2"] = (X_nouveau["rayonnement_predit"] / 360).round(1)
X_nouveau
```

**Ce que vous devez obtenir**

```text
   heures_soleil  humidite  pression  latitude        ile         saison  rayonnement_predit  kWh_par_m2
0              9        76      1012    -16.50  Bora Bora    été austral              2426.0         6.7
1              4        76      1020    -27.62       Rapa  hiver austral              1358.0         3.8
2             11        75      1012    -17.56     Tahiti    été austral              2656.0         7.4
```

### Étape 6 (facultative) · Moins de features, moins bien ?

> **Je prédis → J'exécute → Je constate**

On retire `humidite` et `pression`, on refait le preprocessing et l'entraînement, puis on compare le R².

**Questions**
- Le R² va-t-il baisser beaucoup, un peu, ou pas du tout ?
- Plus de colonnes, est-ce toujours mieux ?

```python
features_reduites = ["heures_soleil", "latitude", "ile", "saison"]
X_train_r = X_train[features_reduites]
X_test_r = X_test[features_reduites]
colonnes_num_r = X_train_r.select_dtypes(include="number").columns

scaler_r = StandardScaler()
encoder_r = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_r_prepare = pd.concat([
    pd.DataFrame(scaler_r.fit_transform(X_train_r[colonnes_num_r]), columns=colonnes_num_r, index=X_train_r.index),
    pd.DataFrame(encoder_r.fit_transform(X_train_r[colonnes_texte]), columns=encoder_r.get_feature_names_out(), index=X_train_r.index),
], axis=1)
X_test_r_prepare = pd.concat([
    pd.DataFrame(scaler_r.transform(X_test_r[colonnes_num_r]), columns=colonnes_num_r, index=X_test_r.index),
    pd.DataFrame(encoder_r.transform(X_test_r[colonnes_texte]), columns=encoder_r.get_feature_names_out(), index=X_test_r.index),
], axis=1)

modele_r = LinearRegression()
modele_r.fit(X_train_r_prepare, y_train)
print("R² avec les 6 features :", round(modele.score(X_test_prepare, y_test), 3))
print("R² avec 4 features     :", round(modele_r.score(X_test_r_prepare, y_test), 3))
```

**Ce que vous devez obtenir**

```text
R² avec les 6 features : 0.739
R² avec 4 features     : 0.733
```

---

## Ce qu'il faut rendre

Un notebook Colab par binôme, avec les deux noms en haut. Il doit contenir :
- toutes les cellules du TP 1 et du TP 2, **exécutées**, avec leurs résultats visibles ;
- sous chaque étape, vos réponses écrites aux questions : votre prédiction, puis vos deux phrases « Je constate ».

Partagez-le avec votre enseignant via « Partager » dans Colab, ou téléchargez-le avec Fichier > Télécharger > .ipynb.

## Glossaire

| Mot | En une phrase |
|---|---|
| **Features / X** | Les informations données au modèle pour qu'il prédise (ici : soleil, humidité, pression, latitude, île, saison). |
| **Cible / y** | Ce qu'on veut prédire (ici : le rayonnement, en J/cm²). |
| **Entraînement / test** | Les journées qui servent à apprendre, et celles, mises de côté, qui servent à vérifier. |
| **Preprocessing** | La préparation des données pour que le modèle puisse les lire : normaliser et encoder. |
| **Normaliser (StandardScaler)** | Mettre chaque colonne numérique à la même échelle (moyenne 0, écart-type 1). |
| **Encoder (OneHotEncoder)** | Transformer une colonne de texte en colonnes de 0 et de 1, une par catégorie. |
| **R² (score)** | La part des variations de la cible que le modèle explique : 1 = parfait, 0 = pas mieux que la moyenne. |
| **MAE et baseline** | L'erreur moyenne des prédictions, à comparer avec celle d'une baseline qui prédit toujours la moyenne. |
