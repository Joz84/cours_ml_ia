# TP · Quels profils d'abonnés à l'eau ?

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Joz84/cours_ml_ia/blob/main/tp/clustering/TP_clustering_ELEVE.ipynb)

UE 5.7 « Data Science et IA » · Licence STES · Université de la Polynésie française

Le service de l'eau veut lancer une campagne d'économies d'eau. Plutôt qu'un seul message pour tout le monde, il voudrait **un message par type d'abonné**. Le problème : personne n'a jamais classé les abonnés. Il n'y a **pas de cible**, pas de « bonne réponse » à apprendre. C'est le terrain du **clustering** : le **K-Means** va regrouper les abonnés qui se ressemblent, et c'est vous qui donnerez un nom à chaque groupe.

- **Durée :** 2 × 45 minutes. Le TP 1 prépare les données, le TP 2 regroupe, choisit K et nomme les groupes.
- **En binôme**, dans **Google Colab**. Vous n'avez rien à installer.
- **Méthode :** copiez chaque cellule de code dans une nouvelle cellule Colab, dans l'ordre, puis exécutez-la avec Maj + Entrée.
- **Quand une cellule est à compléter**, un encadré « Ce que vous dites à l'assistant » vous donne la phrase à dire à ChatGPT ou à Claude.

## Le rituel, à chaque étape

> **Je prédis** : je réponds aux questions *avant* d'exécuter la cellule.
> **J'exécute** : je lance la cellule.
> **Je constate** : j'écris deux phrases sur ce que j'observe, et je corrige ma prédiction si besoin.

Se tromper dans sa prédiction n'est pas grave : c'est même comme ça qu'on apprend le plus.

## Les données

Le fichier `abonnes_eau.csv` décrit **600 abonnés** au service de l'eau de 3 communes de Tahiti (Papara, Paea et Punaauia). **Une ligne = un abonné.**

| Colonne | Contenu |
|---|---|
| `abonne` | Identifiant de l'abonné (texte) |
| `commune` | `Papara`, `Paea` ou `Punaauia` (texte) |
| `conso_moy_m3_mois` | Consommation moyenne, en m³ par mois |
| `pic_ete_pct` | Hausse de la consommation pendant l'été austral, en % |
| `part_nuit_pct` | Part de l'eau consommée la nuit, en % |
| `nb_personnes` | Nombre de personnes déclarées dans le logement (0 pour un local professionnel) |
| `surface_jardin_m2` | Surface de jardin, en m² |

**Attention : ces données sont simulées.** Elles ont été fabriquées pour ce TP, avec des profils d'abonnés réalistes cachés dedans, que vous allez devoir retrouver. Le script qui les fabrique est [`scripts/prepare_abonnes_eau.py`](../../scripts/prepare_abonnes_eau.py). Ne le lisez pas avant la fin du TP : il dévoile les réponses.

---

## TP 1 · Le preprocessing de clustering (45 min)

Objectif : préparer un tableau **que le K-Means sait lire**. En classification, il y avait 5 étapes. Ici, il n'y a pas de cible, donc **pas de séparation entraînement / test** et **pas d'encodage de la cible** : il n'en reste que 3, **trier**, **choisir les colonnes** et **normaliser**.

### Étape 0 · Charger les données

> **Je prédis → J'exécute → Je constate**

Cette cellule importe les outils et charge le fichier directement depuis GitHub. Elle affiche la taille du tableau et ses 5 premières lignes.

**Questions**
- Combien d'abonnés et combien de colonnes ?

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/clean/abonnes_eau.csv"
df = pd.read_csv(URL)

print("Nombre de lignes et de colonnes :", df.shape)
df.head()
```

**Ce que vous devez obtenir**

`Nombre de lignes et de colonnes : (600, 7)`, puis un tableau qui commence ainsi :

```text
   abonne commune  conso_moy_m3_mois  pic_ete_pct  part_nuit_pct  nb_personnes  surface_jardin_m2
0  AB0001  Papara                9.7           12             12             2                 30
1  AB0002  Papara               20.3           46             39             5                688
2  AB0003  Papara               64.9           27             10             0                  7
```

### Étape 1 · Où est la cible ?

> **Je prédis → J'exécute → Je constate**

On affiche la liste des colonnes.

**Questions**
- Quelle colonne est la cible ?
- S'il n'y a pas de cible, pourquoi n'a-t-on besoin ni de séparer entraînement et test, ni d'encoder la cible ?

```python
print(list(df.columns))
```

**Ce que vous devez obtenir**

```text
['abonne', 'commune', 'conso_moy_m3_mois', 'pic_ete_pct', 'part_nuit_pct', 'nb_personnes', 'surface_jardin_m2']
```

Aucune de ces colonnes n'est une cible : **il n'y a rien à prédire**, seulement des abonnés à regrouper.

### Étape 2 · Trier les colonnes numériques et non numériques

> **Je prédis → J'exécute → Je constate**

Comme en classification, on range les colonnes en deux groupes.

**Questions**
- Quelles colonnes ne sont pas numériques ?

```python
colonnes_num = df.select_dtypes(include="number").columns
colonnes_texte = df.select_dtypes(exclude="number").columns

print("Colonnes numériques     :", list(colonnes_num))
print("Colonnes non numériques :", list(colonnes_texte))
```

**Ce que vous devez obtenir**

```text
Colonnes numériques     : ['conso_moy_m3_mois', 'pic_ete_pct', 'part_nuit_pct', 'nb_personnes', 'surface_jardin_m2']
Colonnes non numériques : ['abonne', 'commune']
```

On **écarte `abonne` et `commune`** pour ce TP. `abonne` est un simple identifiant, différent pour chacun, qui ne dit rien des habitudes de consommation. `commune` dit *où* habite l'abonné, et nous cherchons des groupes selon *comment* il consomme. Nous la retrouverons à la fin, en Q7.

### Étape 3 · Choisir les colonnes à garder

> **Je prédis → J'exécute → Je constate**

À vous de choisir les colonnes que le K-Means va utiliser pour regrouper les abonnés : toutes les colonnes numériques.

> **Ce que vous dites à l'assistant**
>
> « Dans Google Colab, j'ai un DataFrame pandas `df` et une variable `colonnes_num` qui contient la liste de ses colonnes numériques. Écris-moi une ligne de code qui crée `X` avec seulement ces colonnes. »

Collez la ligne à la place des `___`.

**Questions**
- Combien de colonnes aura X ?

```python
# À compléter : collez ici la ligne donnée par l'assistant
X = ___

print("X :", X.shape)
print(list(X.columns))
```

**Ce que vous devez obtenir**

```text
X : (600, 5)
['conso_moy_m3_mois', 'pic_ete_pct', 'part_nuit_pct', 'nb_personnes', 'surface_jardin_m2']
```

### Étape 4 · Normaliser

> **Je prédis → J'exécute → Je constate**

Le K-Means regroupe les abonnés **proches** les uns des autres, en calculant des **distances**. Si une colonne a de très grands nombres, c'est elle qui décide de tout : la **normalisation est obligatoire**. Le `StandardScaler` met chaque colonne à moyenne 0 et écart-type 1.

**Questions**
- Avant normalisation, quelle colonne a les plus grands nombres ?

```python
scaler = StandardScaler()
X_normalise = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print("AVANT la normalisation")
print(X.describe().round(1))
print("\nAPRÈS la normalisation")
print(X_normalise.describe().round(1))
```

**Ce que vous devez obtenir**

```text
AVANT la normalisation
       conso_moy_m3_mois  pic_ete_pct  part_nuit_pct  nb_personnes  surface_jardin_m2
count              600.0        600.0          600.0         600.0              600.0
mean                24.8         24.1           17.6           2.4              213.2
std                 19.6         16.6           11.7           1.9              270.2
min                  2.6          0.0            1.0           0.0                0.0
...
max                 94.9         71.0           49.0           7.0             1048.0

APRÈS la normalisation
       conso_moy_m3_mois  pic_ete_pct  part_nuit_pct  nb_personnes  surface_jardin_m2
count              600.0        600.0          600.0         600.0              600.0
mean                -0.0          0.0           -0.0           0.0               -0.0
std                  1.0          1.0            1.0           1.0                1.0
...
```

Dans Colab, ce tableau peut s'afficher sur deux blocs, car il est large. `surface_jardin_m2` se compte en centaines de m², alors que `nb_personnes` va de 0 à 7.

**Fin du TP 1.** Vos données sont prêtes. Vous n'avez rien à sauvegarder : le TP 2 commence par une cellule qui refait tout le TP 1 d'un coup.

---

## TP 2 · Regrouper, choisir K, nommer (45 min)

Objectif : suivre les verbes du cours, **importer et créer → `fit` → lire les groupes → `predict`**, puis répondre aux questions Q1 à Q7 par écrit, dans le notebook, sous chaque question.

### Étape 0 · Tout le TP 1 en une fois

> **J'exécute**

Ouvrez un nouveau notebook Colab, ou continuez dans celui du TP 1, et exécutez cette cellule. Elle refait tout le TP 1, réponse de l'étape 3 comprise.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/clean/abonnes_eau.csv"
df = pd.read_csv(URL)

colonnes_num = df.select_dtypes(include="number").columns
colonnes_texte = df.select_dtypes(exclude="number").columns
X = df[colonnes_num]

scaler = StandardScaler()
X_normalise = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print("TP 1 refait :", X_normalise.shape)
```

**Ce que vous devez obtenir**

`TP 1 refait : (600, 5)`

### Étape 1 · Importer et créer le modèle

> **Je prédis → J'exécute → Je constate**

On importe le K-Means et on crée un modèle qui cherchera **3 groupes**. Ce nombre K, **c'est nous qui le choisissons** : le modèle ne le devine pas.

**Questions**
- Pourquoi 3 ? Que se passerait-il avec K = 600 ?

```python
from sklearn.cluster import KMeans

modele = KMeans(n_clusters=3, random_state=42)
modele
```

**Ce que vous devez obtenir**

Une petite boîte qui affiche `KMeans(n_clusters=3, random_state=42)`.

### Étape 2 · Regrouper : `fit`

> **Je prédis → J'exécute → Je constate**

`fit` place 3 **centroïdes**, c'est-à-dire 3 abonnés « moyens », puis range chaque abonné dans le groupe du centroïde le plus proche, et recommence jusqu'à ce que plus rien ne bouge. Remarquez qu'on lui donne seulement `X_normalise` : **pas de y**, puisqu'il n'y a pas de cible.

**Questions**
- En régression et en classification, on écrivait `fit(X_train, y_train)`. Pourquoi n'y a-t-il qu'un seul argument ici ?

```python
modele.fit(X_normalise)
```

**Ce que vous devez obtenir**

La même petite boîte `KMeans(n_clusters=3, random_state=42)` : les groupes sont faits.

### Étape 3 · Ajouter le groupe de chaque abonné

> **Je prédis → J'exécute → Je constate**

`modele.labels_` contient le numéro de **cluster** de chaque abonné. On l'ajoute comme nouvelle colonne `groupe` dans `df`.

**Questions**
- Les numéros 0, 1 et 2 ont-ils un sens en eux-mêmes ?

```python
df["groupe"] = modele.labels_
df.head()
```

**Ce que vous devez obtenir**

Le même tableau qu'au début, avec une colonne `groupe` en plus :

```text
   abonne commune  conso_moy_m3_mois  pic_ete_pct  part_nuit_pct  nb_personnes  surface_jardin_m2  groupe
0  AB0001  Papara                9.7           12             12             2                 30       1
1  AB0002  Papara               20.3           46             39             5                688       0
2  AB0003  Papara               64.9           27             10             0                  7       2
```

### Étape 4 · Le portrait moyen de chaque groupe

> **Je prédis → J'exécute → Je constate**

On calcule la moyenne de chaque colonne numérique, groupe par groupe. C'est ce tableau qui permet de comprendre qui est dans chaque groupe.

**Questions**
- Quel groupe consomme le plus ? Lequel a les plus grands jardins ?

```python
portrait = df.groupby("groupe")[colonnes_num].mean().round(1)
portrait
```

**Ce que vous devez obtenir**

```text
        conso_moy_m3_mois  pic_ete_pct  part_nuit_pct  nb_personnes  surface_jardin_m2
groupe
0                    24.9         43.8           31.8           4.6              548.0
1                    10.3         10.1           12.0           1.8               28.7
2                    58.7         21.9            5.2           0.0               51.8
```

### Q1 · Combien d'abonnés dans chaque groupe ?

> **Je prédis → J'exécute → Je constate**

**Questions**
- Combien d'abonnés compte chaque groupe ? Les groupes ont-ils la même taille ?

```python
df["groupe"].value_counts().sort_index()
```

**Ce que vous devez obtenir**

```text
groupe
0    208
1    275
2    117
Name: count, dtype: int64
```

### Q2 · Donner un nom à chaque groupe

Relisez le portrait moyen de l'étape 4. **Le modèle numérote les groupes, c'est vous qui les nommez.**

**Questions**
- Donnez un nom à chaque groupe, et décrivez-le en une phrase, comme vous l'écririez à un élu de la commune.

### Q3 · La méthode du coude : combien de groupes ?

> **Je prédis → J'exécute → Je constate**

On refait le K-Means pour K = 1 à 8 et on note à chaque fois l'**inertie**, c'est-à-dire la distance totale entre chaque abonné et son centroïde. Plus K est grand, plus l'inertie baisse. On cherche le **coude** : le K à partir duquel ajouter un groupe ne fait presque plus baisser l'inertie.

**Questions**
- Où est le coude ?
- Est-ce cohérent avec les trois profils que vous avez nommés en Q2 ?

```python
inerties = []
for k in range(1, 9):
    modele_k = KMeans(n_clusters=k, random_state=42)
    modele_k.fit(X_normalise)
    inerties.append(modele_k.inertia_)
    print(f"K = {k} : inertie = {modele_k.inertia_:.0f}")

plt.figure(figsize=(7, 4))
plt.plot(range(1, 9), inerties, marker="o", color="#2a78d6", linewidth=2)
plt.xlabel("Nombre de groupes K")
plt.ylabel("Inertie")
plt.title("Méthode du coude")
plt.grid(alpha=0.3)
plt.show()
```

**Ce que vous devez obtenir**

```text
K = 1 : inertie = 3000
K = 2 : inertie = 1164
K = 3 : inertie = 521
K = 4 : inertie = 448
K = 5 : inertie = 410
K = 6 : inertie = 366
K = 7 : inertie = 344
K = 8 : inertie = 316
```

puis une courbe qui chute très fort de K = 1 à K = 3, et devient presque plate ensuite : elle forme un « coude » net.

### Q4 · Et sans normaliser ?

> **Je prédis → J'exécute → Je constate**

On refait exactement le même K-Means, mais sur `X`, **sans normaliser**, et on compare les groupes obtenus.

**Questions**
- Les groupes vont-ils changer ? Comparez les effectifs et les moyennes avec ceux de Q1 et de l'étape 4.
- Pourquoi est-ce différent ? Quelle colonne a « écrasé » les autres ?

```python
modele_brut = KMeans(n_clusters=3, random_state=42)
modele_brut.fit(X)
df["groupe_brut"] = modele_brut.labels_

print(df["groupe_brut"].value_counts().sort_index())
df.groupby("groupe_brut")[colonnes_num].mean().round(1)
```

**Ce que vous devez obtenir**

```text
groupe_brut
0    115
1    403
2     82
Name: count, dtype: int64
```

puis :

```text
             conso_moy_m3_mois  pic_ete_pct  part_nuit_pct  nb_personnes  surface_jardin_m2
groupe_brut
0                         24.8         43.0           30.6           4.5              452.0
1                         24.7         14.6           10.7           1.4               38.8
2                         25.3         44.2           32.9           4.6              735.6
```

### Q5 · Un nouvel abonné

> **Je prédis → J'exécute → Je constate**

Un nouvel abonné arrive. Avant `predict`, il doit être normalisé avec **le même `scaler`** que les autres (`transform`, jamais `fit`) : sinon, ses valeurs ne seraient pas sur la même échelle.

**Questions**
- Dans quel groupe va-t-il tomber ?
- Quel message d'économie d'eau lui enverriez-vous ?

```python
nouvel_abonne = pd.DataFrame({
    "conso_moy_m3_mois": [31.0],
    "pic_ete_pct": [50],
    "part_nuit_pct": [35],
    "nb_personnes": [5],
    "surface_jardin_m2": [700],
})

nouvel_abonne_normalise = pd.DataFrame(scaler.transform(nouvel_abonne), columns=nouvel_abonne.columns)
print("Groupe du nouvel abonné :", modele.predict(nouvel_abonne_normalise)[0])
```

**Ce que vous devez obtenir**

```text
Groupe du nouvel abonné : 0
```

### Q6 (bonus) · Un autre avis : le score de silhouette

Le **score de silhouette** mesure si chaque abonné est bien plus proche de son groupe que des autres groupes : plus il est proche de 1, plus les groupes sont nets.

> **Ce que vous dites à l'assistant**
>
> « Dans Google Colab, j'ai un DataFrame `X_normalise` déjà normalisé. Écris-moi le code qui calcule le `silhouette_score` de scikit-learn pour un `KMeans(n_clusters=k, random_state=42)` avec K de 2 à 8, qui affiche le score pour chaque K, et qui me dit quel K a le meilleur score. »

Collez le code de l'assistant dans la cellule ci-dessous.

**Questions**
- Quel K l'assistant recommande-t-il ? Est-il d'accord avec le coude ?
- Qui tranche, à votre avis : le coude, la silhouette, ou vous ?

```python
# À compléter : collez ici le code donné par l'assistant
```

**Ce que vous devez obtenir**

Le code de l'assistant peut être écrit un peu différemment, mais il doit donner ces scores :

```text
K = 2 : silhouette = 0.577
K = 3 : silhouette = 0.629
K = 4 : silhouette = 0.503
K = 5 : silhouette = 0.500
K = 6 : silhouette = 0.432
K = 7 : silhouette = 0.422
K = 8 : silhouette = 0.259

Meilleur K selon la silhouette : 3
```

### Q7 (bonus) · Et la commune ?

> **Je prédis → J'exécute → Je constate**

La colonne `commune` n'a pas servi à faire les groupes. On croise maintenant `groupe` et `commune` avec `pd.crosstab` : le tableau compte les abonnés de chaque groupe dans chaque commune.

**Questions**
- Un groupe est-il concentré dans une commune ?
- Qu'en déduire pour organiser la campagne ?

```python
pd.crosstab(df["groupe"], df["commune"])
```

**Ce que vous devez obtenir**

```text
commune  Paea  Papara  Punaauia
groupe
0          59      97        52
1         107      77        91
2          23      24        70
```

---

## Réutiliser ce code pour n'importe quel clustering

Le chemin est **toujours le même** :

> charger → trier → choisir les colonnes numériques → normaliser → `KMeans` → `fit` → portrait moyen par groupe → coude → nommer

Pour un autre problème, par exemple des types de récifs, de parcelles agricoles ou de journées météo, **seules 3 choses changent** :

| | Ce qui change | Dans ce TP | Pour votre problème |
|---|---|---|---|
| **1** | **Le fichier** (`URL`) | `abonnes_eau.csv` | l'adresse de votre fichier CSV |
| **2** | **Les colonnes** (X) | les 5 colonnes numériques | les colonnes qui décrivent vos individus |
| **3** | **Le nombre de groupes** (`n_clusters`) | 3, d'après le coude | le K que vous choisissez avec la méthode du coude |

Tout le reste ne change pas. Et la dernière étape reste toujours la vôtre : **le modèle numérote les groupes, c'est vous qui les nommez.**

## Ce qu'il faut rendre

Un notebook Colab par binôme, avec les deux noms en haut. Il doit contenir :
- toutes les cellules du TP 1 et du TP 2, **exécutées**, avec leurs résultats visibles ;
- sous chaque étape et chaque question Q1 à Q7, votre réponse écrite dans la cellule « Votre réponse ».

Partagez-le avec votre enseignant via « Partager » dans Colab, ou téléchargez-le avec Fichier > Télécharger > .ipynb.

## Glossaire

| Mot | En une phrase |
|---|---|
| **Clustering** | Regrouper des individus qui se ressemblent, **sans cible** : il n'y a pas de bonne réponse à apprendre. |
| **K-Means** | L'algorithme qui forme K groupes autour de K centroïdes. |
| **K** | Le nombre de groupes : **c'est nous qui le choisissons**, pas le modèle. |
| **Centroïde** | Le point moyen d'un groupe : un abonné « type », qui n'existe pas forcément. |
| **Cluster** | Un groupe d'individus proches les uns des autres. |
| **Normalisation obligatoire** | Le K-Means calcule des distances : sans normalisation, la colonne aux plus grands nombres décide seule. |
| **Inertie** | La distance totale entre chaque individu et son centroïde : elle baisse toujours quand K augmente. |
| **Méthode du coude** | Choisir le K à partir duquel l'inertie ne baisse presque plus. |
