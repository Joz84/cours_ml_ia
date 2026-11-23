# Annexe · TP-prompting : explorer, nettoyer et visualiser avec un assistant IA

UE 5.7 « Data Science et IA » · Licence STES · Université de la Polynésie française

Cette annexe propose un atelier différent des TP : on y travaille **avec un assistant IA** (ChatGPT, Claude), en dialogue, à partir de prompts. Les données sont celles d'un **panorama de l'IA et du machine learning** : les modèles d'IA publiés depuis 1950 (Epoch AI), l'usage de l'IA dans les entreprises européennes (Eurostat), les praticiens de la data science (Kaggle), les incidents liés à l'IA (Our World in Data) et l'effet de l'IA sur les objectifs de développement durable (Vinuesa et al. 2020).

La page contient :
- **l'atelier**, une suite de prompts pour explorer puis nettoyer un jeu de données, et une cellule de code par graphique ;
- **les jeux de données**, bruts puis nettoyés, décrits colonne par colonne.

Le notebook [`panorama_ia_ml.ipynb`](../../panorama_ia_ml.ipynb), à la racine du dépôt, répond avec ces mêmes données à quatre questions sur l'IA : depuis quand, où, quels modèles, quels effets.

Tous les fichiers se lisent directement depuis Google Colab, sans rien télécharger :

```python
import pandas as pd
BASE_URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/sources/"
df = pd.read_csv(BASE_URL + "epoch_modeles.csv")
```

---

## Atelier : explorer, nettoyer et visualiser des données avec une IA

### Partir de la question, pas de l'outil

On ne nettoie pas et on ne trace pas des graphiques « parce qu'il faut le faire ». Chaque étape répond à la problématique que l'on cherche à résoudre, par exemple : *comment le nombre de modèles d'IA a-t-il évolué, et dans quels domaines ?* C'est cette question qui décide :
- de ce qu'on regarde dans les données ;
- de ce qui compte comme un problème à corriger ;
- du graphique qui permet d'y répondre.

Les prompts ci-dessous suivent cette logique. Ils servent la question ; ce n'est pas la question qui s'adapte à ce que l'IA sait faire. Avant de copier un prompt, écrivez donc en une phrase ce que vous cherchez à savoir. Relisez chaque réponse de l'IA avec cette question en tête, et gardez votre esprit critique : l'IA propose, c'est vous qui décidez.

Méthode : ouvrez un notebook Google Colab et une conversation avec un assistant IA. Copiez le prompt en remplaçant ce qui est entre crochets. Exécutez chaque cellule proposée et recollez le texte affiché dans la conversation.

### Prompt 1 : faire connaissance avec les données

```text
Tu m'aides à faire connaissance avec un jeu de données dans Google Colab.
Je débute complètement : parle-moi simplement, sans jargon, comme à quelqu'un qui
n'a jamais fait de statistiques ni de code.

Mes données : [URL ou nom du fichier CSV]
Ce que j'en sais : [une phrase : d'où elles viennent, de quoi elles parlent]

Comment on travaille :
- Tu n'as pas accès à mes données. Tu me donnes une seule cellule de code à la fois,
  je l'exécute dans Colab et je te colle le texte qui s'affiche.
- Chaque cellule doit produire deux choses : un graphique clair pour moi (titre et
  axes en français) et un court résumé en texte, de 15 lignes maximum, que je te
  recopie. Tu ne vois pas les graphiques, seulement ce texte.
- Après chaque résultat, tu me dis en 2 phrases ce qu'il faut retenir, puis tu passes
  à l'étape suivante. Pas de détour, pas d'étape supplémentaire.
- Si je colle une erreur, tu corriges la cellule sans longue explication.
- Tu ne devines jamais le nom d'une colonne : tu utilises seulement ce que je t'ai collé.

Les 3 étapes :
1. Le premier coup d'œil : combien de lignes et de colonnes, un aperçu du tableau, et
   tu m'expliques avec tes mots ce que représente une ligne.
2. La carte des trous : un graphique en barres du pourcentage de cases vides par
   colonne, pour voir d'un coup d'œil où il manque des données.
3. Le portrait des colonnes : une planche de petits graphiques, un par colonne
   importante (répartition des valeurs pour les nombres, catégories les plus
   fréquentes pour le texte). Signale-moi tout ce qui a l'air bizarre, et dis-moi si
   le tableau mélange des choses qu'il ne faut pas additionner entre elles.

Puis le bilan, sans code :
- Ce qu'il faut nettoyer avant d'analyser : une liste courte, classée par priorité.
- 3 questions intéressantes qu'on peut se poser avec ces données.

Commence par l'étape 1.
```

Pour `epoch_modeles.csv`, on peut remplir les crochets ainsi :
- *Mes données :* `https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/sources/epoch_modeles.csv`
- *Ce que j'en sais :* « la liste des modèles d'IA publiés depuis 1950, recensés par l'institut de recherche Epoch AI ».

### Prompt 2 : nettoyer les données

```text
On passe au nettoyage, en suivant ta liste « ce qu'il faut nettoyer », dans l'ordre
des priorités. Mêmes règles qu'avant : une cellule à la fois, je te colle le texte
affiché, tu me dis en 2 phrases ce qu'il faut retenir.

Avant de commencer :
- S'il y a un choix que tu ne peux pas trancher sans moi (par exemple : une case vide
  veut-elle dire zéro ou « pas mesuré » ?), pose-moi la question en mots simples, avec
  ta recommandation. Une question par doute, pas plus.
- Ne modifie jamais le tableau d'origine : travaille sur une copie.

Le nettoyage, en 3 cellules maximum :
- Regroupe les actions qui vont ensemble plutôt que d'en faire une par cellule.
- Chaque cellule affiche un « avant / après » court : nombre de lignes, de colonnes
  et pourcentage de cases vides.
- Si une action est seulement cosmétique, fais-la sans commentaire.

Pour finir :
1. Une cellule de vérification visuelle : un graphique simple par tableau nettoyé,
   pour voir d'un coup d'œil que les données ont l'air correctes.
2. Une cellule qui enregistre chaque tableau nettoyé dans un fichier CSV au nom clair.
3. Sans code : le journal du nettoyage (ce qu'on a fait et pourquoi, en 5 lignes
   maximum), et ce qu'on a choisi de ne pas corriger.

Commence par tes questions s'il y en a, sinon par la première cellule.
```


### Les graphiques : une cellule par graphique

Chaque bloc ci-dessous est une cellule Colab complète, à copier-coller telle quelle. Elle importe les bibliothèques, charge le CSV nettoyé et trace le graphique, donc elle fonctionne même toute seule. Toutes utilisent `epoch_modeles_propre.csv`.

Chaque graphique répond à une question différente. On choisit d'abord la question, ensuite la forme :

| Graphique | Il répond à la question… |
|---|---|
| Nuage de points | Deux nombres sont-ils liés ? |
| Lignes | Comment plusieurs groupes évoluent-ils, les uns par rapport aux autres ? |
| Histogramme | Comment les valeurs d'un nombre se répartissent-elles ? |
| Camembert | Comment un tout se partage-t-il en quelques parts (3 à 5 au maximum) ? |
| Série temporelle | Comment une quantité évolue-t-elle, mois après mois ? |
| Carte | Où cela se passe-t-il ? |
| Barres | Quelles catégories sont les plus grandes ? |

#### 1. Nuage de points : la puissance de calcul selon la date

```python
import pandas as pd
import matplotlib.pyplot as plt

URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/sources_propres/epoch_modeles_propre.csv"
df = pd.read_csv(URL, parse_dates=["date_publication"])

fig, ax = plt.subplots(figsize=(9, 4.5))
for notable, couleur, nom in [(False, "#1baf7a", "Autres modèles"), (True, "#2a78d6", "Modèles « notables »")]:
    d = df[df["notable"] == notable]
    ax.scatter(d["date_publication"], d["calcul_entrainement_flop"], s=12, alpha=0.6,
               color=couleur, label=nom, edgecolors="none")
ax.set_yscale("log")  # échelle logarithmique : chaque graduation = 10 fois plus
ax.set_title("La puissance de calcul pour entraîner un modèle a explosé")
ax.set_xlabel("Date de publication")
ax.set_ylabel("Calcul d'entraînement (opérations)")
ax.grid(alpha=0.3)
ax.legend()
plt.show()
```

#### 2. Lignes : les modèles publiés chaque année dans les 4 principaux domaines

```python
import pandas as pd
import matplotlib.pyplot as plt

URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/sources_propres/epoch_modeles_propre.csv"
df = pd.read_csv(URL)

complet = df[~df["annee_partielle"]]  # 2026 n'est pas terminée : on l'exclut
top4 = complet["domaine"].value_counts().head(4).index
par_an = (complet[complet["domaine"].isin(top4) & (complet["annee"] >= 2012)]
          .groupby(["annee", "domaine"]).size().unstack(fill_value=0))

fig, ax = plt.subplots(figsize=(9, 4.5))
for domaine, couleur in zip(top4, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]):
    ax.plot(par_an.index, par_an[domaine], color=couleur, linewidth=2, marker="o",
            markersize=4, label=domaine)
ax.set_title("Le langage (texte) est en tête depuis 2017")
ax.set_xlabel("Année de publication")
ax.set_ylabel("Nombre de modèles publiés")
ax.grid(alpha=0.3)
ax.legend()
plt.show()
```

#### 3. Histogramme : la taille des modèles

```python
import pandas as pd
import matplotlib.pyplot as plt

URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/sources_propres/epoch_modeles_propre.csv"
df = pd.read_csv(URL)

classes = [10 ** (i / 4) for i in range(4, 61)]  # classes de 10 à 10^15, régulières en échelle log
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(df["nb_parametres"].dropna(), bins=classes, color="#2a78d6", edgecolor="white")
ax.set_xscale("log")
ax.set_title("La plupart des modèles : entre 1 million et 100 milliards de paramètres")
ax.set_xlabel("Nombre de paramètres (taille du modèle)")
ax.set_ylabel("Nombre de modèles")
ax.grid(alpha=0.3)
plt.show()
```

#### 4. Camembert : l'accès aux modèles publiés depuis 2023

```python
import pandas as pd
import matplotlib.pyplot as plt

URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/sources_propres/epoch_modeles_propre.csv"
df = pd.read_csv(URL)

acces = df[df["annee"] >= 2023]["acces"].value_counts()  # accès inconnu (case vide) exclu
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(acces, labels=acces.index, autopct="%1.0f %%", startangle=90, counterclock=False,
       colors=["#2a78d6", "#eb6834", "#1baf7a"], wedgeprops={"edgecolor": "white", "linewidth": 2})
ax.set_title(f"Depuis 2023, comment accède-t-on aux modèles ?\n({acces.sum()} modèles dont l'accès est connu)")
plt.show()
```

#### 5. Série temporelle : les modèles publiés chaque mois

```python
import pandas as pd
import matplotlib.pyplot as plt

URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/sources_propres/epoch_modeles_propre.csv"
df = pd.read_csv(URL, parse_dates=["date_publication"])

par_mois = (df.set_index("date_publication").resample("MS").size()
            .loc["2015-01":"2026-10"])  # novembre 2026 : mois incomplet, exclu
moyenne = par_mois.rolling(12).mean()  # lisse les variations d'un mois à l'autre
# Attention : la baisse récente peut venir en partie d'un recensement pas encore à jour

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(par_mois.index, par_mois, width=25, color="#2a78d6", alpha=0.35, label="Chaque mois")
ax.plot(moyenne.index, moyenne, color="#2a78d6", linewidth=2, label="Moyenne sur 12 mois")
ax.set_title("Les publications ont culminé fin 2024, puis baissé")
ax.set_xlabel("Mois de publication")
ax.set_ylabel("Nombre de modèles publiés")
ax.grid(alpha=0.3)
ax.legend()
plt.show()
```

#### 6. Carte : le nombre de modèles par pays

```python
import pandas as pd
import plotly.express as px

URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/sources_propres/epoch_modeles_propre.csv"
df = pd.read_csv(URL)

par_pays = (df.dropna(subset=["code_pays"]).groupby(["code_pays", "pays"]).size()
            .rename("nb_modeles").reset_index())
# des classes plutôt qu'un dégradé continu : sinon les États-Unis écrasent tout le reste
ordre = ["1 à 9", "10 à 49", "50 à 199", "200 ou plus"]
par_pays["classe"] = pd.cut(par_pays["nb_modeles"], [0, 9, 49, 199, 10_000], labels=ordre).astype(str)

fig = px.choropleth(par_pays, locations="code_pays", color="classe", hover_name="pays",
                    hover_data={"nb_modeles": True, "classe": False, "code_pays": False},
                    category_orders={"classe": ordre},
                    color_discrete_sequence=["#cde2fb", "#86b6ef", "#2a78d6", "#104281"],
                    labels={"classe": "Nombre de modèles", "nb_modeles": "Modèles"},
                    title="Où sont créés les modèles d'IA ? (pays de l'organisation, 1950-2026)")
fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth",
                projection_rotation_lon=0, center_lon=0, resolution=50)
fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
fig.show()
```

#### 7. Barres : les 10 domaines qui comptent le plus de modèles

```python
import pandas as pd
import matplotlib.pyplot as plt

URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/sources_propres/epoch_modeles_propre.csv"
df = pd.read_csv(URL)

top10 = df["domaine"].value_counts().head(10).sort_values()
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.barh(top10.index, top10, color="#2a78d6")
for y, v in enumerate(top10):
    ax.text(v + 15, y, str(v), va="center", fontsize=9)
ax.set_title("Le langage domine : un modèle sur deux")
ax.set_xlabel("Nombre de modèles (domaine principal)")
ax.grid(axis="x", alpha=0.3)
plt.show()
```

---

## Les jeux de données (`data/sources/`)

Chaque fichier contient les **données brutes d'une source**, à raison d'une ligne par observation. Seules les colonnes utiles ont été gardées et renommées en français. Les valeurs, elles, n'ont été ni traduites ni recodées. Données téléchargées le 12/11/2026.

### 1. `epoch_modeles.csv` : les modèles d'IA publiés depuis 1950

**Source :** [Epoch AI, *AI Models*](https://epoch.ai/data/ai-models), licence CC BY 4.0.
**Une ligne = un modèle d'IA** : 3 708 modèles publiés entre 1950 et 2026.

| Colonne | Contenu | Exemple |
|---|---|---|
| `modele` | Nom du modèle | `GPT-4` |
| `organisation` | Organisation(s) qui l'a (ont) créé | `OpenAI` |
| `pays_organisation` | Pays de l'organisation | `United States of America` |
| `type_organisation` | Entreprise (`Industry`) ou université (`Academia`) | `Industry` |
| `date_publication` | Date de publication (AAAA-MM-JJ) | `2023-03-15` |
| `annee` | Année de publication, tirée de la date | `2023` |
| `domaine` | Grand domaine d'application | `Language`, `Vision`, `Biology`… |
| `tache` | Tâche précise | `Chat`, `Image classification`… |
| `nb_parametres` | Taille du modèle : nombre de paramètres | `1.75e+11` |
| `calcul_entrainement_flop` | Puissance de calcul utilisée pour l'entraîner (en opérations) | `2.1e+25` |
| `acces` | Modèle ouvert, accessible par API ou non publié | `API access` |
| `notable` | `True` si Epoch le classe parmi les modèles « notables » (1 092 modèles) | `True` |

À savoir :
- Une même case peut contenir plusieurs valeurs séparées par des virgules. C'est le cas, par exemple, de plusieurs pays ou de plusieurs domaines pour un même modèle.
- La taille et le calcul sont souvent inconnus (cases vides).
- La base recense ce qui a été publié et repéré par Epoch : elle n'est pas exhaustive, surtout pour les modèles anciens et les années les plus récentes.

### 2. `eurostat_entreprises_ia.csv` (+ `eurostat_indicateurs.csv`) : l'IA dans les entreprises européennes

**Source :** [Eurostat, table `isoc_eb_ain2`](https://ec.europa.eu/eurostat/databrowser/view/isoc_eb_ain2/default/table), réutilisation libre avec mention de la source.
**Une ligne = un chiffre pour un pays × un secteur × un indicateur × une année** (226 077 lignes, années 2021, 2023, 2024 et 2025).

| Colonne | Contenu |
|---|---|
| `code_pays`, `pays` | Pays européen, ou agrégat (`EU27_2020` = Union européenne) |
| `code_secteur`, `secteur` | Secteur d'activité (nomenclature NACE) |
| `code_indicateur` | Code de l'indicateur. Son intitulé complet est dans `eurostat_indicateurs.csv` |
| `annee` | Année de l'enquête |
| `pct_entreprises` | % des entreprises de 10 salariés ou plus concernées |

Le tableau mélange des pays et des agrégats (UE), des secteurs et des regroupements de secteurs : il ne faut pas les additionner entre eux.

### 3. `kaggle_repondants.csv` : les praticiens de la data science

**Source :** [Kaggle Machine Learning & Data Science Survey](https://www.kaggle.com/kaggle-survey-2022), éditions 2017 à 2022, données publiées par Kaggle sous licence de compétition ou de jeu de données Kaggle.
**Une ligne = un répondant à l'enquête annuelle** (130 298 répondants).

| Colonne | Contenu |
|---|---|
| `annee` | Édition de l'enquête |
| `id_repondant` | Numéro du répondant dans l'édition |
| `pays` | Pays de résidence |
| `age` | Âge : nombre exact en 2017, tranche d'âge (`25-29`…) ensuite |
| `metier` | Intitulé de poste |
| `secteur` | Secteur de l'employeur. Question absente en 2019 et 2020 |
| `algorithmes` | Algorithmes utilisés régulièrement, séparés par « ; ». Question absente en 2018 |

Les questions et les choix de réponse changent d'une année à l'autre : il faut harmoniser avant de comparer.

### 4. `owid_incidents.csv` : les incidents liés à l'IA

**Source :** [Our World in Data](https://ourworldindata.org/artificial-intelligence), d'après l'AI Incident Database (via l'AI Index), licence CC BY 4.0.
**Une ligne = une année** (2012 à 2024) : nombre d'incidents et de controverses liés à l'IA signalés dans le monde. Les incidents sont recensés à partir d'articles de presse, donc probablement sous-déclarés.

### 5. `vinuesa_cibles_odd.csv` : l'IA face aux objectifs de développement durable

**Source :** Vinuesa et al. (2020), [« The role of artificial intelligence in achieving the Sustainable Development Goals »](https://doi.org/10.1038/s41467-019-14108-y), *Nature Communications*, Supplementary Data 1, licence CC BY 4.0.
**Une ligne = une des 169 cibles des 17 ODD de l'ONU.**

| Colonne | Contenu |
|---|---|
| `odd`, `objectif` | Numéro et intitulé de l'objectif |
| `cible`, `texte` | Numéro et texte de la cible (ex. `13.2`) |
| `levier` | 1 si des publications montrent que l'IA peut **aider** à atteindre la cible |
| `frein` | 1 si des publications montrent que l'IA peut **nuire** à la cible |
| `justification` | Raisonnement des auteurs |

Une cible peut être à la fois levier et frein. Il s'agit d'un avis d'experts fondé sur la littérature, pas d'une mesure.

---

## Les jeux de données nettoyés (`data/sources_propres/`)

Chaque fichier brut a un équivalent nettoyé, prêt pour la visualisation : même nom, suivi de `_propre`. Une ligne représente toujours la même chose que dans le fichier brut.

Ce que le nettoyage a fait :
- libellés traduits en français et harmonisés ;
- cases à valeurs multiples dédoublonnées ;
- code pays ISO-3 ajouté pour les cartes ;
- colonnes oui/non transformées en vrai/faux.

Rien n'a été inventé : une valeur inconnue reste vide.

```python
BASE_URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/sources_propres/"
df = pd.read_csv(BASE_URL + "epoch_modeles_propre.csv", parse_dates=["date_publication"])
```

### 1. `epoch_modeles_propre.csv` : 3 708 modèles, 19 colonnes

| Colonne | Contenu |
|---|---|
| `modele`, `organisation`, `tache` | Comme dans le fichier brut |
| `pays`, `code_pays` | Premier pays cité, en français, et son code ISO-3 (vide pour « Multinationale ») |
| `nb_pays`, `pays_tous` | Nombre de pays différents et liste complète (séparés par « ; ») |
| `type_organisation` | `Entreprise`, `Université`, `Organisme public`, `Collectif de recherche` ou `Plusieurs types` |
| `date_publication`, `annee`, `mois` | Date de publication, puis l'année et le mois tirés de cette date |
| `domaine` | Premier domaine cité, en français (`Langage (texte)`, `Vision (images)`…) |
| `nb_domaines`, `domaines_tous` | Nombre de domaines et liste complète |
| `nb_parametres`, `calcul_entrainement_flop` | Comme dans le fichier brut |
| `acces` | En 3 catégories : `Ouvert (poids téléchargeables)`, `Accessible en ligne (API ou service)`, `Non publié ou accès limité` |
| `notable` | `True` si Epoch le classe parmi les modèles notables |
| `annee_partielle` | `True` pour 2026 : l'année n'est pas terminée, il faut l'exclure des comparaisons annuelles |

### 2. `eurostat_entreprises_ia_propre.csv` : 20 051 lignes

Seuls sont gardés les 34 pays européens, l'UE à 27, les 12 secteurs principaux et 17 indicateurs traduits.

| Colonne | Contenu |
|---|---|
| `code_pays`, `pays` | Code ISO-3 et nom en français. Pour `UE à 27`, `code_pays` est vide |
| `est_pays` | `False` pour l'agrégat UE à 27, à ne pas additionner avec les pays |
| `secteur` | Secteur en français. `Ensemble des secteurs couverts` est un total |
| `theme` | `Usage` (utilise au moins une IA), `Technologie` (type d'IA utilisé) ou `Frein` (raison de ne pas l'utiliser) |
| `indicateur` | Intitulé en français |
| `annee`, `pct_entreprises` | Année et % des entreprises de 10 salariés ou plus |

### 3. `kaggle_repondants_propre.csv` : 130 298 répondants, 21 colonnes

| Colonne | Contenu |
|---|---|
| `annee`, `id_repondant` | Comme dans le fichier brut |
| `pays`, `code_pays` | Pays harmonisé en français (ex. « United States » et « United States of America » donnent « États-Unis ») et code ISO-3 |
| `tranche_age` | Mêmes tranches toutes les années (en 2017, l'âge exact a été converti en tranche) |
| `metier` | Regroupé en 13 familles (`Data scientist`, `Études`, `Recherche`…) |
| `secteur` | Secteur harmonisé en français (vide en 2019 et 2020 : question absente) |
| `algorithmes`, `nb_algorithmes` | Familles d'algorithmes utilisées, harmonisées en français, et leur nombre |
| `algo_regression` … `algo_graphes` | Une colonne vrai/faux par famille. Vide si la personne n'a pas répondu, ou en 2018 (question absente). La moyenne d'une colonne donne directement le % de répondants |

### 4. `owid_incidents_propre.csv` : 13 années

`annee`, `nb_incidents` et `nb_incidents_cumules` (total depuis 2012).

### 5. `vinuesa_cibles_odd_propre.csv` : 169 cibles

`odd`, `objectif` (en français), `cible`, `texte`, `levier` et `frein` (vrai/faux), et `effet`, qui résume les deux : `Levier seulement`, `Frein seulement`, `Levier et frein` ou `Aucun effet documenté`.

### Les autres fichiers

- `data/clean/panorama_ia.csv` et `panorama_ia_large.csv` : chiffres déjà agrégés utilisés par le notebook `panorama_ia_ml.ipynb`.
- Les scripts s'enchaînent ainsi :
  - `download_data.py` télécharge les données brutes dans `data/raw/` (non versionné) ;
  - `make_sources.py` produit `data/sources/` ;
  - `clean_sources.py` produit `data/sources_propres/` ;
  - `prepare_data.py` produit `data/clean/`.
