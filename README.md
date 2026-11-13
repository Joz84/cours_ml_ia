# Panorama de l'IA et du machine learning

Données et supports du cours « Data Science et IA » (licence).

Tous les fichiers se lisent directement depuis Google Colab, sans rien télécharger :

```python
import pandas as pd
BASE_URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/sources/"
df = pd.read_csv(BASE_URL + "epoch_modeles.csv")
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

### Les autres fichiers

- `data/clean/panorama_ia.csv` et `panorama_ia_large.csv` : chiffres déjà agrégés utilisés par le notebook `panorama_ia_ml.ipynb`.
- `download_data.py` télécharge les données brutes dans `data/raw/` (non versionné). Ensuite, `make_sources.py` produit `data/sources/` et `prepare_data.py` produit `data/clean/`.

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
