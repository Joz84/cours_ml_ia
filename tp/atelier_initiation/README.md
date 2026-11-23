# Atelier 0 · L'électricité de Tahiti

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Joz84/cours_ml_ia/blob/main/tp/atelier_initiation/Atelier_initiation_ELEVE.ipynb)

UE 5.7 « Data Science et IA » · Licence STES et SPI · Université de la Polynésie française

Quand vous allumez la climatisation à Papeete, d'où vient l'électricité ? D'un moteur qui brûle du fioul, d'un barrage, ou du soleil ? Et à quelle heure Tahiti consomme-t-il le plus ? Dans cet atelier, vous allez répondre à ces questions **avec de vraies données** : la production d'électricité de Tahiti, **heure par heure, de 2015 à 2025**. Cela fait 96 432 lignes, trop pour les lire à l'œil, mais pas pour Python.

- **Durée :** environ 1 h 30, en binôme, dans **Google Colab**. Vous n'avez rien à installer.
- **Aucune connaissance en programmation n'est demandée.** Chaque cellule est donnée : vous la lancez, vous lisez le résultat, et parfois vous changez **un seul mot**.
- **Le jeu :** chaque mission commence par un **QCM**. Vous **pariez** sur la réponse *avant* de regarder les données, puis le graphique vous dit si vous aviez raison. La cellule `verifier` corrige et compte les points. Score final sur 9 !

## Comment ça marche

1. Ouvrez le carnet avec le bouton « Ouvrir dans Colab » ci-dessus.
2. Exécutez les cellules **dans l'ordre**, avec **Maj + Entrée**.
3. À chaque QCM, **écrivez votre lettre** entre les guillemets de `verifier(1, "")`, par exemple `verifier(1, "B")`, puis exécutez.
4. Quand une cellule dit **À VOUS**, changez le mot indiqué et relancez-la.
5. Si une cellule plante, copiez-la avec le message d'erreur dans ChatGPT ou Claude, et demandez : « Explique-moi cette erreur simplement. »

Deux outils, que vous retrouverez dans tout le cours :
- **pandas** range les données dans un tableau (un *DataFrame*, comme une feuille Excel) et sait le trier, le filtrer, le résumer ;
- **matplotlib** dessine les graphiques.

## Les données

Le tableau `electricite_tahiti.csv` donne, **pour chaque heure** de 2015 à 2025, la puissance électrique moyenne produite à Tahiti par chaque source, en **mégawatts (MW)**. Une puissance de 60 MW pendant une heure fournit une énergie de 60 mégawattheures (MWh).

| Colonne | Contenu |
|---|---|
| `date_heure` | le début de l'heure, par exemple `2025-10-08 11:00:00` |
| `annee`, `mois`, `jour`, `heure` | la même date, découpée |
| `jour_semaine` | `lundi`, `mardi`… |
| `thermique_MW` | les centrales thermiques : des moteurs qui brûlent du fioul |
| `hydraulique_MW` | les barrages et les centrales sur les rivières |
| `solaire_MW` | les panneaux solaires, raccordés au réseau |
| `total_MW` | la somme des trois : ce que Tahiti consomme à cette heure-là |

**Source :** Service des énergies de la Polynésie française, [« Production d'électricité sur Tahiti »](https://www.service-public.pf/sde/production-delectricite-sur-tahiti-2/) (mesures EDT, toutes les 10 minutes, par source), 2015 à 2025. Données utilisées à des fins pédagogiques dans le cadre du partenariat entre l'Université de la Polynésie française et le Service des énergies. Le script [`scripts/prepare_energie_tahiti.py`](../../scripts/prepare_energie_tahiti.py) harmonise les 11 fichiers annuels (noms de colonnes, kW ou MW) et calcule les moyennes par heure. Quelques heures sont vides, quand l'enregistrement a manqué : rien n'a été inventé.

---

## Le carnet

### Préparer · Les outils et les données

La cellule importe pandas et matplotlib, récupère le petit module de QCM, et charge le tableau dans une variable nommée `df` (pour *DataFrame*).

```python
import os, urllib.request
import pandas as pd
import matplotlib.pyplot as plt

if not os.path.exists("quiz.py"):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/tp/atelier_initiation/quiz.py", "quiz.py")
from quiz import verifier, score

URL = "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/data/clean/electricite_tahiti.csv"
df = pd.read_csv(URL, parse_dates=["date_heure"])   # parse_dates : la colonne date_heure est lue comme une date
print("Données chargées !")
```

**Résultat attendu**

```text
Données chargées !
```

### Mission 1 · Faire connaissance avec le tableau

**QCM 1. Combien de lignes compte le tableau ?**
A. 8 760 · B. 96 432 · C. 1 million · D. 11

Pariez d'abord ! Puis exécutez : `df.head()` montre les 5 premières lignes, `df.shape` donne le nombre de lignes et de colonnes.

```python
print("Lignes, colonnes :", df.shape)
df.head()
```

**Résultat attendu**

```text
Lignes, colonnes : (96432, 10)
           date_heure  annee  mois  jour  heure jour_semaine  thermique_MW  hydraulique_MW  solaire_MW  total_MW
0 2015-01-01 00:00:00   2015     1     1      0        jeudi         44.12            9.94         0.0     54.06
1 2015-01-01 01:00:00   2015     1     1      1        jeudi         41.98            9.91         0.0     51.89
…
```

<!-- id: q1 -->
```python
verifier(1, "")   # votre lettre entre les guillemets
```

Et un résumé de chaque colonne en une ligne : `describe()` donne le nombre de valeurs (`count`), la moyenne (`mean`), le minimum (`min`) et le maximum (`max`).

```python
df.describe().round(1)
```

**À retenir.** `df.head()`, `df.shape` et `df.describe()` : les trois premiers réflexes devant n'importe quel tableau.

### Mission 2 · Une semaine dans la vie du réseau

**QCM 2. À quelle heure de la journée Tahiti consomme-t-il le plus d'électricité ?**
A. 3 h du matin · B. vers 11 h · C. vers 20 h · D. à minuit

On **filtre** le tableau pour ne garder qu'une semaine, puis on trace la colonne `total_MW` en fonction du temps : c'est une **courbe**.

```python
semaine = df[(df["date_heure"] >= "2024-03-04") & (df["date_heure"] < "2024-03-11")]   # du lundi 4 au dimanche 10 mars 2024

plt.figure(figsize=(12, 4))
plt.plot(semaine["date_heure"], semaine["total_MW"])
plt.title("Une semaine d'électricité à Tahiti (mars 2024)")
plt.ylabel("puissance (MW)")
plt.show()
```

**Résultat attendu :** une courbe qui monte le jour et descend la nuit, sept fois, avec deux jours plus bas à la fin : le week-end.

**Le saviez-vous ?** Le mardi 5 mars est aussi bas qu'un dimanche. Pourquoi ? (Indice : regardez le calendrier des jours fériés de Polynésie.)

Une semaine, c'est peu. Faisons la **moyenne de chaque heure sur les 11 ans** : `groupby("heure")` regroupe les lignes par heure, `.mean()` fait la moyenne de chaque groupe.

```python
par_heure = df.groupby("heure")["total_MW"].mean()

plt.figure(figsize=(8, 4))
plt.plot(par_heure.index, par_heure.values, marker="o")
plt.title("Une journée moyenne à Tahiti, 2015-2025")
plt.xlabel("heure"); plt.ylabel("puissance moyenne (MW)")
plt.xticks(range(0, 24, 2))
plt.show()
```

**Résultat attendu :** une bosse en milieu de journée, un creux en pleine nuit.

<!-- id: q2 -->
```python
verifier(2, "")
```

**À VOUS.** Dans la cellule ci-dessous, remplacez `"total_MW"` par `"solaire_MW"`, puis par `"hydraulique_MW"`, et relancez à chaque fois. Le soleil, sans surprise, ne produit que le jour. Et l'eau ?

<!-- id: a_vous_colonne -->
```python
colonne = "total_MW"   # À VOUS : essayez "solaire_MW", puis "hydraulique_MW", puis "thermique_MW"

plt.figure(figsize=(8, 4))
plt.plot(df.groupby("heure")[colonne].mean(), marker="o")
plt.title(f"Une journée moyenne : {colonne}")
plt.xlabel("heure"); plt.ylabel("puissance moyenne (MW)")
plt.show()
```

### Mission 3 · Qui produit l'électricité de Tahiti ?

**QCM 3. Sur 2015-2025, quelle source a produit le plus d'électricité ?**
A. le soleil · B. l'eau (hydraulique) · C. le fioul (thermique) · D. le vent

On additionne chaque colonne avec `.sum()` : la somme des puissances heure par heure donne l'énergie en MWh, et on divise par 1 000 pour avoir des gigawattheures (GWh). Puis un graphique en **barres**.

```python
sources = ["thermique_MW", "hydraulique_MW", "solaire_MW"]
energie = df[sources].sum() / 1000          # MWh -> GWh
print(energie.round(0))

plt.figure(figsize=(6, 4))
plt.bar(["thermique", "hydraulique", "solaire"], energie, color=["#7f7f7f", "#2a78d6", "#f2b701"])
plt.title("Électricité produite à Tahiti, 2015-2025")
plt.ylabel("énergie (GWh)")
plt.show()
```

**Résultat attendu**

```text
thermique_MW      3595.0
hydraulique_MW    1905.0
solaire_MW         456.0
```

<!-- id: q3 -->
```python
verifier(3, "")
```

### Mission 4 · Le jour du record solaire

**QCM 4. Le 8 octobre 2025 à 11 h, le solaire bat son record. Quelle part de l'électricité de Tahiti vient alors du soleil ?**
A. 5 % · B. 25 % · C. 66 % · D. 100 %

On garde une seule journée, et on empile les trois sources avec `stackplot` : la hauteur totale est la consommation, chaque couleur montre qui la fournit.

```python
jour = df[(df["date_heure"] >= "2025-10-08") & (df["date_heure"] < "2025-10-09")]

plt.figure(figsize=(10, 4.5))
plt.stackplot(jour["heure"], jour["thermique_MW"], jour["hydraulique_MW"], jour["solaire_MW"],
              labels=["thermique", "hydraulique", "solaire"], colors=["#7f7f7f", "#2a78d6", "#f2b701"])
plt.title("Le 8 octobre 2025, heure par heure")
plt.xlabel("heure"); plt.ylabel("puissance (MW)")
plt.legend(loc="upper left")
plt.show()

midi = jour[jour["heure"] == 11]
print("À 11 h : solaire", midi["solaire_MW"].values[0], "MW sur", midi["total_MW"].values[0], "MW")
```

**Résultat attendu**

```text
À 11 h : solaire 55.37 MW sur 83.99 MW
```

<!-- id: q4 -->
```python
verifier(4, "")
```

### Mission 5 · Le solaire décolle

**QCM 5. Combien de fois plus d'électricité solaire Tahiti a-t-il produit en 2025 qu'en 2015 ?**
A. 1,5 fois · B. 2 fois · C. 3,7 fois · D. 10 fois

On regroupe par **année** et on additionne.

```python
solaire_par_an = df.groupby("annee")["solaire_MW"].sum() / 1000     # GWh par an
print(solaire_par_an.round(1))

plt.figure(figsize=(9, 4))
plt.bar(solaire_par_an.index, solaire_par_an.values, color="#f2b701")
plt.title("Électricité solaire produite à Tahiti chaque année")
plt.ylabel("énergie (GWh)")
plt.xticks(solaire_par_an.index)
plt.show()
```

**Résultat attendu**

```text
annee
2015    26.7
2016    27.3
…
2024    54.2
2025    99.2
```

<!-- id: q5 -->
```python
verifier(5, "")
```

**À VOUS.** Et la part de l'électricité qui vient du renouvelable (eau + soleil), année par année ? La cellule est prête : lancez-la, et regardez 2025.

```python
par_an = df.groupby("annee")[sources + ["total_MW"]].sum()
part_renouvelable = 100 * (par_an["hydraulique_MW"] + par_an["solaire_MW"]) / par_an["total_MW"]

plt.figure(figsize=(9, 4))
plt.plot(part_renouvelable.index, part_renouvelable.values, marker="o", color="#2a9d4b")
plt.title("Part du renouvelable (eau + soleil) dans l'électricité de Tahiti")
plt.ylabel("%"); plt.ylim(0, 100)
plt.xticks(part_renouvelable.index)
plt.show()
```

### Mission 6 · La pluie et les barrages

**QCM 6. En quelle saison les barrages produisent-ils le plus ?**
A. en saison des pluies (novembre à février) · B. en saison sèche (juin à septembre) · C. pareil toute l'année

```python
hydro_par_mois = df.groupby("mois")["hydraulique_MW"].mean()

plt.figure(figsize=(9, 4))
plt.bar(hydro_par_mois.index, hydro_par_mois.values, color="#2a78d6")
plt.title("Production hydraulique moyenne, mois par mois (2015-2025)")
plt.xlabel("mois"); plt.ylabel("puissance moyenne (MW)")
plt.xticks(range(1, 13), ["jan", "fév", "mar", "avr", "mai", "juin", "juil", "août", "sep", "oct", "nov", "déc"])
plt.show()
```

<!-- id: q6 -->
```python
verifier(6, "")
```

**Pour aller plus loin.** Quand les barrages produisent moins, qui compense ? Reprenez la cellule ci-dessus en remplaçant `"hydraulique_MW"` par `"thermique_MW"`.

### Mission 7 · Le dimanche à Tahiti

**QCM 7. Le dimanche, par rapport à un jour de semaine, Tahiti consomme…**
A. plus · B. autant · C. environ 17 % de moins · D. moitié moins

```python
ordre = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
par_jour = df.groupby("jour_semaine")["total_MW"].mean().reindex(ordre)   # reindex : dans l'ordre de la semaine
print(par_jour.round(1))

plt.figure(figsize=(8, 4))
plt.bar(par_jour.index, par_jour.values, color=["#2a78d6"] * 5 + ["#eb6834"] * 2)
plt.title("Consommation moyenne selon le jour de la semaine")
plt.ylabel("puissance moyenne (MW)")
plt.show()
```

**Résultat attendu**

```text
jour_semaine
lundi       64.1
mardi       65.3
mercredi    65.0
jeudi       65.0
vendredi    63.7
samedi      56.5
dimanche    53.8
```

<!-- id: q7 -->
```python
verifier(7, "")
```

### Mission 8 · Le confinement de 2020

Du 21 mars au 21 mai 2020, la Polynésie a vécu son premier confinement face au Covid-19. Est-ce que ça se voit dans l'électricité ?

**QCM 8. En avril 2020, par rapport à avril 2019, la consommation d'électricité a…**
A. augmenté · B. pas bougé · C. baissé d'environ 9 % · D. baissé de moitié

On calcule la consommation moyenne de **chaque jour** (`resample("D")` regroupe les heures par jour), puis on superpose 2019 et 2020.

```python
par_jour_calendrier = df.set_index("date_heure")["total_MW"].resample("D").mean()

plt.figure(figsize=(11, 4))
for annee, couleur in [(2019, "#7f7f7f"), (2020, "#eb6834")]:
    p = par_jour_calendrier[f"{annee}-02-01":f"{annee}-06-30"]
    plt.plot(p.index.dayofyear, p.values, color=couleur, label=str(annee))
plt.axvspan(81, 142, color="#eb6834", alpha=0.1, label="confinement 2020")   # du 21 mars au 21 mai
plt.title("Consommation moyenne de chaque jour, de février à juin")
plt.xlabel("jour de l'année"); plt.ylabel("puissance moyenne (MW)")
plt.legend()
plt.show()

for annee in [2019, 2020]:
    avril = df[(df["annee"] == annee) & (df["mois"] == 4)]
    print(f"Avril {annee} : {avril['total_MW'].mean():.1f} MW en moyenne")
```

**Résultat attendu**

```text
Avril 2019 : 66.2 MW en moyenne
Avril 2020 : 60.4 MW en moyenne
```

<!-- id: q8 -->
```python
verifier(8, "")
```

### Mission 9 · L'enquête : la nuit la plus sombre

Dans 96 432 heures, une est bien plus basse que toutes les autres. `idxmin()` trouve la ligne du minimum, `loc` l'affiche.

```python
ligne = df["total_MW"].idxmin()
print(df.loc[ligne, ["date_heure", "thermique_MW", "hydraulique_MW", "solaire_MW", "total_MW"]])
```

**Résultat attendu**

```text
date_heure        2017-03-29 02:00:00
thermique_MW                     3.08
hydraulique_MW                    0.0
solaire_MW                        0.0
total_MW                         3.08
```

Regardons cette nuit-là de plus près, source par source.

```python
nuit = df[(df["date_heure"] >= "2017-03-28 18:00") & (df["date_heure"] <= "2017-03-29 10:00")]

plt.figure(figsize=(10, 4))
plt.plot(nuit["date_heure"], nuit["thermique_MW"], marker="o", label="thermique", color="#7f7f7f")
plt.plot(nuit["date_heure"], nuit["hydraulique_MW"], marker="o", label="hydraulique", color="#2a78d6")
plt.plot(nuit["date_heure"], nuit["total_MW"], marker="o", label="total", color="black")
plt.title("La nuit du 28 au 29 mars 2017")
plt.ylabel("puissance (MW)")
plt.legend()
plt.show()
```

La courbe est coupée vers 1 h : cette heure-là, rien n'a été enregistré, et la case est restée vide.

**QCM 9. D'après le graphique, que s'est-il probablement passé ?**
A. une coupure générale, puis un rétablissement progressif · B. une erreur de saisie isolée · C. tout le monde a éteint la lumière en même temps · D. une éclipse

<!-- id: q9 -->
```python
verifier(9, "")
```

### Mission 10 · À vous de jouer

Vous savez maintenant **charger** un tableau, le **filtrer**, le **regrouper** (`groupby`), le **résumer** (`mean`, `sum`) et **tracer** une courbe ou des barres. Choisissez une question, et répondez-y par un graphique. Quelques idées :
- Quel mois de l'année consomme le plus ? Le solaire produit-il plus en décembre ou en juin ?
- Le 31 décembre à minuit, la consommation change-t-elle ?
- Le solaire a-t-il fait baisser le thermique en milieu de journée, entre 2015 et 2025 ?

Partez d'une cellule qui marche plus haut, et changez une chose à la fois. Si vous êtes bloqués, demandez à ChatGPT ou à Claude : « Avec pandas, à partir d'un tableau df qui a les colonnes …, comment tracer … ? »

<!-- id: libre -->
```python
# À VOUS : votre question, votre graphique
par_mois = df.groupby("mois")["total_MW"].mean()
plt.bar(par_mois.index, par_mois.values)
plt.title("Votre titre : écrivez-y la réponse à votre question")
plt.show()
```

**Questions**
- Quelle question avez-vous posée ? Qu'avez-vous trouvé ? Écrivez-le en une phrase, avec un chiffre.

### Score · Combien de points ?

```python
score()
```

## Ce qu'il faut rendre

Rien ! Montrez votre score, et votre plus beau graphique, à l'enseignant. Vous avez manipulé 96 432 lignes de vraies données : c'est exactement ce que vous ferez pendant tout le cours, puis avec le machine learning.

## Les mots de l'atelier

| Mot | En une phrase |
|---|---|
| **DataFrame** (`df`) | Un tableau de pandas : des lignes, des colonnes, comme une feuille Excel. |
| **Colonne** | `df["total_MW"]` : une colonne du tableau. |
| **Filtrer** | `df[df["annee"] == 2025]` : ne garder que les lignes qui respectent une condition. |
| **Regrouper** | `df.groupby("heure")["total_MW"].mean()` : la moyenne de chaque groupe. |
| **MW, MWh, GWh** | La puissance (à un instant), l'énergie (la puissance pendant une durée) ; 1 GWh = 1 000 MWh. |
| **Thermique, hydraulique, solaire** | L'électricité du fioul, de l'eau des rivières, du soleil. Les deux dernières sont renouvelables. |
