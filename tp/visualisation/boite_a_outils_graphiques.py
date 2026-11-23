"""Boîte à outils graphiques du cours UE 5.7 « Data Science et IA ».

Une fonction par graphique standard. Toutes suivent les mêmes règles, celles d'un
bon graphique en 5 points :
1. un titre qui dit le message (paramètre `titre` obligatoire) ;
2. des axes nommés, avec leur unité (paramètre `unite`) ;
3. une seule idée par graphique ;
4. peu de couleurs : deux au plus, plus du gris ;
5. la source et la date, en bas à droite (paramètre `source`).

Et une règle de plus : les barres partent de zéro.

Le verbe de la question choisit le graphique :

    comparer des catégories            -> barres
    suivre dans le temps               -> courbe
    voir la répartition                -> histogramme
    chercher un lien                   -> nuage
    comparer des groupes en détail     -> boite
    montrer une part d'un tout         -> camembert (5 parts au plus, avec prudence)
    croiser deux catégories            -> barres_croisees
    voir tous les liens d'un coup      -> correlations

Si une colonne n'existe pas, la fonction s'arrête avec un message qui donne la liste
des colonnes disponibles. Chaque fonction affiche directement son graphique.

Utilisation dans Colab :

    import urllib.request
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/Joz84/cours_ml_ia/main/tp/visualisation/boite_a_outils_graphiques.py",
        "boite_a_outils_graphiques.py")
    from boite_a_outils_graphiques import *
"""
import datetime

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

__all__ = ["barres", "courbe", "histogramme", "nuage", "boite", "camembert", "barres_croisees", "correlations"]

BLEU, ORANGE, GRIS, GRIS_CLAIR = "#2a78d6", "#eb6834", "#7f7f7f", "#c4c4c4"
COULEURS = [BLEU, ORANGE, GRIS, GRIS_CLAIR]
AGREGATIONS = {"mean": "moyenne", "sum": "total", "median": "médiane", "count": "nombre", "max": "maximum", "min": "minimum"}
PAS = {"J": "D", "S": "W-MON", "M": "MS", "A": "YS"}   # jour, semaine, mois, année


# ---------------------------------------------------------------- garde-fous communs
def _verifier(df, *colonnes):
    """Arrête tout, avec un message clair, si une colonne n'existe pas."""
    for colonne in colonnes:
        if colonne is not None and colonne not in df.columns:
            raise ValueError(f"La colonne « {colonne} » n'existe pas dans le tableau. "
                             f"Colonnes disponibles : {list(df.columns)}. Vérifiez l'orthographe.")


def _verifier_titre(titre):
    if not titre:
        raise ValueError("Donnez un titre qui dit le message du graphique, par exemple : "
                         "titre=\"Vallée reçoit trois fois plus de pluie que Port\".")


def _etiquette(colonne, unite, agg=None):
    texte = colonne if agg is None else f"{colonne} ({AGREGATIONS.get(agg, agg)})"
    return f"{texte} [{unite}]" if unite else texte


def _habiller(ax, titre, source):
    """Titre, grille discrète, cadre allégé, source et date en bas à droite."""
    ax.set_title(titre, fontsize=12, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    pied = f"Source : {source} · " if source else ""
    ax.figure.text(0.99, 0.005, pied + "tracé le " + datetime.date.today().strftime("%d/%m/%Y"),
                   ha="right", va="bottom", fontsize=8, color=GRIS)
    ax.figure.tight_layout(rect=(0, 0.03, 1, 1))


def _format(v):
    return f"{v:,.0f}".replace(",", " ") if abs(v) >= 100 else f"{v:.1f}"


# ---------------------------------------------------------------- les graphiques
def barres(df, categorie, valeur, agg="mean", *, titre=None, unite="", source="", debut_axe=0):
    """Comparer des catégories : une barre par catégorie, triées de la plus grande à la plus petite.

    agg : "mean" (moyenne), "sum" (total), "median" ou "count".
    Les barres partent de zéro. `debut_axe` permet de couper l'axe, mais un avertissement s'affiche.
    Sous chaque catégorie, n rappelle le nombre de valeurs utilisées.

    Exemple :
        barres(df, "station", "pluie_mm", agg="sum", titre="Vallée est la station la plus arrosée",
               unite="mm", source="Relevés 2024")
    """
    _verifier(df, categorie, valeur)
    _verifier_titre(titre)
    valeurs = df.groupby(categorie)[valeur].agg(agg).sort_values(ascending=False)
    n = df.groupby(categorie)[valeur].count()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([f"{c}\n(n={n[c]})" for c in valeurs.index], valeurs.values, color=BLEU)
    for x, v in enumerate(valeurs.values):
        ax.text(x, v, _format(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylim(bottom=debut_axe)
    if debut_axe != 0:
        print(f"ATTENTION : l'axe commence à {debut_axe} et pas à 0. "
              "Les différences entre barres paraissent bien plus grandes qu'elles ne sont.")
    ax.set_xlabel(categorie)
    ax.set_ylabel(_etiquette(valeur, unite, agg))
    _habiller(ax, titre, source)
    plt.show()


def courbe(df, date, valeur, par=None, pas="M", agg="mean", garder=None, *, titre=None, unite="", source=""):
    """Suivre dans le temps : la valeur regroupée au pas choisi, une courbe par catégorie si `par` est donné.

    pas : "J" (jour), "S" (semaine), "M" (mois) ou "A" (année).
    garder : la liste des catégories à tracer (4 au plus), par exemple ["Vallée", "Port"].

    Exemple :
        courbe(df, "date", "pluie_mm", par="station", garder=["Vallée", "Port"], agg="sum",
               titre="Il pleut surtout de décembre à mars", unite="mm", source="Relevés 2024")
    """
    _verifier(df, date, valeur, par)
    _verifier_titre(titre)
    donnees = df.assign(**{date: pd.to_datetime(df[date])})
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if par is None:
        serie = donnees.groupby(pd.Grouper(key=date, freq=PAS.get(pas, pas)))[valeur].agg(agg)
        ax.plot(serie.index, serie.values, color=BLEU, linewidth=2, marker="o")
    else:
        categories = garder if garder is not None else list(donnees[par].dropna().unique())
        if len(categories) > 4:
            print(f"ATTENTION : {len(categories)} courbes, c'est trop pour un graphique lisible. "
                  f"Seules les 4 premières sont tracées : {categories[:4]}. Choisissez-les avec garder=[...].")
            categories = categories[:4]
        for categorie, couleur, marque in zip(categories, COULEURS, ["o", "s", "^", "D"]):
            morceau = donnees[donnees[par] == categorie]
            serie = morceau.groupby(pd.Grouper(key=date, freq=PAS.get(pas, pas)))[valeur].agg(agg)
            ax.plot(serie.index, serie.values, color=couleur, linewidth=2, marker=marque, label=str(categorie))
        ax.legend(title=par, frameon=False)
    ax.set_xlabel(date)
    ax.set_ylabel(_etiquette(valeur, unite, agg))
    _habiller(ax, titre, source)
    plt.show()


def histogramme(df, valeur, bins=20, *, titre=None, unite="", source=""):
    """Voir la répartition : l'histogramme d'une colonne, avec sa moyenne et sa médiane.

    Exemple :
        histogramme(df, "temp_max_C", titre="La plupart des journées montent entre 27 et 31 °C",
                    unite="°C", source="Relevés 2024")
    """
    _verifier(df, valeur)
    _verifier_titre(titre)
    serie = df[valeur].dropna()
    moyenne, mediane = serie.mean(), serie.median()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(serie, bins=bins, color=BLEU, edgecolor="white")
    ax.axvline(moyenne, color=ORANGE, linewidth=2, label=f"moyenne = {_format(moyenne)}")
    ax.axvline(mediane, color=GRIS, linewidth=2, linestyle="--", label=f"médiane = {_format(mediane)}")
    ax.legend(frameon=False)
    ax.set_xlabel(_etiquette(valeur, unite))
    ax.set_ylabel("Nombre de lignes")
    _habiller(ax, titre, source)
    plt.show()


def nuage(df, x, y, etiquettes=None, *, titre=None, unite=("", ""), source=""):
    """Chercher un lien : un point par ligne, et le coefficient de corrélation r dans le titre.

    unite : un couple (unité de x, unité de y).
    etiquettes : une colonne dont la valeur s'écrit à côté de chaque point (pour un petit tableau).

    Exemple :
        nuage(df, "temp_min_C", "temp_max_C", titre="Quand le minimum monte, le maximum aussi",
              unite=("°C", "°C"), source="Relevés 2024")
    """
    _verifier(df, x, y, etiquettes)
    _verifier_titre(titre)
    r = df[x].corr(df[y])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df[x], df[y], s=25 if etiquettes else 10, alpha=0.9 if etiquettes else 0.4, color=BLEU)
    if etiquettes:
        for _, ligne in df.iterrows():
            ax.annotate(str(ligne[etiquettes]), (ligne[x], ligne[y]), textcoords="offset points",
                        xytext=(5, 5), fontsize=9)
    unite_x, unite_y = unite if isinstance(unite, tuple) else (unite, unite)
    ax.set_xlabel(_etiquette(x, unite_x))
    ax.set_ylabel(_etiquette(y, unite_y))
    ax.grid(axis="x", alpha=0.3)
    _habiller(ax, f"{titre} (r = {r:.2f})", source)
    plt.show()


def boite(df, categorie, valeur, *, titre=None, unite="", source=""):
    """Comparer des groupes en détail : une boîte à moustaches par catégorie, triées par médiane.

    La boîte va du 1er au 3e quartile, le trait est la médiane, les points sont les valeurs extrêmes.

    Exemple :
        boite(df, "station", "temp_max_C", titre="Il fait plus frais en altitude",
              unite="°C", source="Relevés 2024")
    """
    _verifier(df, categorie, valeur)
    _verifier_titre(titre)
    ordre = df.groupby(categorie)[valeur].median().sort_values(ascending=False).index
    groupes = [df.loc[df[categorie] == c, valeur].dropna() for c in ordre]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(groupes, patch_artist=True, boxprops={"facecolor": BLEU, "alpha": 0.6},
               medianprops={"color": ORANGE, "linewidth": 2}, flierprops={"markersize": 3, "markeredgecolor": GRIS})
    ax.set_xticks(range(1, len(ordre) + 1), [f"{c}\n(n={len(g)})" for c, g in zip(ordre, groupes)])
    ax.set_xlabel(categorie)
    ax.set_ylabel(_etiquette(valeur, unite))
    _habiller(ax, titre, source)
    plt.show()


def camembert(df, categorie, valeur=None, *, titre=None, source=""):
    """Montrer une part d'un tout : un camembert, 5 parts au plus (sinon, il refuse et le dit).

    Sans `valeur`, compte les lignes de chaque catégorie ; avec `valeur`, additionne cette colonne.

    Exemple :
        camembert(df, "station", titre="Col ne fournit qu'un relevé sur neuf", source="Relevés 2024")
    """
    _verifier(df, categorie, valeur)
    _verifier_titre(titre)
    parts = df[categorie].value_counts() if valeur is None else df.groupby(categorie)[valeur].sum()
    parts = parts.sort_values(ascending=False)
    if len(parts) > 5:
        print(f"REFUS : {len(parts)} parts, c'est trop pour un camembert (5 au plus). "
              "Utilisez plutôt barres(...), bien plus lisible.")
        return None
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(parts.values, labels=parts.index, autopct="%1.0f %%", startangle=90, counterclock=False,
           colors=[BLEU, ORANGE, GRIS, GRIS_CLAIR, "#e6e6e6"][:len(parts)],
           wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title(titre, fontsize=12, fontweight="bold", loc="left")
    pied = f"Source : {source} · " if source else ""
    fig.text(0.99, 0.005, pied + "tracé le " + datetime.date.today().strftime("%d/%m/%Y"),
             ha="right", va="bottom", fontsize=8, color=GRIS)
    plt.show()


def barres_croisees(df, categorie, sous_categorie, valeur, agg="mean", *, titre=None, unite="", source=""):
    """Croiser deux catégories : des barres groupées, une couleur par sous-catégorie (le regard 7).

    Exemple :
        barres_croisees(df, "station", "saison", "pluie_mm", titre="Partout, il pleut plus en saison des pluies",
                        unite="mm", source="Relevés 2024")
    """
    _verifier(df, categorie, sous_categorie, valeur)
    _verifier_titre(titre)
    table = df.pivot_table(index=categorie, columns=sous_categorie, values=valeur, aggfunc=agg)
    table = table.loc[table.mean(axis=1).sort_values(ascending=False).index]
    if table.shape[1] > 4:
        print(f"ATTENTION : {table.shape[1]} sous-catégories, c'est beaucoup. Le graphique risque d'être illisible.")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    largeur = 0.8 / table.shape[1]
    for i, (colonne, couleur) in enumerate(zip(table.columns, COULEURS + [GRIS_CLAIR] * 10)):
        positions = [x + (i - (table.shape[1] - 1) / 2) * largeur for x in range(len(table))]
        ax.bar(positions, table[colonne].values, width=largeur, color=couleur, label=str(colonne))
    ax.set_xticks(range(len(table)), table.index)
    ax.legend(title=sous_categorie, frameon=False)
    ax.set_xlabel(categorie)
    ax.set_ylabel(_etiquette(valeur, unite, agg))
    _habiller(ax, titre, source)
    plt.show()


def correlations(df, colonnes=None, *, titre=None, source=""):
    """Voir tous les liens d'un coup : la matrice des corrélations, colorée, avec les valeurs écrites.

    Bleu = lien positif, orange = lien négatif, blanc = pas de lien en ligne droite.

    Exemple :
        correlations(df, titre="Les deux températures vont ensemble", source="Relevés 2024")
    """
    if colonnes is not None:
        _verifier(df, *colonnes)
    _verifier_titre(titre)
    tableau = df[colonnes] if colonnes is not None else df.select_dtypes("number")
    matrice = tableau.corr()
    couleurs = LinearSegmentedColormap.from_list("orange_blanc_bleu", [ORANGE, "white", BLEU])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrice.values, cmap=couleurs, vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrice)), matrice.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrice)), matrice.columns)
    for i in range(len(matrice)):
        for j in range(len(matrice)):
            ax.text(j, i, f"{matrice.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, label="corrélation r")
    ax.set_title(titre, fontsize=12, fontweight="bold", loc="left")
    pied = f"Source : {source} · " if source else ""
    fig.text(0.99, 0.005, pied + "tracé le " + datetime.date.today().strftime("%d/%m/%Y"),
             ha="right", va="bottom", fontsize=8, color=GRIS)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    plt.show()
