"""
Analyse de l'offre (densité d'établissements par département, clusters d'offre).

Usage :
    from sector_offer_analyzer import run_analysis

    results = run_analysis(
        filepath="NAF files/81.29A_Pest Control 3D.csv",
        naf_code="81.29A",
        output_dir="./outputs/pest_control_3D_offer",
        pop_filepath="popdept.xlsx",
    )
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# -------------------------------------------------------------------
# Palette couleurs (adapter si tu as déjà un dictionnaire global)
# -------------------------------------------------------------------
try:
    # Si tu as déjà défini company_palette dans un autre module
    from sector_entities_analyzer import COMPANY_PALETTE  # type: ignore
except ImportError:
    COMPANY_PALETTE = {
        
    "forest": "#10433E",
    "muted_green": "#4C7470",
    "mint": "#A4C4B6",
    "grey": "#505050",
    "light_grey": "#BEBEBE",
    "coral": "#F08878",
    "nude": "#F5D5C6",
    "purple_grey": "#756F7B",
    "taupe": "#A8958F",
    "red_highlight": "#FF4441",
    "gold_highlight": "#CA9D66"
    }

GRAND_PARIS_DEPTS = {"75", "92", "93", "94"}


# -------------------------------------------------------------------
# Helpers géographiques / population
# -------------------------------------------------------------------

def _download_if_not_exists(url: str, output: Path) -> Path:
    """Télécharge un fichier si non présent en local."""
    if not output.exists():
        r = requests.get(url)
        r.raise_for_status()
        with open(output, "wb") as f:
            f.write(r.content)
    return output


def load_departements_geojson(cache_dir: Path = Path(".")) -> gpd.GeoDataFrame:
    """Charge les géométries des départements (GeoJSON)."""
    url = "https://france-geojson.gregoiredavid.fr/repo/departements.geojson"
    output = cache_dir / "departements.geojson"
    _download_if_not_exists(url, output)
    departements_gdf = gpd.read_file(output)

    # Codes en str
    departements_gdf["code"] = departements_gdf["code"].astype(str)

    # Buffer(0) pour corriger les géométries bizarres
    departements_gdf["geometry"] = departements_gdf.buffer(0)

    # Groupement Grand Paris
    departements_gdf["dept_group"] = np.where(
        departements_gdf["code"].isin(GRAND_PARIS_DEPTS),
        "75*",
        departements_gdf["code"],
    )

    # Dissolve par groupe (Paris + 1ère couronne fusionnés)
    departements_grouped = departements_gdf.dissolve(by="dept_group", as_index=False)

    return departements_grouped


def load_population(pop_filepath: str) -> pd.DataFrame:
    """
    Charge la population par département et construit le regroupement Grand Paris.
    Fichier attendu : 'popdept.xlsx', onglet '2023', avec colonnes 'Unnamed: 0','Unnamed: 1','Total'.
    """
    pop = pd.read_excel(pop_filepath, sheet_name="2023", header=4)
    pop = pop[["Unnamed: 0", "Unnamed: 1", "Total"]].copy()
    pop.rename(
        columns={
            "Unnamed: 0": "dept_code",
            "Unnamed: 1": "dept_name",
            "Total": "population",
        },
        inplace=True,
    )

    # Groupement Grand Paris
    pop["dept_code"] = pop["dept_code"].astype(str)
    pop["dept_group"] = pop["dept_code"].apply(
        lambda x: "75*" if x in GRAND_PARIS_DEPTS else x
    )
    pop["dept_name"] = pop.apply(
        lambda row: "Paris + 1st Couronne"
        if row["dept_code"] in GRAND_PARIS_DEPTS
        else row["dept_name"],
        axis=1,
    )

    pop_grouped = (
        pop.groupby("dept_group")
        .agg({"dept_name": "first", "population": "sum"})
        .reset_index()
    )

    return pop_grouped


# -------------------------------------------------------------------
# Helpers pour les codes communes / départements
# -------------------------------------------------------------------

def normalize_commune(code):
    """Normalise les codes communes (fusion des arrondissements de Paris, Lyon, Marseille)."""
    if pd.isna(code):
        return np.nan
    c = str(code)
    # Paris arrondissements (75101–75120) → 75056
    if c.startswith("751") and c != "75056":
        return "75056"
    # Lyon arrondissements (69381–69389) → 69123
    if c.startswith("6938") and c != "69123":
        return "69123"
    # Marseille arrondissements (13201–13216) → 13055
    if c.startswith("132") and c != "13055":
        return "13055"
    return c


def dept_from_commune(code: str) -> str:
    """Récupère le code département à partir du code commune INSEE."""
    if pd.isna(code):
        return np.nan
    c = str(code).upper()
    if c.startswith("2A"):
        return "2A"
    if c.startswith("2B"):
        return "2B"
    if c.startswith(("971", "972", "973", "974", "975", "976")):
        return c[:3]
    return c[:2].zfill(2)


# -------------------------------------------------------------------
# Construction de l'offre par département
# -------------------------------------------------------------------

def compute_offer_by_dept(data: pd.DataFrame, pop_grouped: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un DataFrame 'offre' :
    - dept_group
    - dept_name
    - population
    - establishments_count (établissements actifs uniques)
    - establishments_per_100k
    """

    df = data.copy()

    # Normalisation commune et département
    df["commune"] = df["codeCommuneEtablissement"].apply(normalize_commune)
    df["dept_code"] = df["commune"].apply(dept_from_commune)
    df["dept_group"] = df["dept_code"].apply(
        lambda x: "75*" if x in GRAND_PARIS_DEPTS else x
    )

    # Filtre unités légales + établissements actifs
    df = df[df["etatAdministratifUniteLegale"] == "A"]
    df = df[df["etatAdministratifEtablissement"] == "A"]

    # Nombre d'établissements actifs uniques par dept_group (unique siret)
    estab_by_dept = (
        df.groupby("dept_group")["siret"].nunique().reset_index(name="establishments_count")
    )

    # Merge avec population
    offre = estab_by_dept.merge(pop_grouped, on="dept_group", how="left")

    # Densité : nb d'établissements / 100k habitants
    offre["establishments_per_100k"] = (
        offre["establishments_count"] / offre["population"] * 100000
    )

    return offre


# -------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------

def plot_offer_map(
    departements_grouped: gpd.GeoDataFrame,
    offre: pd.DataFrame,
    naf_code: str,
    save_path: Path | None = None,
):
    """
    Carte choroplèthe : établissements actifs uniques / 100k hab par département (dept_group).
    """
    map_data = departements_grouped.merge(
        offre, on="dept_group", how="left"
    )

    # GeoDataFrame
    map_data = gpd.GeoDataFrame(map_data, geometry="geometry", crs=departements_grouped.crs)

    fig, ax = plt.subplots(figsize=(20, 15))

    # Choroplèthe
    cmap = "Blues"
    map_data.plot(
        column="establishments_per_100k",
        cmap=cmap,
        legend=True,
        legend_kwds={
            "label": "Active unique establishments per 100k inhabitants",
            "orientation": "horizontal",
            "shrink": 0.3,
            "pad": 0.01,
            "aspect": 50,
        },
        ax=ax,
        edgecolor="white",
    )

    # Labels dans chaque polygone
    for _, row in map_data.iterrows():
        if pd.notna(row["establishments_per_100k"]):
            centroid = row["geometry"].centroid
            plt.annotate(
                text=f"{row['establishments_per_100k']:.0f}",
                xy=(centroid.x, centroid.y),
                ha="center",
                fontsize=8,
                color="black",
                bbox=dict(
                    facecolor="white",
                    alpha=1,
                    edgecolor="none",
                    pad=1,
                ),
            )

    # Petite légende "x"
    plt.text(
        -2.5,
        38.5,
        "x",
        fontsize=10,
        color="black",
        bbox=dict(facecolor="white", alpha=1, edgecolor="black", pad=2),
    )
    plt.text(
        -2,
        38.5,
        "Number of unique establishments/100k inhab. for each department",
        fontsize=10,
        color=COMPANY_PALETTE["ocean"],
        va="center",
    )

    plt.title(
        f"Active unique establishments per 100k inhabitants by department (NAF {naf_code})",
        pad=20,
    )
    ax.set_axis_off()

    plt.figtext(
        0.35,
        0.02,
        f"INSEE company registry (NAF {naf_code}). REKOLT analysis. "
        "Data for 2025 are current as of September 30.\nParis (75) and 1st couronne are grouped.",
        ha="left",
        fontsize=8,
        color=COMPANY_PALETTE["grey"],
        fontstyle="italic",
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_estab_per_100k_hist(offre: pd.DataFrame, naf_code: str, save_path: Path | None = None):
    """
    Histogramme de la distribution des établissements / 100k hab par département.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(
        offre["establishments_per_100k"],
        bins=20,
        kde=True,
        color=COMPANY_PALETTE["ocean"],
    )
    plt.title(
        f"Distribution of active unique establishments per 100k inhabitants by department (NAF {naf_code})"
    )
    plt.xlabel("Active unique establishments per 100k inhabitants")
    plt.ylabel("Number of departments")
    plt.grid(axis="y")

    plt.figtext(
        0.5,
        0.01,
        f"INSEE company registry (NAF {naf_code}). REKOLT analysis.",
        ha="center",
        fontsize=8,
        color=COMPANY_PALETTE["grey"],
        fontstyle="italic",
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def add_supply_clusters(offre: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Ajoute une colonne cluster_offre en fonction de la densité d'établissements / 100k :
    - 'high supply' : > median + 0.5 * std
    - 'normal or low supply' : sinon
    Retourne (offre_avec_cluster, threshold).
    """
    df = offre.copy()
    median_estab = df["establishments_per_100k"].median()
    std_estab = df["establishments_per_100k"].std()
    threshold = median_estab + 0.5 * std_estab

    df["cluster_offre"] = np.where(
        df["establishments_per_100k"] > threshold,
        "high supply",
        "normal or low supply",
    )

    return df, threshold


def plot_cluster_map(
    departements_grouped: gpd.GeoDataFrame,
    offre_clustered: pd.DataFrame,
    naf_code: str,
    threshold: float,
    save_path: Path | None = None,
):
    """
    Carte des clusters d'offre :
    - couleur par cluster_offre
    - label : nb d'établissements / 100k hab
    """
    cluster_map_data = departements_grouped.merge(
        offre_clustered, on="dept_group", how="left"
    )
    cluster_map_data = gpd.GeoDataFrame(
        cluster_map_data, geometry="geometry", crs=departements_grouped.crs
    )

    cluster_colors = {
        "high supply": COMPANY_PALETTE["ocean"],
        "normal or low supply": COMPANY_PALETTE["light_grey"],
    }
    cluster_map_data["color"] = cluster_map_data["cluster_offre"].map(cluster_colors)
    cluster_map_data["color"] = cluster_map_data["color"].fillna("#DDDDDD")

    fig, ax = plt.subplots(figsize=(20, 10))
    cluster_map_data.plot(
        color=cluster_map_data["color"],
        edgecolor="white",
        ax=ax,
    )

    # Labels densité
    for _, row in cluster_map_data.iterrows():
        if pd.notna(row["establishments_per_100k"]):
            centroid = row["geometry"].centroid
            plt.annotate(
                text=f"{row['establishments_per_100k']:.0f}",
                xy=(centroid.x, centroid.y),
                ha="center",
                fontsize=8,
                color="black",
                bbox=dict(facecolor="white", alpha=1, edgecolor="none", pad=1),
            )

    plt.title(
        f"Departments clustered by supply index (based on active establishments) – NAF {naf_code}",
        pad=20,
    )
    ax.set_axis_off()

    # Légende des clusters
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color, edgecolor="white", label=cluster)
        for cluster, color in cluster_colors.items()
    ]
    plt.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=2,
        fontsize=10,
    )

    # Petit "x" explicatif
    plt.text(
        -2.5,
        40.5,
        "x",
        fontsize=10,
        color="black",
        bbox=dict(facecolor="white", alpha=1, edgecolor="black", pad=2),
    )
    plt.text(
        -2,
        40.5,
        "Number of unique establishments/100k inhab. for each department",
        fontsize=10,
        color=COMPANY_PALETTE["ocean"],
        va="center",
    )

    plt.figtext(
        0.5,
        0.0,
        f"INSEE company registry (NAF {naf_code}). REKOLT analysis.\n"
        "Paris (75) and 1st couronne (92,93,94) are grouped.\n"
        "Supply clusters are defined by the number of establishments per 100k inhabitants.\n"
        f'For instance, "high supply" are departments with a high density of establishments '
        f"(> median + 0.5 × std, i.e. > {threshold:.1f} establishments/100k).",
        ha="center",
        fontsize=8,
        color=COMPANY_PALETTE["grey"],
        fontstyle="italic",
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# -------------------------------------------------------------------
# run_analysis : point d'entrée principal
# -------------------------------------------------------------------

def run_analysis(
    filepath: str,
    naf_code: str,
    output_dir: str | None = None,
    pop_filepath: str = "popdept.xlsx",
):
    """
    Analyse de l'offre pour un fichier d'établissements filtré sur un code NAF.
    - filepath : CSV/TXT INSEE déjà filtré ou non
    - naf_code : ex. '81.29A'
    - output_dir : dossier où sauver les graphes (optionnel)
    - pop_filepath : fichier Excel de population par département
    """

    # Pour rester cohérent avec le loader existant
    try:
        from sector_entities_analyzer import load_and_clean_data  # type: ignore
        data = load_and_clean_data(filepath, naf_code)
    except ImportError:
        # fallback : lecture simple
        data = pd.read_csv(filepath, dtype=str)
        # si besoin tu peux rajouter ici un filtrage naf_code sur activitePrincipaleEtablissement
        if "activitePrincipaleEtablissement" in data.columns:
            data = data[data["activitePrincipaleEtablissement"] == naf_code]

    print(f"ANALYSE DE L'OFFRE POUR LE SECTEUR NAF {naf_code}")
    print("=" * 70)

    output_path = None
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # 1. Données géographiques et population
    print("\n--- Chargement des données géographiques et de population ---")
    departements_grouped = load_departements_geojson()
    pop_grouped = load_population(pop_filepath)

    # 2. Construction de l'offre par département
    print("\n--- Calcul de la densité d'établissements par département ---")
    offre = compute_offer_by_dept(data, pop_grouped)

    # 3. Carte des établissements par 100k hab
    print("\n--- Carte : établissements actifs / 100k habitants ---")
    save_map_path = output_path / "01_offer_map.png" if output_path else None
    plot_offer_map(departements_grouped, offre, naf_code, save_map_path)

    # 4. Histogramme de la distribution
    print("\n--- Distribution de la densité d'établissements ---")
    save_hist_path = output_path / "02_offer_histogram.png" if output_path else None
    plot_estab_per_100k_hist(offre, naf_code, save_hist_path)

    # 5. Clusters d'offre
    print("\n--- Clusters d'offre (high supply vs normal/low) ---")
    offre_clustered, threshold = add_supply_clusters(offre)
    save_cluster_map_path = (
        output_path / "03_offer_clusters.png" if output_path else None
    )
    plot_cluster_map(departements_grouped, offre_clustered, naf_code, threshold, save_cluster_map_path)

    # Retourner les DF utiles pour analyse complémentaire
    results = {
        "offre": offre,
        "offre_clustered": offre_clustered,
        "threshold": threshold,
        "departements_grouped": departements_grouped
    }

    return results
