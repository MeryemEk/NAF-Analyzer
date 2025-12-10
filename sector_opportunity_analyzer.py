"""
sector_opportunity_analyzer.py

Construit une matrice d'opportunités en confrontant :
- les clusters de DEMANDE par département (ex : active colonies, breeding grounds, dormant nests)
- les clusters d'OFFRE par département (ex : high supply, normal or low supply)

Puis trace une carte des opportunités.

Usage typique (dans un notebook) :

    from sector_opportunity_analyzer import run_opportunity_analysis

    confront = run_opportunity_analysis(
        df_demand_clusters=demand_df,        # doit contenir dept_group + cluster_demand
        df_offer_clusters=offer_df,          # doit contenir dept_group + cluster_offre
        departements_grouped=departements_gdf,  # GeoDataFrame avec dept_group + geometry
        total_per_dept=total_per_dept_df     # optionnel, avec dept_name si dispo
    )

Le script :
- print le dataframe d'opportunité (trié par priorité)
- affiche la carte
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    # Pour garder la même charte que les autres scripts
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



# --------------------------------------------------------------------
# 1. Construction de la matrice d'opportunité
# --------------------------------------------------------------------

def compute_opportunity_matrix(
    df_demand_clusters: pd.DataFrame,
    df_offer_clusters: pd.DataFrame,
    total_per_dept: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Combine les clusters de demande et d'offre pour produire une matrice
    d'opportunités par département.

    Hypothèses de colonnes :
    - df_demand_clusters : ['dept_group', 'cluster_demand', ...]
    - df_offer_clusters  : ['dept_group', 'cluster_offre', ...]
    - total_per_dept (optionnel) : ['dept_group', 'dept_name', ...] pour ajouter le nom du département.

    Règles :

        demand = active colonies, supply = normal or low supply   -> white space (prio 1)
        demand = breeding grounds, supply = normal or low supply  -> growth frontiers (prio 2)
        demand = active colonies, supply = high supply            -> crowded spots (prio 3)
        demand = breeding grounds, supply = high supply           -> emerging competition (prio 4)
        demand = dormant nests, supply = normal or low supply     -> quiet zones (prio 5)
        demand = dormant nests, supply = high supply              -> over-served areas (prio 6)
        sinon                                                     -> undefined (prio 7)
    """

    # On garde les colonnes nécessaires
    demand = df_demand_clusters[["dept_group", "cluster_demand"]].copy()
    offer = df_offer_clusters[["dept_group", "cluster_offre"]].copy()

    confront = demand.merge(offer, on="dept_group", how="left")

    def categorize_opportunity(row):
        d = row["cluster_demand"]
        s = row["cluster_offre"]
        if d == "active colonies" and s == "normal or low supply":
            return "white space", 1
        elif d == "breeding grounds" and s == "normal or low supply":
            return "growth frontiers", 2
        elif d == "active colonies" and s == "high supply":
            return "crowded spots", 3
        elif d == "breeding grounds" and s == "high supply":
            return "emerging competition", 4
        elif d == "dormant nests" and s == "normal or low supply":
            return "quiet zones", 5
        elif d == "dormant nests" and s == "high supply":
            return "over-served areas", 6
        else:
            return "undefined", 7

    confront[["opportunity", "priority"]] = confront.apply(
        categorize_opportunity, axis=1, result_type="expand"
    )

    # Ajout du nom de département si dispo
    if total_per_dept is not None and "dept_group" in total_per_dept.columns:
        cols = ["dept_group"]
        if "dept_name" in total_per_dept.columns:
            cols.append("dept_name")
        total_unique = total_per_dept[cols].drop_duplicates()
        confront = confront.merge(total_unique, on="dept_group", how="left")

    # Tri pour lecture
    confront = confront.sort_values(["priority", "dept_group"]).reset_index(drop=True)

    # Print du tableau comme demandé
    print("\n=== Opportunity matrix (offer vs demand) ===")
    print(confront)

    return confront


# --------------------------------------------------------------------
# 2. Carte des opportunités
# --------------------------------------------------------------------

def plot_opportunity_map(
    confront: pd.DataFrame,
    departements_grouped: gpd.GeoDataFrame,
    save_path: Optional[str] = None,
):
    """
    Trace la carte des opportunités par département.

    Hypothèses :
    - confront contient ['dept_group', 'opportunity', 'priority', ...]
    - departements_grouped contient ['dept_group', 'geometry']
    """

    cluster_map_data = departements_grouped.merge(
        confront, on="dept_group", how="left"
    )

    cluster_colors = {
        "white space": "#53D6C7",
        "growth frontiers": COMPANY_PALETTE["forest"],
        "crowded spots": COMPANY_PALETTE["coral"],
        "emerging competition": COMPANY_PALETTE["nude"],
        "quiet zones": COMPANY_PALETTE["light_grey"],
        "over-served areas": COMPANY_PALETTE["grey"]
    }

    cluster_map_data["color"] = cluster_map_data["opportunity"].map(cluster_colors)
    # Debug : afficher les lignes sans couleur
    missing_colors = cluster_map_data[cluster_map_data["color"].isna()][
        ["dept_group", "opportunity", "cluster_demand", "cluster_offre"]
    ].drop_duplicates()
    if not missing_colors.empty:
        print("\nOpportunités non mappées dans cluster_colors :")
        print(missing_colors)

    # Couleur par défaut pour éviter l'erreur Matplotlib
    cluster_map_data["color"] = cluster_map_data["color"].fillna("#DDDDDD")

    fig, ax = plt.subplots(figsize=(20, 10))
    cluster_map_data.plot(
        color=cluster_map_data["color"],
        edgecolor="white",
        ax=ax,
    )

    plt.title(
        "Market opportunity map by department\n"
        "(based on supply and demand clusters)",
        pad=20,
    )
    ax.set_axis_off()

    # Codes de département au centre de chaque polygone
    for _, row in cluster_map_data.iterrows():
        if pd.notna(row["dept_group"]):
            centroid = row["geometry"].centroid
            # Couleur du texte : blanc si fond sombre (crowded / over-served), noir sinon
            text_color = (
                "white"
                if row["opportunity"] in ["crowded spots", "over-served areas"]
                else "black"
            )
            ax.text(
                centroid.x,
                centroid.y,
                row["dept_group"],
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="bold",
            )

    # Légende
    legend_elements = [
        Patch(facecolor=color, edgecolor="white", label=cluster)
        for cluster, color in cluster_colors.items()
    ]
    plt.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=3,
        fontsize=10,
    )

    # Note bas de page
    plt.figtext(
        0.5,
        -0.15,
        "REKOLT analysis.\n"
        "Paris (75) and 1st couronne (92,93,94) are grouped.\n"
        "Market opportunity clusters are defined by combining supply clusters "
        "(based on number of 3D establishments per 100k inhab.)\n"
        "and demand clusters (based on public tenders).\n"
        'For instance, "white space" are departments with high demand but normal or low supply.',
        ha="center",
        fontsize=8,
        color=COMPANY_PALETTE["grey"],
        fontstyle="italic",
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# --------------------------------------------------------------------
# 3. Fonction "one shot" : compute + print + plot
# --------------------------------------------------------------------

def run_opportunity_analysis(
    df_demand_clusters: pd.DataFrame,
    df_offer_clusters: pd.DataFrame,
    departements_grouped: gpd.GeoDataFrame,
    total_per_dept: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Enchaîne :
    - construction de la matrice d'opportunités
    - print du dataframe
    - affichage de la carte

    Retourne le dataframe d'opportunité (confront).
    """
    confront = compute_opportunity_matrix(
        df_demand_clusters=df_demand_clusters,
        df_offer_clusters=df_offer_clusters,
        total_per_dept=total_per_dept,
    )
    
    save_path = None
    if output_dir is not None:
        from pathlib import Path
        save_path = str(Path(output_dir) / "opportunity_map.png")
    
    plot_opportunity_map(confront, departements_grouped, save_path=save_path)
    
    return confront