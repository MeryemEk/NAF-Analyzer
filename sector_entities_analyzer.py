#!/usr/bin/env python3
"""
Sector Entities Analyzer
========================

Script pour analyser les entités d'un secteur à partir des données INSEE/SIRENE.

Analyses incluses:
- Évolution du nombre d'unités légales par année de création
- Évolution du nombre d'établissements par année de création  
- Distribution des établissements par entité légale
- Top 10 entités par nombre d'établissements (par SIREN)
- Greenfield vs Brownfield establishments
- Taux de survie par durée d'existence (greenfield vs brownfield)

Usage:
    python sector_entities_analyzer.py data.csv --naf 81.30Z --output ./outputs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from pathlib import Path
from typing import Optional
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - PALETTE ET STYLE
# =============================================================================

COMPANY_PALETTE = {
    "ocean": "#071F3B",
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


def setup_style():
    """Configure le style matplotlib et seaborn."""
    full_palette = list(COMPANY_PALETTE.values())
    sns.set_theme(style="whitegrid", palette=full_palette)
    
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'axes.edgecolor': COMPANY_PALETTE["grey"],
        'axes.labelcolor': COMPANY_PALETTE["ocean"],
        'xtick.color': COMPANY_PALETTE["grey"],
        'ytick.color': COMPANY_PALETTE["grey"],
        'grid.color': COMPANY_PALETTE["light_grey"],
        'figure.facecolor': 'white',
        'figure.titlesize': 16,
        'figure.titleweight': 'bold'
    })


# =============================================================================
# CHARGEMENT ET NETTOYAGE DES DONNÉES
# =============================================================================

def load_and_clean_data(filepath: str, naf_code: str = "81.30Z") -> pd.DataFrame:
    """
    Charge et nettoie les données INSEE/SIRENE.
    
    Parameters
    ----------
    filepath : str
        Chemin vers le fichier CSV
    naf_code : str
        Code NAF pour filtrer les données
        
    Returns
    -------
    pd.DataFrame
        Données nettoyées et filtrées
    """
    print(f"Chargement des données depuis {filepath}...")
    data = pd.read_csv(filepath)
    
    # Colonnes à supprimer
    cols_to_drop = [
        'nic', 'trancheEffectifsEtablissement', 'anneeEffectifsEtablissement',
        'unitePurgeeUniteLegale', 'sigleUniteLegale',
        'denominationUsuelle1UniteLegale', 'denominationUsuelle2UniteLegale',
        'denominationUsuelle3UniteLegale', 'codeCommune2Etablissement',
        'codeCedex2Etablissement', 'libelleCedex2Etablissement',
        'codePaysEtranger2Etablissement', 'libellePaysEtranger2Etablissement',
        'codeCedexEtablissement', 'numeroVoie2Etablissement',
        'indiceRepetition2Etablissement', 'typeVoie2Etablissement',
        'libelleVoie2Etablissement', 'codePostal2Etablissement',
        'libelleCommune2Etablissement', 'libelleCommuneEtranger2Etablissement',
        'distributionSpeciale2Etablissement', 'complementAdresseEtablissement',
        'numeroVoieEtablissement', 'indiceRepetitionEtablissement',
        'dernierNumeroVoieEtablissement', 'indiceRepetitionDernierNumeroVoieEtablissement',
        'typeVoieEtablissement', 'libelleVoieEtablissement',
        'nomenclatureActivitePrincipaleUniteLegale', 'identifiantAssociationUniteLegale',
        'economieSocialeSolidaireUniteLegale', 'societeMissionUniteLegale',
        'statutDiffusionEtablissement', 'statutDiffusionUniteLegale',
        'caractereEmployeurUniteLegale'
    ]
    
    existing_cols = [c for c in cols_to_drop if c in data.columns]
    data = data.drop(columns=existing_cols, errors="ignore")

    # --- Conversion robuste des dates (Excel numérique OU texte) ---
    def _convert_excel_or_string_date(col: pd.Series) -> pd.Series:
        # On remplace les chaînes vides par NA
        col = col.replace(["", " "], pd.NA)

        # Tentative de conversion en numérique (format Excel)
        numeric = pd.to_numeric(col, errors="coerce")

        # Si au moins ~50 % des valeurs sont numériques, on considère que c'est du Excel
        if numeric.notna().sum() > 0 and numeric.notna().sum() >= 0.5 * len(col):
            dates = pd.to_datetime("1899-12-30") + pd.to_timedelta(numeric, unit="D")
        else:
            # Sinon on parse comme des dates "classiques" (texte)
            dates = pd.to_datetime(col, errors="coerce", dayfirst=True)

        # Correction des années aberrantes > 2025
        dates = dates.apply(
            lambda x: x.replace(year=x.year - 100) if pd.notna(x) and x.year > 2025 else x
        )
        return dates

    if "dateCreationEtablissement" in data.columns:
        data["dateCreationEtablissement"] = _convert_excel_or_string_date(
            data["dateCreationEtablissement"]
        )

    if "dateCreationUniteLegale" in data.columns:
        data["dateCreationUniteLegale"] = _convert_excel_or_string_date(
            data["dateCreationUniteLegale"]
        )

    # Filtrage par code NAF
    data = data[
        (data["activitePrincipaleEtablissement"] == naf_code)
        & (data["activitePrincipaleUniteLegale"] == naf_code)
    ]

    print(f"  → {len(data)} établissements chargés")
    print(f"  → {data['siren'].nunique()} unités légales")
    
    return data


# =============================================================================
# ANALYSES
# =============================================================================

def analyze_units_per_year(data: pd.DataFrame) -> pd.DataFrame:
    """Évolution du nombre d'unités légales par année de création."""
    units_per_year = data.groupby(
        data['dateCreationUniteLegale'].dt.year
    )['siren'].nunique().reset_index()
    units_per_year.columns = ['Year', 'Number of Unique Legal Units']
    return units_per_year


def analyze_establishments_per_year(data: pd.DataFrame) -> pd.DataFrame:
    """Évolution du nombre d'établissements par année de création."""
    estab_per_year = data.groupby(
        data['dateCreationEtablissement'].dt.year
    )['siret'].nunique().reset_index()
    estab_per_year.columns = ['Year', 'Number of Unique Establishments']
    return estab_per_year


def analyze_distribution_by_entity(data: pd.DataFrame) -> pd.DataFrame:
    """
    Distribution des établissements par entité légale (SIREN).
    - Uniquement unités légales actives
    - Uniquement établissements actifs
    - Résultat : nombre d'unités légales par nombre d'établissements actifs
    """
    df1 = data.copy()

    # Filtrer uniquement les unités légales actives et établissements actifs
    df1 = df1[df1["etatAdministratifUniteLegale"] == "A"]
    df1 = df1[df1["etatAdministratifEtablissement"] == "A"]

    # On ne garde que siren / siret
    df1 = df1[["siren", "siret"]]

    # Nombre d'établissements par SIREN
    df2 = df1.groupby("siren")["siret"].nunique().reset_index()
    df2 = df2.rename(columns={"siret": "number_of_establishments"})

    # Distribution : combien d’unités légales ont 1, 2, 3… établissements
    distribution = df2["number_of_establishments"].value_counts().reset_index()
    # Forcer les noms de colonnes proprement
    distribution.columns = ["Number of Establishments", "Number of Unique Legal Units"]

    # Trier par nombre d'établissements croissant
    distribution = distribution.sort_values(by="Number of Establishments")

    # Part dans le total (%)
    distribution["Share of Total (%)"] = (
        distribution["Number of Unique Legal Units"]
        / distribution["Number of Unique Legal Units"].sum()
        * 100
    ).round(2)

    return distribution

def analyze_top_entities(data: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Top N entités par nombre d'établissements actifs.
    Groupé par SIREN uniquement (pas d'harmonisation de noms).
    """
    df = data.copy()
    # Filtrer actifs uniquement
    df = df[df['etatAdministratifEtablissement'] == 'A']
    
    # Compter établissements par SIREN
    df2 = df.groupby('siren')['siret'].nunique().reset_index()
    df2.columns = ['siren', 'number_of_establishments']
    df2 = df2.sort_values('number_of_establishments', ascending=False)
    
    # Top N
    top_n = df2.head(n).copy()
    top_n['siren'] = top_n['siren'].astype(str)
    
    return top_n





def analyze_greenfield_brownfield(data: pd.DataFrame, start_year: int = 2005) -> pd.DataFrame:
    """
    Analyse greenfield vs brownfield par année.
    
    Greenfield: établissement créé en même temps que l'unité légale
    Brownfield: établissement ajouté à une entité existante
    """
    df = data.copy()
    df['yearCreationEtablissement'] = df['dateCreationEtablissement'].dt.year
    
    df1 = df[['siren', 'dateCreationUniteLegale', 'dateCreationEtablissement', 
              'siret', 'yearCreationEtablissement']].drop_duplicates()
    
    df1 = df1[df1['yearCreationEtablissement'] >= 1980]
    
    # Greenfield = même date de création
    df1['greenfield'] = np.where(
        df1['dateCreationUniteLegale'] == df1['dateCreationEtablissement'], 1, 0
    )
    
    # Pivot
    df2 = df1.pivot_table(
        index='yearCreationEtablissement',
        columns='greenfield',
        values='siret',
        aggfunc='count',
        fill_value=0
    ).reset_index()
    
    df2.columns = ['Year', 'Brownfield Establishments', 'Greenfield Establishments']
    df2['Total Establishments'] = df2['Brownfield Establishments'] + df2['Greenfield Establishments']
    df2['% Greenfield'] = (df2['Greenfield Establishments'] / df2['Total Establishments'] * 100).round(2)
    df2['% Brownfield'] = (df2['Brownfield Establishments'] / df2['Total Establishments'] * 100).round(2)
    
    # Filtrer depuis start_year
    df2 = df2[df2['Year'] >= start_year]
    
    return df2


def analyze_survival_by_duration(data: pd.DataFrame, current_year: int = 2025) -> tuple:
    """
    Taux de survie par durée d'existence, séparé greenfield vs brownfield.
    
    Returns
    -------
    tuple
        (merged_greenfield, merged_brownfield)
    """
    df = data.copy()
    df['yearCreationEtablissement'] = df['dateCreationEtablissement'].dt.year
    
    df1 = df[['siren', 'dateCreationUniteLegale', 'dateCreationEtablissement', 
              'siret', 'yearCreationEtablissement', 'etatAdministratifEtablissement']].drop_duplicates()
    
    df1 = df1[df1['yearCreationEtablissement'] >= 1980]
    df1['greenfield'] = np.where(
        df1['dateCreationUniteLegale'] == df1['dateCreationEtablissement'], 1, 0
    )
    df1['Duration of Existence'] = current_year - df1['yearCreationEtablissement']
    
    # Séparer greenfield et brownfield
    df_greenfield = df1[df1['greenfield'] == 1]
    df_brownfield = df1[df1['greenfield'] == 0]
    
    # Total par durée - Greenfield
    estab_greenfield = df_greenfield.groupby('Duration of Existence')['siret'].nunique().reset_index()
    estab_greenfield.columns = ['Duration', 'Total Greenfield Establishments']
    
    # Actifs par durée - Greenfield
    active_greenfield = df_greenfield[
        df_greenfield['etatAdministratifEtablissement'] == 'A'
    ].groupby('Duration of Existence')['siret'].nunique().reset_index()
    active_greenfield.columns = ['Duration', 'Active Greenfield Establishments']
    
    # Merge Greenfield
    merged_greenfield = pd.merge(estab_greenfield, active_greenfield, on='Duration', how='left')
    merged_greenfield['Active Greenfield Establishments'] = merged_greenfield['Active Greenfield Establishments'].fillna(0)
    merged_greenfield['Percentage Active Greenfield'] = (
        merged_greenfield['Active Greenfield Establishments'] / 
        merged_greenfield['Total Greenfield Establishments'] * 100
    )
    
    # Total par durée - Brownfield
    estab_brownfield = df_brownfield.groupby('Duration of Existence')['siret'].nunique().reset_index()
    estab_brownfield.columns = ['Duration', 'Total Brownfield Establishments']
    
    # Actifs par durée - Brownfield
    active_brownfield = df_brownfield[
        df_brownfield['etatAdministratifEtablissement'] == 'A'
    ].groupby('Duration of Existence')['siret'].nunique().reset_index()
    active_brownfield.columns = ['Duration', 'Active Brownfield Establishments']
    
    # Merge Brownfield
    merged_brownfield = pd.merge(estab_brownfield, active_brownfield, on='Duration', how='left')
    merged_brownfield['Active Brownfield Establishments'] = merged_brownfield['Active Brownfield Establishments'].fillna(0)
    merged_brownfield['Percentage Active Brownfield'] = (
        merged_brownfield['Active Brownfield Establishments'] / 
        merged_brownfield['Total Brownfield Establishments'] * 100
    )
    
    return merged_greenfield, merged_brownfield


# =============================================================================
# VISUALISATIONS
# =============================================================================

def plot_units_evolution(units_per_year: pd.DataFrame, naf_code: str, save_path: Optional[str] = None):
    """Graphique: Évolution des unités légales par année."""
    plt.figure(figsize=(12, 6))
    
    plt.bar(units_per_year['Year'], units_per_year['Number of Unique Legal Units'], 
            color=COMPANY_PALETTE["forest"])
    
    # Colorer 2025 en gris
    mask_2025 = units_per_year['Year'] == 2025
    if mask_2025.any():
        plt.bar(units_per_year[mask_2025]['Year'], 
                units_per_year[mask_2025]['Number of Unique Legal Units'],
                color=COMPANY_PALETTE["light_grey"])
    
    plt.title('Evolution of the number of new legal entities by creation year, 1980 - September 2025')
    plt.xlabel('')
    plt.xlim(1980, 2026)
    plt.xticks(list(np.arange(1980, 2026, 5)) + [2025],
               labels=[str(int(x)) for x in np.arange(1980, 2026, 5)] + ['2025*'])
    plt.ylabel('Number of new legal entities')
    plt.grid(True)
    
    plt.figtext(0.1, -0.01, 
                f'INSEE company registry (NAF {naf_code}). REKOLT analysis. Data for 2025 are current as of September 30',
                ha='left', fontsize=8, color=COMPANY_PALETTE["grey"], fontstyle='italic')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_establishments_evolution(estab_per_year: pd.DataFrame, naf_code: str, save_path: Optional[str] = None):
    """Graphique: Évolution des établissements par année."""
    plt.figure(figsize=(12, 6))
    
    plt.bar(estab_per_year['Year'], estab_per_year['Number of Unique Establishments'],
            color=COMPANY_PALETTE["muted_green"])
    
    # Colorer 2025 en gris
    mask_2025 = estab_per_year['Year'] == 2025
    if mask_2025.any():
        plt.bar(estab_per_year[mask_2025]['Year'],
                estab_per_year[mask_2025]['Number of Unique Establishments'],
                color=COMPANY_PALETTE["light_grey"])
    
    plt.title('Evolution of the number of unique establishments by creation year, 1980 - September 2025')
    plt.xlabel('')
    plt.xlim(1980, 2026)
    plt.xticks(list(np.arange(1980, 2026, 5)) + [2025],
               labels=[str(int(x)) for x in np.arange(1980, 2026, 5)] + ['2025*'])
    plt.ylabel('Number of unique establishments')
    plt.grid(True)
    
    plt.figtext(0.1, -0.01,
                f'INSEE company registry (NAF {naf_code}). REKOLT analysis. Data for 2025 are current as of September 30, a legal entity can have multiple establishments',
                ha='left', fontsize=8, color=COMPANY_PALETTE["grey"], fontstyle='italic')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_distribution(distribution: pd.DataFrame,
                      naf_code: str,
                      save_path: Optional[str] = None):
    """
    Graphique : Nombre d’unités légales par nombre d’établissements actifs (échelle log).
    (Version avec axe X catégoriel, un seul graphique)
    """
    dist = distribution.copy()

    # Axe X catégoriel
    dist["Number of Establishments"] = dist["Number of Establishments"].astype(str)

    plt.figure(figsize=(12, 6))
    plt.bar(
        dist["Number of Establishments"],
        dist["Number of Unique Legal Units"],
        color=COMPANY_PALETTE["forest"],   # même nom que dans le reste du script
    )

    # Échelle log sur Y pour lisibilité
    plt.yscale("log")
    plt.title(
        "Number of unique legal entities by number of active establishments, September 2025"
    )
    plt.xlabel("Number of active establishments")
    plt.ylabel("Number of unique legal entities (log scale for readability)")

    # Labels au-dessus de chaque barre
    for _, row in dist.iterrows():
        plt.text(
            row["Number of Establishments"],
            row["Number of Unique Legal Units"] * 1.1,
            str(row["Number of Unique Legal Units"]),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

    plt.grid(False)

    # Note de bas de graphique
    plt.figtext(
        0.1,
        -0.01,
        f"INSEE company registry (NAF {naf_code}). REKOLT analysis. "
        "Data for 2025 are current as of September 30, reflecting INSEE database updates.",
        ha="left",
        fontsize=8,
        color=COMPANY_PALETTE["grey"],
        fontstyle="italic",
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

def plot_top_entities(top_entities: pd.DataFrame, naf_code: str, save_path: Optional[str] = None):
    """Graphique: Top 10 entités par SIREN."""
    # Trier par ordre croissant pour affichage
    top_entities = top_entities.sort_values('number_of_establishments', ascending=True)
    
    plt.figure(figsize=(12, 6))
    
    # Wrap SIREN numbers (afficher SIREN)
    top_entities['wrapped_siren'] = top_entities['siren'].apply(
        lambda x: '\n'.join(textwrap.wrap(str(x), 10))
    )
    
    plt.bar(top_entities['wrapped_siren'], top_entities['number_of_establishments'],
            color=COMPANY_PALETTE["muted_green"])
    
    # Data labels
    for i, row in enumerate(top_entities.itertuples()):
        plt.text(i, row.number_of_establishments,
                 str(row.number_of_establishments),
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
    
    # Note explicative
    plt.text(0.1, top_entities['number_of_establishments'].max() * 0.7,
             'Important note:\nSome groups operate multiple locations under a single SIRET,\nactual footprints may be higher.\nFor consistency, we rely on INSEE\'s legal view.\nIt may underestimate field presence for entities with multiple "agences"',
             fontsize=10,
             bbox=dict(facecolor=COMPANY_PALETTE["light_grey"], edgecolor='white'))
    
    plt.title('Top 10 legal entities by number of active establishments, September 2025')
    plt.xlabel('')
    plt.xticks(range(len(top_entities)), top_entities['wrapped_siren'], rotation=0, ha='center')
    plt.ylabel('')
    plt.yticks([])
    plt.grid(False)
    
    plt.figtext(0.1, -0.01,
                f'INSEE company registry (NAF {naf_code}). REKOLT analysis. Data for 2025 are current as of September 30, reflecting INSEE database updates.',
                ha='left', fontsize=8, color=COMPANY_PALETTE["grey"], fontstyle='italic')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()




def plot_greenfield_brownfield(gf_bf: pd.DataFrame, naf_code: str, save_path: Optional[str] = None):
    """Graphique: Greenfield vs Brownfield (barres empilées)."""
    plt.figure(figsize=(12, 6))
    
    # Barres empilées
    plt.bar(gf_bf['Year'], gf_bf['Brownfield Establishments'],
            color=COMPANY_PALETTE["gold_highlight"], label='Brownfield Establishments')
    plt.bar(gf_bf['Year'], gf_bf['Greenfield Establishments'],
            bottom=gf_bf['Brownfield Establishments'],
            color=COMPANY_PALETTE["forest"], label='Greenfield Establishments')
    
    # Colorer 2025 en gris
    mask_2025 = gf_bf['Year'] == 2025
    if mask_2025.any():
        plt.bar(gf_bf[mask_2025]['Year'], gf_bf[mask_2025]['Brownfield Establishments'],
                color=COMPANY_PALETTE["light_grey"])
        plt.bar(gf_bf[mask_2025]['Year'], gf_bf[mask_2025]['Greenfield Establishments'],
                bottom=gf_bf[mask_2025]['Brownfield Establishments'],
                color=COMPANY_PALETTE["light_grey"])
    
    plt.title('Evolution of greenfield vs brownfield establishments by year of creation, 2005 - September 2025')
    plt.xlabel('')
    plt.xlim(2004, 2026)
    plt.xticks(list(np.arange(2005, 2026, 5)) + [2025],
               labels=[str(int(x)) for x in np.arange(2005, 2026, 5)] + ['2025*'])
    plt.ylabel('Number of Establishments')
    plt.grid(True)
    plt.legend()
    
    # Labels avec % greenfield
    for i, row in gf_bf.iterrows():
        plt.text(row['Year'], row['Total Establishments'] / 2,
                 f"{round(row['% Greenfield'])}%",
                 ha='center', color='white', fontsize=10, fontweight='bold')
    
    plt.figtext(0.1, -0.05,
                f'INSEE company registry (NAF {naf_code}). REKOLT analysis. Data for 2025 are current as of September 30.\n'
                'A legal entity can have multiple establishments. Greenfield establishments are created by newly registered legal entities,\n'
                'while brownfield establishments are created by already existing legal entities.',
                ha='left', fontsize=8, color=COMPANY_PALETTE["grey"], fontstyle='italic')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_survival_by_duration(merged_greenfield: pd.DataFrame, merged_brownfield: pd.DataFrame,
                              naf_code: str, save_path: Optional[str] = None):
    """Graphique: Taux de survie par durée d'existence (greenfield vs brownfield)."""
    plt.figure(figsize=(14, 7))
    
    # Ligne Greenfield
    plt.plot(merged_greenfield['Duration'], merged_greenfield['Percentage Active Greenfield'],
             color=COMPANY_PALETTE["forest"], marker='o', label='Greenfield Establishments')
    
    # Ligne Brownfield
    plt.plot(merged_brownfield['Duration'], merged_brownfield['Percentage Active Brownfield'],
             color=COMPANY_PALETTE["gold_highlight"], marker='o', label='Brownfield Establishments')
    
    # Zone grisée pour >25 ans
    plt.axvspan(25, 65, color=COMPANY_PALETTE["grey"], alpha=0.3)
    
    plt.title('Percentage of Active Establishments by Duration of Existence, 1980 - September 2025')
    plt.xlabel('Duration of Existence (Years)')
    plt.ylabel('% Active Establishments')
    plt.xlim(0, 45)
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def run_analysis(
    filepath: str,
    naf_code: str = "81.30Z",
    output_dir: Optional[str] = None
):
    """
    Exécute toutes les analyses et génère les graphiques.
    
    Parameters
    ----------
    filepath : str
        Chemin vers le fichier CSV des données
    naf_code : str
        Code NAF pour filtrer
    output_dir : str, optional
        Répertoire de sortie pour les graphiques
    """
    # Setup
    setup_style()
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None
    
    # Chargement des données
    print("\n" + "="*60)
    print(f"ANALYSE DU SECTEUR NAF {naf_code}")
    print("="*60)
    
    data = load_and_clean_data(filepath, naf_code)
    
    # 1. Évolution unités légales
    print("\n--- Évolution des unités légales ---")
    units_per_year = analyze_units_per_year(data)
    save_path = output_path / "01_units_evolution.png" if output_path else None
    plot_units_evolution(units_per_year, naf_code, save_path)
    
    # 2. Évolution établissements
    print("\n--- Évolution des établissements ---")
    estab_per_year = analyze_establishments_per_year(data)
    save_path = output_path / "02_establishments_evolution.png" if output_path else None
    plot_establishments_evolution(estab_per_year, naf_code, save_path)
    
    # 3. Distribution par entité
    print("\n--- Distribution établissements/entité ---")
    distribution = analyze_distribution_by_entity(data)
    save_path = output_path / "03_distribution.png" if output_path else None
    plot_distribution(distribution, naf_code, save_path)
    
    # 4. Top 10 entités (par SIREN)
    print("\n--- Top 10 entités ---")
    top10 = analyze_top_entities(data, n=10)
    print(top10.to_string(index=False))
    save_path = output_path / "04_top10_entities.png" if output_path else None
    plot_top_entities(top10, naf_code, save_path)
    
    # 5. Distribution excluant top 10
    #print("\n--- Distribution (excluant top 10) ---")
    #dist_excl = analyze_entities_excluding_top(data, top10['siren'].tolist())
    #save_path = output_path / "05_distribution_excl_top10.png" if output_path else None
    
    #plot_distribution_excluding_top(dist_excl, naf_code, save_path)
    
    # 6. Greenfield vs Brownfield
    print("\n--- Greenfield vs Brownfield ---")
    gf_bf = analyze_greenfield_brownfield(data, start_year=2005)
    save_path = output_path / "06_greenfield_brownfield.png" if output_path else None
    plot_greenfield_brownfield(gf_bf, naf_code, save_path)
    
    # 7. Taux de survie par durée
    print("\n--- Taux de survie par durée ---")
    merged_gf, merged_bf = analyze_survival_by_duration(data)
    save_path = output_path / "07_survival_by_duration.png" if output_path else None
    plot_survival_by_duration(merged_gf, merged_bf, naf_code, save_path)
    
    # Export CSV
    if output_path:
        units_per_year.to_csv(output_path / "units_per_year.csv", index=False)
        estab_per_year.to_csv(output_path / "establishments_per_year.csv", index=False)
        distribution.to_csv(output_path / "distribution.csv", index=False)
        top10.to_csv(output_path / "top10_entities.csv", index=False)
        gf_bf.to_csv(output_path / "greenfield_brownfield.csv", index=False)
        merged_gf.to_csv(output_path / "survival_greenfield.csv", index=False)
        merged_bf.to_csv(output_path / "survival_brownfield.csv", index=False)
        print(f"\n✅ Résultats exportés dans {output_path}")
    
    print("\n" + "="*60)
    print("Analyse terminée!")
    print("="*60)
    
    return {
        'data': data,
        'units_per_year': units_per_year,
        'establishments_per_year': estab_per_year,
        'distribution': distribution,
        'top10': top10,
        'greenfield_brownfield': gf_bf,
        'survival_greenfield': merged_gf,
        'survival_brownfield': merged_bf
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyse des entités du secteur')
    parser.add_argument('filepath', help='Chemin vers le fichier CSV')
    parser.add_argument('--naf', default='81.30Z', help='Code NAF (default: 81.30Z)')
    parser.add_argument('--output', '-o', default=None, help='Répertoire de sortie')
    
    args = parser.parse_args()
    
    run_analysis(args.filepath, args.naf, args.output)
