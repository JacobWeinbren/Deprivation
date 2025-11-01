#!/usr/bin/env python3
"""
Combines all IMD domain scores from 2010, 2015, 2019, and 2025
into a single GeoJSON file based on 2011 LSOA boundaries.

*** UPDATE: This version performs a robust trend and consistency analysis:
1.  Calculates z-scores for all scores (2010-2025) to make years comparable.
2.  Creates an "Average Z-Score" (Consistency) metric. A hotspot on this
    means "consistently deprived".
3.  Calculates a "Trend" metric (slope of z-scores over time). A coldspot
    on this means "consistently improving".
4.  Runs hotspot analysis on these 32 new (16 consistency, 16 trend) metrics.
5.  CLEANS UP intermediate z-score columns before saving for a smaller file.
***
"""

import pandas as pd
import geopandas as gpd
import os
import warnings

# --- LIBRARIES FOR HOTSPOT & TREND ANALYSIS ---
from pysal.lib import weights
from esda.moran import Moran_Local
import numpy as np

# Suppress warnings from pysal/esda
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---

# 1. File Paths
SHP_2011_FILE = "LSOA/2011/LSOA_2011_EW_BFC_V3.shp"
LOOKUP_FILE = "LSOA_2011_2021_lookup.csv"
IMD_2010_FILE = "Dep/2010.csv"
IMD_2015_FILE = "Dep/2015.csv"
IMD_2019_FILE = "Dep/2019.csv"
IMD_2025_SOURCE_FILE = "Dep/2025.csv"
PREDICTIONS_DIR = "."  # Directory where IMD_2025_*.csv files are
OUTPUT_GEOJSON = (
    "IMD_Domains_2010-2025_England_Trend_Hotspots.geojson"  # Renamed output
)

# 2. Domain Mapping
# (No changes to this section)
DOMAIN_CONFIG = [
    ("IMD", "IMD SCORE", "Index of Multiple Deprivation (IMD) Score"),
    ("Income", "INCOME SCORE", "Income Score (rate)"),
    ("Employment", "EMPLOYMENT SCORE", "Employment Score (rate)"),
    (
        "Education",
        "EDUCATION SKILLS AND TRAINING SCORE",
        "Education, Skills and Training Score",
    ),
    (
        "Health",
        "HEALTH DEPRIVATION AND DISABILITY SCORE",
        "Health Deprivation and Disability Score",
    ),
    ("Crime", "CRIME AND DISORDER SCORE", "Crime Score"),
    (
        "Barriers",
        "BARRIERS TO HOUSING AND SERVICES SCORE",
        "Barriers to Housing and Services Score",
    ),
    ("LivingEnv", "LIVING ENVIRONMENT SCORE", "Living Environment Score"),
    (
        "IDACI",
        "IDACI score",
        "Income Deprivation Affecting Children Index (IDACI) Score (rate)",
    ),
    (
        "IDAOPI",
        "IDAOPI score",
        "Income Deprivation Affecting Older People (IDAOPI) Score (rate)",
    ),
    (
        "CYP",
        "Children/Young People Sub-domain Score",
        "Children and Young People Sub-domain Score",
    ),
    ("AdultSkills", "Skills Sub-domain Score", "Adult Skills Sub-domain Score"),
    (
        "GeoBarriers",
        "Geographical Barriers Sub-domain Score",
        "Geographical Barriers Sub-domain Score",
    ),
    (
        "WiderBarriers",
        "Wider Barriers Sub-domain Score",
        "Wider Barriers Sub-domain Score",
    ),
    ("Indoors", "Indoors Sub-domain Score", "Indoors Sub-domain Score"),
    ("Outdoors", "Outdoors Sub-domain Score", "Outdoors Sub-domain Score"),
]

# --- Helper Functions ---


def load_deprivation_data(filepath, lsoa_col, year_suffix, col_map):
    """Generic function to load a deprivation CSV."""
    print(f"Loading {year_suffix} data from {filepath}...")
    try:
        df = pd.read_csv(filepath, encoding="latin1")
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return pd.DataFrame()
    rename_cols = {lsoa_col: "LSOA11CD"}
    use_cols = [lsoa_col]
    for short_name, col_name in col_map.items():
        if col_name in df.columns:
            rename_cols[col_name] = f"{short_name}_{year_suffix}"
            use_cols.append(col_name)
        else:
            print(f"  Warning: Column '{col_name}' not found in {year_suffix} data.")
    df = df[use_cols].rename(columns=rename_cols)
    print(f"  Loaded {len(df)} rows.")
    return df


def load_2025_hybrid(unchanged_11_codes):
    """Loads the 2025 data using the hybrid (source + prediction) approach."""
    print(f"Creating hybrid 2025 dataset...")
    col_map_2025 = {short: col_other for short, _, col_other in DOMAIN_CONFIG}
    print(f"Loading 2025 SOURCE data from {IMD_2025_SOURCE_FILE}...")
    try:
        df_source = pd.read_csv(IMD_2025_SOURCE_FILE)
    except Exception as e:
        print(f"  Error: Could not load 2025 source data: {e}")
        return pd.DataFrame()
    rename_cols = {"LSOA code (2021)": "LSOA11CD"}
    use_cols = ["LSOA code (2021)"]
    for short_name, col_name in col_map_2025.items():
        if col_name in df_source.columns:
            rename_cols[col_name] = f"{short_name}_2025"
            use_cols.append(col_name)
    df_source = df_source[use_cols].rename(columns=rename_cols)
    df_source_unchanged = df_source[
        df_source["LSOA11CD"].isin(unchanged_11_codes)
    ].copy()
    print(f"  Found {len(df_source_unchanged)} unchanged LSOAs in 2025 source file.")
    print(f"Loading 2025 PREDICTED data from '{PREDICTIONS_DIR}'...")
    all_preds = pd.DataFrame()
    for short_name, _, _ in DOMAIN_CONFIG:
        pred_file = os.path.join(
            PREDICTIONS_DIR, f"IMD_2025_{short_name}_2011_boundaries.csv"
        )
        low_r2_file = os.path.join(PREDICTIONS_DIR, f"IMD_2025_{short_name}_LOW_R2.csv")
        file_to_load = None
        is_low_r2 = False
        if os.path.exists(pred_file):
            file_to_load = pred_file
        elif os.path.exists(low_r2_file):
            file_to_load = low_r2_file
            is_low_r2 = True
            print(
                f"  Warning: Using LOW R2 prediction file for {short_name}: {low_r2_file}"
            )
        if file_to_load:
            try:
                pred_df = pd.read_csv(file_to_load)
                final_col = f"{short_name}_2025"
                if is_low_r2:
                    if "prediction" in pred_df.columns:
                        pred_df = pred_df[["LSOA11CD", "prediction"]].rename(
                            columns={"prediction": final_col}
                        )
                    else:
                        print(
                            f"  Error: 'prediction' column not found in {file_to_load}"
                        )
                        continue
                else:
                    pred_col = f"{short_name}_2025_pred"
                    if pred_col in pred_df.columns:
                        pred_df = pred_df[["LSOA11CD", pred_col]].rename(
                            columns={pred_col: final_col}
                        )
                    else:
                        print(
                            f"  Error: Prediction column '{pred_col}' not found in {file_to_load}"
                        )
                        continue
                if all_preds.empty:
                    all_preds = pred_df
                else:
                    all_preds = all_preds.merge(pred_df, on="LSOA11CD", how="outer")
            except Exception as e:
                print(f"  Error loading prediction file {file_to_load}: {e}")
        else:
            print(f"  Warning: No prediction file found for {short_name}. Skipping.")
    if all_preds.empty:
        print("  Error: No 2025 prediction files were found or loaded.")
        return df_source_unchanged
    df_preds_changed = all_preds[~all_preds["LSOA11CD"].isin(unchanged_11_codes)].copy()
    print(f"  Found {len(df_preds_changed)} changed LSOAs in prediction files.")
    df_2025_final = pd.concat(
        [df_source_unchanged, df_preds_changed], ignore_index=True
    )
    print(f"  Total 2025 hybrid dataset has {len(df_2025_final)} rows.")
    return df_2025_final


def add_hotspot_analysis(gdf, columns_to_analyze):
    """
    Performs a Local Moran's I (LISA) analysis on a specific list of columns.
    Classifies Hotspots and Coldspots by confidence level (99%, 95%, 90%).
    """
    print("\nRunning Hotspot/Coldspot Analysis (LISA)...")
    print("  1. Building spatial weights matrix (this may take a moment)...")
    gdf = gdf.reset_index(drop=True)
    try:
        w = weights.Queen.from_dataframe(gdf, silence_warnings=True)
    except Exception as e:
        print(f"  ERROR: Could not build spatial weights: {e}")
        print("  This can happen with invalid geometries. Skipping hotspot analysis.")
        return gdf
    print(f"  2. Calculating LISA for {len(columns_to_analyze)} specified columns...")
    for i, col in enumerate(columns_to_analyze):
        if col not in gdf.columns:
            print(f"     - Warning: Column '{col}' not found. Skipping.")
            continue
        print(f"     - Processing {col} ({i + 1} of {len(columns_to_analyze)})...")
        y = gdf[col].fillna(gdf[col].mean())
        lisa = Moran_Local(y, w, permutations=999, seed=12345)
        new_col_name = f"{col}_HS"
        quadrant = lisa.q
        p_value = lisa.p_sim
        gdf[new_col_name] = "NS"
        is_hotspot_cluster = quadrant == 1  # HH (High-High)
        is_coldspot_cluster = quadrant == 3  # LL (Low-Low)
        p_lt_0_01 = p_value < 0.01  # 99% confidence
        p_lt_0_05 = p_value < 0.05  # 95% confidence
        p_lt_0_10 = p_value < 0.10  # 90% confidence
        gdf.loc[is_hotspot_cluster & p_lt_0_01, new_col_name] = "Hotspot_99"
        gdf.loc[is_coldspot_cluster & p_lt_0_01, new_col_name] = "Coldspot_99"
        gdf.loc[is_hotspot_cluster & p_lt_0_05 & (p_value >= 0.01), new_col_name] = (
            "Hotspot_95"
        )
        gdf.loc[is_coldspot_cluster & p_lt_0_05 & (p_value >= 0.01), new_col_name] = (
            "Coldspot_95"
        )
        gdf.loc[is_hotspot_cluster & p_lt_0_10 & (p_value >= 0.05), new_col_name] = (
            "Hotspot_90"
        )
        gdf.loc[is_coldspot_cluster & p_lt_0_10 & (p_value >= 0.05), new_col_name] = (
            "Coldspot_90"
        )
    print("  3. Hotspot analysis complete.")
    return gdf


def calculate_trend(row, z_cols, x_axis):
    """
    Calculates the slope of a linear regression for a single row.
    """
    y_axis = row[z_cols].values.astype(float)

    if np.isnan(y_axis).any():
        return np.nan

    try:
        slope = np.polyfit(x_axis, y_axis, 1)[0]
        return slope
    except Exception:
        return np.nan


# --- Main Execution ---


def main():
    print("Starting GeoJSON creation process...")

    # Load 2011 Base Geometry
    print(f"Loading base 2011 geometry from {SHP_2011_FILE}...")
    try:
        gdf = gpd.read_file(SHP_2011_FILE)
        gdf = gdf[["LSOA11CD", "geometry"]]
    except Exception as e:
        print(f"  FATAL: Could not load base shapefile: {e}")
        return
    print(f"  Loaded {len(gdf)} 2011 LSOA geometries (England & Wales).")

    # Filter to England-only
    gdf = gdf[gdf["LSOA11CD"].str.startswith("E01")].copy()
    print(f"  Filtered to {len(gdf)} England-only LSOAs.")

    # Load 2011-2021 Lookup
    print(f"Loading LSOA lookup from {LOOKUP_FILE}...")
    try:
        lookup = pd.read_csv(LOOKUP_FILE)
        unchanged_11_codes = lookup[lookup["overlap_pct"] == 100]["LSOA11CD"].unique()
        print(f"  Found {len(unchanged_11_codes)} LSOAs with 100% overlap.")
    except Exception as e:
        print(f"  FATAL: Could not load lookup file: {e}")
        return

    # Load 2010, 2015, 2019, 2025 Data
    col_map_2010 = {short: col_2010 for short, col_2010, _ in DOMAIN_CONFIG}
    data_2010 = load_deprivation_data(IMD_2010_FILE, "LSOA CODE", "2010", col_map_2010)
    col_map_2015 = {short: col_other for short, _, col_other in DOMAIN_CONFIG}
    data_2015 = load_deprivation_data(
        IMD_2015_FILE, "LSOA code (2011)", "2015", col_map_2015
    )
    col_map_2019 = {short: col_other for short, _, col_other in DOMAIN_CONFIG}
    data_2019 = load_deprivation_data(
        IMD_2019_FILE, "LSOA code (2011)", "2019", col_map_2019
    )
    data_2025 = load_2025_hybrid(unchanged_11_codes)

    # --- Merge all data into the GeoDataFrame ---
    print("\nMerging all datasets...")
    if not data_2010.empty:
        gdf = gdf.merge(data_2010, on="LSOA11CD", how="left")
    if not data_2015.empty:
        gdf = gdf.merge(data_2015, on="LSOA11CD", how="left")
    if not data_2019.empty:
        gdf = gdf.merge(data_2019, on="LSOA11CD", how="left")
    if not data_2025.empty:
        gdf = gdf.merge(data_2025, on="LSOA11CD", how="left")

    # --- CALCULATE TREND AND CONSISTENCY METRICS ---
    print("\nCalculating Trend and Consistency (Z-Score) metrics...")

    x_axis_years = [2010, 2015, 2019, 2025]

    consistency_cols_to_analyze = []  # e.g., IMD_Avg_z
    trend_cols_to_analyze = []  # e.g., IMD_Trend
    cols_to_drop = []  # To store z-score column names

    for short_name, _, _ in DOMAIN_CONFIG:
        print(f"  - Processing {short_name}...")

        # 1. Define column names
        cols_raw = [f"{short_name}_{y}" for y in ["2010", "2015", "2019", "2025"]]
        cols_z = [f"{c}_z" for c in cols_raw]
        cols_to_drop.extend(cols_z)  # Add them to the drop list

        if not all(c in gdf.columns for c in cols_raw):
            print(f"    Skipping {short_name}: Missing one or more year's data.")
            continue

        # 2. Standardize (calculate z-scores for each year)
        for i, col in enumerate(cols_raw):
            z_col_name = cols_z[i]
            mean = gdf[col].mean()
            std = gdf[col].std()
            if std == 0:
                gdf[z_col_name] = 0.0
            else:
                gdf[z_col_name] = (gdf[col] - mean) / std

        # 3. Calculate "Consistency" (Average Z-Score)
        consistency_col = f"{short_name}_Avg_z"
        gdf[consistency_col] = gdf[cols_z].mean(axis=1)
        consistency_cols_to_analyze.append(consistency_col)

        # 4. Calculate "Trend" (Slope of Z-Scores over time)
        trend_col = f"{short_name}_Trend"
        gdf[trend_col] = gdf.apply(calculate_trend, args=(cols_z, x_axis_years), axis=1)
        trend_cols_to_analyze.append(trend_col)

    print(
        f"  Created {len(consistency_cols_to_analyze)} 'Consistency' columns (e.g., 'IMD_Avg_z')."
    )
    print(
        f"  Created {len(trend_cols_to_analyze)} 'Trend' columns (e.g., 'IMD_Trend')."
    )

    # --- Run the Hotspot Analysis ---
    all_cols_to_analyze = consistency_cols_to_analyze + trend_cols_to_analyze
    gdf = add_hotspot_analysis(gdf, all_cols_to_analyze)

    # --- !!! NEW: FINAL CLEANUP STEP !!! ---
    # Drop the intermediate z-score columns
    cols_to_drop_existing = [col for col in cols_to_drop if col in gdf.columns]
    gdf = gdf.drop(columns=cols_to_drop_existing)
    print(f"\nCleaned up {len(cols_to_drop_existing)} intermediate z-score columns.")
    # --- END OF NEW STEP ---

    # Clean up columns for a tidy GeoJSON
    geo_cols = ["LSOA11CD", "geometry"]
    data_cols = sorted([col for col in gdf.columns if col not in geo_cols])
    gdf = gdf[geo_cols + data_cols]

    # --- Save to GeoJSON ---
    print(f"\nSaving final merged data to {OUTPUT_GEOJSON}...")
    try:
        gdf.to_file(OUTPUT_GEOJSON, driver="GeoJSON")

        num_data_cols = len(data_cols)
        num_hotspot_cols = len([col for col in data_cols if col.endswith("_HS")])
        num_raw_cols = len(
            [
                c
                for c in data_cols
                if any(y in c for y in ["2010", "2015", "2019", "2025"])
                and "_z" not in c
            ]
        )
        num_analysis_cols = num_data_cols - num_hotspot_cols - num_raw_cols

        print("\n" + "=" * 50)
        print(f"✅ Success! GeoJSON file created:")
        print(f"   {os.path.abspath(OUTPUT_GEOJSON)}")
        print(f"   Total LSOAs: {len(gdf)} (England-only)")
        print(f"   Total data columns: {num_data_cols}")
        print(f"   (Includes {num_raw_cols} raw score columns from 2010-2025)")
        print(f"   (Includes {num_analysis_cols} 'Consistency' and 'Trend' columns)")
        print(f"   (Includes {num_hotspot_cols} focused hotspot columns)")
        print("=" * 50)
    except Exception as e:
        print(f"  FATAL: Could not save GeoJSON file: {e}")


if __name__ == "__main__":
    main()
