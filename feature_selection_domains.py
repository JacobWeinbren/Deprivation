import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Config
CENSUS_2011 = "2011_census_master_lsoa.csv"
IMD_2019 = "Dep/2019.csv"
IMD_2025 = "Dep/2025.csv"
LOOKUP = "LSOA_2011_2021_lookup.csv"


def load_data_optimized(target_domain_col: str):
    """Load data with smart feature selection based on domain relationships"""
    print(f"Loading data for: {target_domain_col}")

    # Load 2025 target
    imd25 = pd.read_csv(IMD_2025)
    if target_domain_col not in imd25.columns:
        print(f"Error: Target column '{target_domain_col}' not in 2025 data")
        return pd.DataFrame()

    imd25 = imd25[["LSOA code (2021)", target_domain_col]]
    imd25.columns = ["LSOA21CD", "TARGET_2025"]

    # Load lookup - only exact matches for training
    lookup = pd.read_csv(LOOKUP)
    exact_match = lookup[lookup["overlap_pct"] == 100].copy()
    print(f"Found {len(exact_match)} LSOAs with exact boundary match")

    train_data = exact_match.merge(imd25, on="LSOA21CD")

    # Load census
    census = pd.read_csv(CENSUS_2011, index_col="LSOA11CD")

    # Load 2019 IMD - SMART LOADING
    imd19 = pd.read_csv(IMD_2019)

    # Remove any duplicate columns from source
    if imd19.columns.duplicated().any():
        print(f"Warning: Removing {imd19.columns.duplicated().sum()} duplicate columns")
        imd19 = imd19.loc[:, ~imd19.columns.duplicated(keep="first")]

    # Identify key columns to load based on domain
    is_subdomain = "Sub-domain" in target_domain_col

    # Core columns always needed
    cols_to_load = ["LSOA code (2011)", target_domain_col]

    # CRITICAL: Always include main IMD score as it's highly predictive
    main_imd_col = "Index of Multiple Deprivation (IMD) Score"
    if main_imd_col in imd19.columns and target_domain_col != main_imd_col:
        cols_to_load.append(main_imd_col)

    # For subdomains, include the parent domain
    parent_domain_map = {
        "Children and Young People Sub-domain Score": "Education, Skills and Training Score",
        "Adult Skills Sub-domain Score": "Education, Skills and Training Score",
        "Geographical Barriers Sub-domain Score": "Barriers to Housing and Services Score",
        "Wider Barriers Sub-domain Score": "Barriers to Housing and Services Score",
        "Indoors Sub-domain Score": "Living Environment Score",
        "Outdoors Sub-domain Score": "Living Environment Score",
    }

    if target_domain_col in parent_domain_map:
        parent_col = parent_domain_map[target_domain_col]
        if parent_col in imd19.columns:
            cols_to_load.append(parent_col)
            print(f"Including parent domain: {parent_col}")

    # Add top correlated domains
    score_cols = [col for col in imd19.columns if "Score" in col or "(rate)" in col]
    score_data = imd19[score_cols].select_dtypes(include=[np.number])

    if target_domain_col in score_data.columns:
        corr_matrix = score_data.corr()
        target_corrs = corr_matrix[target_domain_col].abs().sort_values(ascending=False)
        # Get top 5 related domains (excluding self)
        related_domains = target_corrs.iloc[1:6].index.tolist()
        cols_to_load.extend([col for col in related_domains if col not in cols_to_load])
        print(f"Including correlated domains: {related_domains[:3]}")

    # Also add ranks for important scores
    for col in cols_to_load[1:]:  # Skip LSOA code
        rank_col = col.replace("Score", "Rank").replace("(rate)", "Rank")
        if rank_col in imd19.columns:
            cols_to_load.append(rank_col)

    # Remove duplicates while preserving order
    cols_to_load = list(dict.fromkeys(cols_to_load))

    # Load selected columns
    available_cols = [col for col in cols_to_load if col in imd19.columns]
    imd19 = imd19[available_cols].copy()

    # Rename columns for clarity
    rename_dict = {"LSOA code (2011)": "LSOA11CD", target_domain_col: "DOMAIN_2019"}

    # Keep other domain scores with clear names
    for col in imd19.columns:
        if col not in rename_dict:
            if col == main_imd_col:
                rename_dict[col] = "IMD_2019"
            elif "Score" in col or "(rate)" in col:
                # Shorten but keep clear
                if "Income Deprivation Affecting Children" in col:
                    rename_dict[col] = "IDACI_2019"
                elif "Income Deprivation Affecting Older" in col:
                    rename_dict[col] = "IDAOPI_2019"
                elif "Income" in col and "(rate)" in col:
                    rename_dict[col] = "Income_2019"
                elif "Employment" in col and "(rate)" in col:
                    rename_dict[col] = "Employment_2019"
                elif col in parent_domain_map.values():
                    rename_dict[col] = "Parent_Domain_2019"
            elif "Rank" in col:
                # Keep ranks with clear suffix
                if "Index of Multiple Deprivation" in col:
                    rename_dict[col] = "IMD_Rank_2019"
                elif (
                    target_domain_col.replace("Score", "Rank").replace("(rate)", "Rank")
                    == col
                ):
                    rename_dict[col] = "DOMAIN_Rank_2019"

    imd19.rename(columns=rename_dict, inplace=True)

    # Merge everything
    train_data = train_data.merge(census, left_on="LSOA11CD", right_index=True)
    train_data = train_data.merge(imd19, on="LSOA11CD")

    print(
        f"Final training data: {len(train_data)} LSOAs with {len(train_data.columns)} features"
    )

    return train_data.dropna()


def engineer_features_organic(data):
    """
    Applies a theory-free, organic transformation to all census data.
    1. Log-transforms all census columns to normalize skew.
    2. Standardizes all census columns to make them comparable.
    3. Creates new aggregate features (mean, std) from the entire census.
    """
    print("Running organic feature engineering...")

    # Separate census data from IMD data
    census_cols = [col for col in data.columns if col.startswith(("KS", "QS"))]
    if not census_cols:
        print("No census columns found to engineer.")
        return data

    census_data = data[census_cols].clip(lower=0)  # Ensure no negatives

    # 1. Log Transform (np.log1p handles 0s safely)
    print(f"Applying log-transform to {len(census_cols)} census features...")
    census_log = np.log1p(census_data)

    # 2. Standardize
    print("Standardizing all census features (Mean=0, Std=1)...")
    scaler = StandardScaler()
    # Create a new DataFrame with the scaled data, preserving columns and index
    census_scaled = pd.DataFrame(
        scaler.fit_transform(census_log), columns=census_cols, index=data.index
    )

    # 3. Create Aggregate Features (Theory-free)
    print("Creating organic aggregate features (Census_Agg_Mean, Census_Agg_Std)...")
    data["Census_Agg_Mean"] = census_scaled.mean(axis=1)
    data["Census_Agg_Std"] = census_scaled.std(axis=1)

    # Replace the old raw census columns with the new scaled ones
    data_pre_merge = data.drop(columns=census_cols)
    data = pd.concat([data_pre_merge, census_scaled], axis=1)

    print(f"Organic engineering complete. New feature count: {len(data.columns)}")
    return data.fillna(0)  # Fill any NaNs from std() on a single-feature LSOA


def select_features_optimized(data, short_name, n_features=100):
    """Optimized feature selection combining multiple methods"""

    # Apply smart feature engineering
    data = engineer_features_organic(data)

    # Prepare features and target
    y = data["TARGET_2025"]
    exclude = ["LSOA11CD", "LSOA21CD", "TARGET_2025", "overlap_pct"]
    X = data.drop(columns=exclude)

    # Handle any NaN/Inf from engineering
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"Total features available: {len(X.columns)}")

    # CRITICAL: Ensure DOMAIN_2019 is always included
    critical_features = ["DOMAIN_2019"]
    if "DOMAIN_2019_log" in X.columns:
        critical_features.append("DOMAIN_2019_log")
    if "IMD_2019" in X.columns and "IMD" not in short_name:
        critical_features.append("IMD_2019")
    if "Parent_Domain_2019" in X.columns:
        critical_features.append("Parent_Domain_2019")

    # 1. Random Forest Importance (captures non-linear relationships)
    print("Calculating Random Forest importance...")
    rf = RandomForestRegressor(
        n_estimators=300,  # More trees for stability
        max_depth=25,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns)

    # 2. Univariate F-scores (captures linear relationships)
    print("Calculating F-scores...")
    f_selector = SelectKBest(f_regression, k=min(150, len(X.columns)))
    f_selector.fit(X, y)
    f_scores = pd.Series(f_selector.scores_, index=X.columns).fillna(0)

    # 3. Mutual Information (captures any dependencies)
    print("Calculating Mutual Information...")
    mi_scores = mutual_info_regression(X, y, random_state=42, n_neighbors=10)
    mi_scores = pd.Series(mi_scores, index=X.columns)

    # 4. Correlation with target (simple but effective)
    correlations = X.corrwith(y).abs().fillna(0)

    # Normalize all scores to 0-1
    rf_norm = (
        rf_importance / rf_importance.max()
        if rf_importance.max() > 0
        else rf_importance
    )
    f_norm = f_scores / f_scores.max() if f_scores.max() > 0 else f_scores
    mi_norm = mi_scores / mi_scores.max() if mi_scores.max() > 0 else mi_scores
    corr_norm = (
        correlations / correlations.max() if correlations.max() > 0 else correlations
    )

    # Weighted ensemble of methods (RF gets highest weight as it works best)
    combined_scores = (
        0.40 * rf_norm  # Random Forest captures complex patterns
        + 0.25 * f_norm  # F-scores for linear relationships
        + 0.20 * mi_norm  # Mutual info for any dependencies
        + 0.15 * corr_norm  # Simple correlation as baseline
    ).sort_values(ascending=False)

    # Feature selection with correlation filtering
    selected = []

    # Always include critical features first
    for feat in critical_features:
        if feat in combined_scores.index:
            selected.append(feat)

    # Get correlation matrix for filtering
    X_scaled = StandardScaler().fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    corr_matrix = X_scaled.corr().abs()

    # Add remaining features
    for feat in combined_scores.index:
        if feat not in selected:
            # Check correlation with already selected features
            if selected:
                max_corr = max(
                    [
                        corr_matrix.loc[feat, sel]
                        for sel in selected
                        if sel in corr_matrix.columns
                    ]
                )
                # Only add if not highly correlated (threshold 0.90)
                if max_corr < 0.90 or feat in critical_features:
                    selected.append(feat)
            else:
                selected.append(feat)

            if len(selected) >= n_features:
                break

    # Print top features
    print(f"\nTop 10 features for {short_name}:")
    for i, feat in enumerate(selected[:10]):
        score = combined_scores.get(feat, 0)
        print(f"  {i + 1:2}. {feat:30} Score={score:.4f}")

    # Validate selection with quick model
    print("\nValidating feature selection...")
    rf_test = RandomForestRegressor(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
    )
    rf_test.fit(X[selected], y)
    pred = rf_test.predict(X[selected])
    r2 = r2_score(y, pred)
    print(f"Training R² with {len(selected)} features: {r2:.4f}")

    # Save features
    output_filename = f"selected_features_{short_name}.txt"
    with open(output_filename, "w") as f:
        for feat in selected:
            f.write(f"{feat}\n")

    print(f"✓ Saved {len(selected)} features to {output_filename}")

    return selected, output_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimized feature selection for IMD domain"
    )
    parser.add_argument(
        "--domain_col", type=str, required=True, help="Exact column name of the domain"
    )
    parser.add_argument(
        "--short_name", type=str, required=True, help="Short name for output files"
    )
    parser.add_argument(
        "--n_features",
        type=int,
        default=100,
        help="Number of features to select (default: 100)",
    )
    args = parser.parse_args()

    # Load data
    data = load_data_optimized(args.domain_col)

    if not data.empty:
        features, filename = select_features_optimized(
            data, args.short_name, args.n_features
        )
    else:
        print("Failed to load data. Exiting.")
