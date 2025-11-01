import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, KFold, train_test_split
import geopandas as gpd
import warnings
import matplotlib.pyplot as plt
import argparse
import time
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Config
CENSUS_FILE = "2011_census_master_lsoa.csv"
IMD_2019_FILE = "Dep/2019.csv"
IMD_2025_FILE = "Dep/2025.csv"
LOOKUP_FILE = "LSOA_2011_2021_lookup.csv"
LSOA_2011_SHP = "LSOA/2011/LSOA_2011_EW_BFC_V3.shp"


def load_features(short_name: str) -> list:
    """Loads the feature list from a file."""
    features_file = f"selected_features_{short_name}.txt"
    try:
        with open(features_file) as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        print(f"Error: Feature file not found at {features_file}")
        return []


def prepare_data_optimized(features: list, target_domain_col: str):
    """Prepares data with same feature engineering as feature selection"""
    print(f"Preparing data for: {target_domain_col}")

    # Load census
    census = pd.read_csv(CENSUS_FILE, index_col="LSOA11CD")

    # Load 2019 IMD with smart column selection
    imd19 = pd.read_csv(IMD_2019_FILE)

    # Remove duplicates from source
    if imd19.columns.duplicated().any():
        print(f"Removing {imd19.columns.duplicated().sum()} duplicate columns")
        imd19 = imd19.loc[:, ~imd19.columns.duplicated(keep="first")]

    # Determine columns to load (matching feature_selection logic)
    cols_to_load = ["LSOA code (2011)", target_domain_col]

    # Always include main IMD
    main_imd_col = "Index of Multiple Deprivation (IMD) Score"
    if main_imd_col in imd19.columns and target_domain_col != main_imd_col:
        cols_to_load.append(main_imd_col)

    # Include parent domain for subdomains
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

    # Add correlated domains
    score_cols = [col for col in imd19.columns if "Score" in col or "(rate)" in col]
    score_data = imd19[score_cols].select_dtypes(include=[np.number])

    if target_domain_col in score_data.columns:
        corr_matrix = score_data.corr()
        target_corrs = corr_matrix[target_domain_col].abs().sort_values(ascending=False)
        related_domains = target_corrs.iloc[1:6].index.tolist()
        cols_to_load.extend([col for col in related_domains if col not in cols_to_load])
        print(f"Including related domains: {related_domains[:3]}")

    # Add ranks
    for col in cols_to_load[1:]:
        rank_col = col.replace("Score", "Rank").replace("(rate)", "Rank")
        if rank_col in imd19.columns:
            cols_to_load.append(rank_col)

    # Remove duplicates
    cols_to_load = list(dict.fromkeys(cols_to_load))
    available_cols = [col for col in cols_to_load if col in imd19.columns]
    imd19 = imd19[available_cols].copy()

    # Rename columns (matching feature_selection)
    rename_dict = {"LSOA code (2011)": "LSOA11CD", target_domain_col: "DOMAIN_2019"}

    for col in imd19.columns:
        if col not in rename_dict:
            if col == main_imd_col:
                rename_dict[col] = "IMD_2019"
            elif "Income Deprivation Affecting Children" in col:
                rename_dict[col] = "IDACI_2019"
            elif "Income Deprivation Affecting Older" in col:
                rename_dict[col] = "IDAOPI_2019"
            elif "Income" in col and "(rate)" in col:
                rename_dict[col] = "Income_2019"
            elif "Employment" in col and "(rate)" in col:
                rename_dict[col] = "Employment_2019"
            elif col in parent_domain_map.values():
                rename_dict[col] = "Parent_Domain_2019"
            elif "Index of Multiple Deprivation" in col and "Rank" in col:
                rename_dict[col] = "IMD_Rank_2019"
            elif (
                target_domain_col.replace("Score", "Rank").replace("(rate)", "Rank")
                == col
            ):
                rename_dict[col] = "DOMAIN_Rank_2019"

    imd19.rename(columns=rename_dict, inplace=True)

    # Merge all data
    all_data = census.merge(imd19, left_index=True, right_on="LSOA11CD")
    all_data = all_data.reset_index(drop=True)

    # Load 2025 targets
    imd25 = pd.read_csv(IMD_2025_FILE)
    if target_domain_col not in imd25.columns:
        print(f"Error: {target_domain_col} not in {IMD_2025_FILE}")
        return None, None, None

    imd25 = imd25[["LSOA code (2021)", target_domain_col]]
    imd25.columns = ["LSOA21CD", "TARGET_2025"]

    # Load lookup
    lookup = pd.read_csv(LOOKUP_FILE)
    train_lookup = lookup[lookup["overlap_pct"] == 100]
    print(f"Found {len(train_lookup)} LSOAs with 100% overlap for training")

    train_data = train_lookup.merge(imd25, on="LSOA21CD")
    train_data = train_data.merge(all_data, on="LSOA11CD")

    # Identify changed boundaries
    changed_lsoas = lookup[lookup["overlap_pct"] < 100]["LSOA11CD"].unique()
    print(f"Found {len(changed_lsoas)} LSOAs with boundary changes")
    print(f"Total LSOAs to predict: {len(all_data)}")

    return train_data.dropna(), all_data, changed_lsoas


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


def add_spatial_features_fixed(data, features):
    """Fixed spatial feature generation"""
    print("Adding spatial features...")
    start_time = time.time()

    try:
        # Load shapefile
        gdf = gpd.read_file(LSOA_2011_SHP)[["LSOA11CD", "geometry"]]

        # Only use key features for spatial lags
        key_features = []
        if "DOMAIN_2019" in features[:10]:
            key_features.append("DOMAIN_2019")
        if "IMD_2019" in features[:10]:
            key_features.append("IMD_2019")

        # Add top census features
        for feat in features[:10]:
            if feat not in key_features and feat in data.columns:
                key_features.append(feat)
            if len(key_features) >= 5:
                break

        # Prepare data for spatial operations
        spatial_data = data[["LSOA11CD"] + key_features].copy()
        spatial_geo = spatial_data.merge(gdf, on="LSOA11CD", how="inner")

        if len(spatial_geo) == 0:
            print("No matching geometries. Skipping spatial features.")
            return data

        spatial_geo = gpd.GeoDataFrame(spatial_geo)

        # Create spatial weights
        from libpysal.weights import Queen

        w = Queen.from_dataframe(spatial_geo, use_index=False)

        # Calculate spatial lags
        new_cols = []
        for feat in key_features:
            if feat in spatial_geo.columns:
                lag_vals = np.zeros(len(spatial_geo))
                for i in range(len(spatial_geo)):
                    neighbors = w[i].keys() if i in w.neighbors else []
                    if neighbors:
                        neighbor_vals = spatial_geo.iloc[list(neighbors)][feat].values
                        lag_vals[i] = np.nanmean(neighbor_vals)
                    else:
                        lag_vals[i] = spatial_geo.iloc[i][feat]

                lag_name = f"{feat[:15]}_lag"
                spatial_geo[lag_name] = lag_vals
                new_cols.append(lag_name)

        # Merge back
        if new_cols:
            merge_cols = ["LSOA11CD"] + new_cols
            data = data.merge(spatial_geo[merge_cols], on="LSOA11CD", how="left")
            data[new_cols] = data[new_cols].fillna(0)
            print(f"Added {len(new_cols)} spatial features")

    except Exception as e:
        print(f"Spatial features failed: {e}. Continuing without them.")

    print(f"Spatial features took {time.time() - start_time:.2f}s")
    return data


def train_optimized_ensemble(train_data, all_data, features, changed_lsoas):
    """Optimized ensemble with multiple models"""
    print("\nTraining optimized ensemble...")
    start_time = time.time()

    # Apply feature engineering
    train_data = engineer_features_organic(train_data)
    all_data = engineer_features_organic(all_data)

    # Add spatial features
    train_data = add_spatial_features_fixed(train_data, features)
    all_data = add_spatial_features_fixed(all_data, features)

    # Get all features including engineered ones
    spatial_cols = [col for col in train_data.columns if col.endswith("_lag")]
    engineered_cols = [
        "DOMAIN_2019_log",
        "DOMAIN_2019_pct",
        "DOMAIN_vs_IMD",
        "DOMAIN_IMD_diff",
        "Income_Employment_avg",
        "Census_Deprivation_Score",
    ]

    # Combine all features
    all_features = (
        features
        + spatial_cols
        + [col for col in engineered_cols if col in train_data.columns]
    )
    all_features = list(dict.fromkeys(all_features))  # Remove duplicates

    # Filter to existing columns
    features_final = [
        f for f in all_features if f in train_data.columns and f in all_data.columns
    ]

    # Ensure DOMAIN_2019 is first
    if "DOMAIN_2019" in features_final:
        features_final.remove("DOMAIN_2019")
        features_final = ["DOMAIN_2019"] + features_final

    print(
        f"Using {len(features_final)} total features (including {len(spatial_cols)} spatial)"
    )

    # Prepare training data
    X_train_full = train_data[features_final].fillna(0)
    y_train_full = train_data["TARGET_2025"]

    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    val_indices = X_val.index
    val_lsoas = train_data.iloc[val_indices]["LSOA11CD"].values

    # Prepare prediction data
    X_all = all_data[features_final].fillna(0)

    print(f"Training on {len(X_train):,} LSOAs, validating on {len(X_val):,} LSOAs")

    # MODEL 1: Tuned Random Forest (Less Overfitting)
    print("\n1. Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=500,  # FASTER: 800 is likely overkill
        max_depth=20,  # REDUCE OVERFITTING: 30 is too deep
        min_samples_split=5,  # REDUCE OVERFITTING: Force splits to be more significant
        min_samples_leaf=2,  # REDUCE OVERFITTING: Stop creating leaves for 1-2 LSOAs
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    # MODEL 2: XGBoost with better hyperparameters
    print("\n2. Training XGBoost...")
    xgb_params = {
        "n_estimators": 2000,  # MORE ROOM: Give more trees to find the best stop
        "max_depth": 6,  # REDUCE OVERFITTING: 8 is very deep for XGB
        "learning_rate": 0.01,  # SLOW DOWN: Learn slower for a more stable result
        "subsample": 0.8,  # (Slightly less than 0.85)
        "colsample_bytree": 0.8,  # (Slightly less than 0.85)
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1,
        "early_stopping_rounds": 100,  # MORE PATIENCE: Allow 100 rounds to find the true minimum
        "eval_metric": "rmse",
    }

    gb = xgb.XGBRegressor(**xgb_params)

    # MODEL 3: LightGBM (often better than XGBoost)
    print("\n3. Training LightGBM...")
    lgb_params = {
        "n_estimators": 2000,  # MORE ROOM: Give more trees
        "max_depth": -1,
        "num_leaves": 41,  # MORE COMPLEXITY: 31 is default, let's allow more
        "learning_rate": 0.01,  # SLOW DOWN: Learn slower
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        # Note: early_stopping_rounds is applied in the .fit() call, not here
        # You should also increase it to 100 in the .fit() call
    }

    lgbm = lgb.LGBMRegressor(**lgb_params)

    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\nCross-validation scores:")
    models = {"Random Forest": rf, "XGBoost": gb, "LightGBM": lgbm}

    cv_scores = {}
    for name, model in models.items():
        if name == "XGBoost":
            # XGBoost with eval set
            scores = []
            for train_idx, val_idx in cv.split(X_train):
                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model_copy = xgb.XGBRegressor(**xgb_params)
                model_copy.fit(
                    X_cv_train,
                    y_cv_train,
                    eval_set=[(X_cv_val, y_cv_val)],
                    verbose=False,
                )
                pred = model_copy.predict(X_cv_val)
                scores.append(r2_score(y_cv_val, pred))
            cv_scores[name] = np.array(scores)
        else:
            cv_scores[name] = cross_val_score(
                model, X_train, y_train, cv=cv, scoring="r2"
            )

        print(
            f"{name}: R²={cv_scores[name].mean():.4f} (+/- {cv_scores[name].std():.4f})"
        )

    # Train final models
    print("\nTraining final models...")

    # Random Forest
    rf.fit(X_train, y_train)
    if hasattr(rf, "oob_score_"):
        print(f"RF OOB Score: {rf.oob_score_:.4f}")

    # XGBoost with early stopping
    gb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    # LightGBM
    lgbm.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )

    # Validation performance
    print("\nValidation set performance:")
    val_preds = {}
    for name, model in models.items():
        val_preds[name] = model.predict(X_val)
        r2 = r2_score(y_val, val_preds[name])
        mae = mean_absolute_error(y_val, val_preds[name])
        print(f"{name}: R²={r2:.4f}, MAE={mae:.3f}")

    # Feature importance from best model
    best_model_name = max(cv_scores, key=lambda x: cv_scores[x].mean())
    print(f"\nBest model: {best_model_name}")

    if best_model_name == "Random Forest":
        importances = rf.feature_importances_
    elif best_model_name == "XGBoost":
        importances = gb.feature_importances_
    else:
        importances = lgbm.feature_importances_

    top_features = pd.Series(importances, index=features_final).sort_values(
        ascending=False
    )
    print("\nTop 10 most important features:")
    for i, (feat, imp) in enumerate(top_features.head(10).items()):
        print(f"  {i + 1:2}. {feat[:30]:30} {imp:.4f}")

    # Retrain on full data
    print(f"\nRetraining on full training set ({len(X_train_full):,} LSOAs)...")
    rf.fit(X_train_full, y_train_full)
    gb.fit(
        X_train_full,
        y_train_full,
        eval_set=[(X_train_full, y_train_full)],
        verbose=False,
    )
    lgbm.fit(
        X_train_full,
        y_train_full,
        eval_set=[(X_train_full, y_train_full)],
        callbacks=[lgb.log_evaluation(0)],
    )

    # Generate predictions
    print("\nGenerating predictions...")
    pred_rf = rf.predict(X_all)
    pred_gb = gb.predict(X_all)
    pred_lgb = lgbm.predict(X_all)

    # Optimized ensemble weights based on validation performance
    total_score = sum([r2_score(y_val, pred) for pred in val_preds.values()])
    weights = {
        name: r2_score(y_val, val_preds[name]) / total_score for name in val_preds
    }

    print(
        f"\nEnsemble weights: RF={weights['Random Forest']:.3f}, "
        f"XGB={weights['XGBoost']:.3f}, LGBM={weights['LightGBM']:.3f}"
    )

    # Weighted ensemble
    predictions = (
        weights["Random Forest"] * pred_rf
        + weights["XGBoost"] * pred_gb
        + weights["LightGBM"] * pred_lgb
    )

    # Create output
    predictions_df = pd.DataFrame(
        {"LSOA11CD": all_data["LSOA11CD"].values, "prediction": predictions}
    )

    # Confidence scores
    confidence = np.ones(len(all_data))
    confidence[all_data["LSOA11CD"].isin(changed_lsoas)] = 0.75

    # Validation ensemble
    val_ensemble = (
        weights["Random Forest"] * val_preds["Random Forest"]
        + weights["XGBoost"] * val_preds["XGBoost"]
        + weights["LightGBM"] * val_preds["LightGBM"]
    )

    val_data = pd.DataFrame(
        {"LSOA11CD": val_lsoas, "actual": y_val.values, "predicted": val_ensemble}
    )

    print(f"\nTotal training time: {time.time() - start_time:.2f}s")

    return predictions_df, confidence, val_data, train_data


def validate_and_save(
    val_data, all_data, predictions_df, confidence, changed_lsoas, short_name
):
    """Validate and save results"""

    # Calculate metrics
    r2 = r2_score(val_data["actual"], val_data["predicted"])
    mae = mean_absolute_error(val_data["actual"], val_data["predicted"])
    rmse = np.sqrt(mean_squared_error(val_data["actual"], val_data["predicted"]))

    print(f"\n{'=' * 60}")
    print(f"FINAL VALIDATION RESULTS ({short_name})")
    print(f"{'=' * 60}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

    # Create validation plot
    plt.figure(figsize=(10, 8))
    plt.scatter(val_data["actual"], val_data["predicted"], alpha=0.6, s=20)

    # Add perfect fit line
    min_val = min(val_data["actual"].min(), val_data["predicted"].min())
    max_val = max(val_data["actual"].max(), val_data["predicted"].max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, alpha=0.7)

    # Add metrics to plot
    plt.text(
        0.05,
        0.95,
        f"R² = {r2:.4f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.title(f"{short_name} - Actual vs Predicted (2025)")
    plt.xlabel("Actual 2025 Score")
    plt.ylabel("Predicted 2025 Score")
    plt.grid(True, alpha=0.3)

    plot_filename = f"validation_{short_name}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Validation plot saved to {plot_filename}")

    # Save results
    if r2 > 0.7:
        print("\n✓ Model performance acceptable. Saving results...")

        # Prepare results dataframe
        results = all_data[["LSOA11CD", "DOMAIN_2019"]].copy()
        results = results.merge(predictions_df, on="LSOA11CD", how="left")
        results.rename(
            columns={
                "DOMAIN_2019": f"{short_name}_2019",
                "prediction": f"{short_name}_2025_pred",
            },
            inplace=True,
        )

        results["confidence"] = confidence
        results["boundary_changed"] = (
            results["LSOA11CD"].isin(changed_lsoas).astype(int)
        )
        results["change"] = (
            results[f"{short_name}_2025_pred"] - results[f"{short_name}_2019"]
        )

        # Calculate percentage change
        with np.errstate(divide="ignore", invalid="ignore"):
            results["pct_change"] = (
                100 * results["change"] / results[f"{short_name}_2019"]
            )
            results["pct_change"] = results["pct_change"].replace(
                [np.inf, -np.inf], np.nan
            )

        # Save CSV
        csv_filename = f"IMD_2025_{short_name}_2011_boundaries.csv"
        results.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")

        # Save shapefile
        try:
            gdf = gpd.read_file(LSOA_2011_SHP)[["LSOA11CD", "geometry"]]
            gdf = gdf.merge(results, on="LSOA11CD", how="left")

            # Shorten column names for shapefile
            gdf.rename(
                columns={
                    f"{short_name}_2019": f"{short_name[:7]}_19"
                    if len(short_name) > 7
                    else f"{short_name}_19",
                    f"{short_name}_2025_pred": f"{short_name[:7]}_25"
                    if len(short_name) > 7
                    else f"{short_name}_25",
                    "boundary_changed": "bound_chg",
                    "confidence": "conf",
                    "change": "chg",
                    "pct_change": "pct_chg",
                },
                inplace=True,
            )

            shp_filename = f"IMD_2025_{short_name}_2011_boundaries.shp"
            gdf.to_file(shp_filename)
            print(f"Shapefile saved to {shp_filename}")
        except Exception as e:
            print(f"Could not save shapefile: {e}")

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"SUMMARY STATISTICS ({short_name})")
        print(f"{'=' * 60}")
        print(f"Total LSOAs: {len(results):,}")
        print(f"Unchanged boundaries: {(results['boundary_changed'] == 0).sum():,}")
        print(f"Changed boundaries: {results['boundary_changed'].sum():,}")

        if f"{short_name}_2019" in results.columns:
            print(
                f"\n2019: Mean={results[f'{short_name}_2019'].mean():.3f}, "
                f"Std={results[f'{short_name}_2019'].std():.3f}"
            )
            print(
                f"2025: Mean={results[f'{short_name}_2025_pred'].mean():.3f}, "
                f"Std={results[f'{short_name}_2025_pred'].std():.3f}"
            )
            print(f"Mean change: {results['change'].mean():.3f}")
            print(f"Median % change: {results['pct_change'].median():.1f}%")

    else:
        print(f"\n⚠ Warning: Low R² ({r2:.3f}). Results may be unreliable.")
        csv_filename = f"IMD_2025_{short_name}_LOW_R2.csv"
        predictions_df.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")

    return r2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimized transfer model for IMD domain"
    )
    parser.add_argument(
        "--domain_col",
        type=str,
        required=True,
        help="Column name (e.g., 'Income Score (rate)')",
    )
    parser.add_argument(
        "--short_name", type=str, required=True, help="Short name (e.g., 'Income')"
    )
    args = parser.parse_args()

    total_start = time.time()

    print(f"\n{'=' * 60}")
    print(f"PROCESSING: {args.short_name}")
    print(f"{'=' * 60}")

    # Load features
    features = load_features(args.short_name)
    if not features:
        print("No features loaded. Exiting.")
        exit(1)
    print(f"Loaded {len(features)} features")

    # Prepare data
    data_prep_start = time.time()
    train_data, all_data, changed_lsoas = prepare_data_optimized(
        features, args.domain_col
    )
    if train_data is None or all_data is None:
        print("Failed to prepare data. Exiting.")
        exit(1)
    print(f"Data prepared in {time.time() - data_prep_start:.2f}s")

    # Train models
    predictions_df, confidence, val_data, train_data_final = train_optimized_ensemble(
        train_data, all_data, features, changed_lsoas
    )

    if predictions_df is None:
        print("Model training failed. Exiting.")
        exit(1)

    # Validate and save
    print("\nValidating and saving results...")
    val_save_start = time.time()
    r2 = validate_and_save(
        val_data, all_data, predictions_df, confidence, changed_lsoas, args.short_name
    )
    print(f"Validation and saving took {time.time() - val_save_start:.2f}s")

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"COMPLETED: {args.short_name}")
    print(f"Final R² Score: {r2:.4f}")
    print(f"Total time: {time.time() - total_start:.2f}s")
    print(f"{'=' * 60}\n")
