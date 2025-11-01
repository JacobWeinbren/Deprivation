# prepare_lookup.py
import pandas as pd
import geopandas as gpd


def create_lookup():
    """Create lookup table with overlap percentages"""

    # Option 2: Calculate from shapefiles
    print("Calculating overlaps from shapefiles...")
    lsoa_2011 = gpd.read_file("LSOA/2011/LSOA_2011_EW_BFC_V3.shp")
    lsoa_2021 = gpd.read_file("LSOA/2021/LSOA_2021_EW_BFC_V10.shp")

    # For most LSOAs, codes are identical = 100% overlap
    all_2011 = set(lsoa_2011["LSOA11CD"])
    all_2021 = set(lsoa_2021["LSOA21CD"])
    unchanged = all_2011.intersection(all_2021)

    # Create lookup for unchanged
    lookup = pd.DataFrame(
        {
            "LSOA11CD": list(unchanged),
            "LSOA21CD": list(unchanged),
            "overlap_pct": 100,
        }
    )

    # Add changed LSOAs with spatial overlay
    changed_2011 = all_2011 - unchanged
    if changed_2011:
        overlay = gpd.overlay(
            lsoa_2011[lsoa_2011["LSOA11CD"].isin(changed_2011)],
            lsoa_2021,
            how="intersection",
        )
        overlay["area"] = overlay.geometry.area
        overlay = overlay.groupby(["LSOA11CD", "LSOA21CD"])["area"].sum().reset_index()

        # Get original areas
        lsoa_2011["area_2011"] = lsoa_2011.geometry.area
        overlay = overlay.merge(lsoa_2011[["LSOA11CD", "area_2011"]], on="LSOA11CD")
        overlay["overlap_pct"] = (overlay["area"] / overlay["area_2011"]) * 100

        # Add to lookup
        lookup = pd.concat([lookup, overlay[["LSOA11CD", "LSOA21CD", "overlap_pct"]]])

    lookup.to_csv("LSOA_2011_2021_lookup.csv", index=False)

    print(f"\nLookup Summary:")
    print(f"Total 2011 LSOAs: {lookup['LSOA11CD'].nunique()}")
    print(f"100% overlap (unchanged): {(lookup['overlap_pct'] == 100).sum()}")
    print(f"<100% overlap (changed): {(lookup['overlap_pct'] < 100).sum()}")
    print("\nSaved to LSOA_2011_2021_lookup.csv")


if __name__ == "__main__":
    create_lookup()
