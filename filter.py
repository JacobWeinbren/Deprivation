#
# master_file.py (v10 - Handles both flat and nested folder structures)
#
import pandas as pd
import glob
import os

# --- Configuration ---
CENSUS_ROOT_FOLDER = "census_2011_complete"
LSOA_CODE_COL = "LSOA11CD"
OUTPUT_FILE = "2011_census_master_lsoa.csv"


def create_lsoa_master_file(root_folder):
    """
    Loops through all census subfolders, finds the *DATA.CSV file in each,
    (handling both flat and nested structures), extracts 'E01' LSOA rows,
    and joins them into one file.
    """
    try:
        # Get the first level of subfolders (e.g., "KS101EW_oa")
        topic_folders = [
            entry.path for entry in os.scandir(root_folder) if entry.is_dir()
        ]
    except FileNotFoundError:
        print(f"Error: Root folder not found at '{root_folder}'.")
        print("Please check the CENSUS_ROOT_FOLDER variable.")
        return

    if not topic_folders:
        print(f"Error: No data subfolders found in '{root_folder}'.")
        return

    print(f"Found {len(topic_folders)} census topic folders. Starting processing...")
    master_df = pd.DataFrame()

    for folder_path in topic_folders:
        basename = os.path.basename(folder_path)  # e.g., "KS101EW_oa"
        csv_full_path = None

        try:
            # --- NEW ROBUST LOGIC ---
            # We need to find the main data file, e.g., "KS101EWDATA.CSV"
            base_code_upper = basename.split("_")[0].upper()
            expected_csv_name = f"{base_code_upper}DATA.CSV"

            # Check 1: Is the file in the "flat" structure? (e.g., .../KS205EW_oa/KS205EWDATA.CSV)
            path_check_1 = os.path.join(folder_path, expected_csv_name)

            if os.path.exists(path_check_1):
                csv_full_path = path_check_1
            else:
                # Check 2: Is the file in the "nested" structure? (e.g., .../QS114EW_oa/qs114ew_2011_oa/...)
                base_code_lower = base_code_upper.lower()
                found_data_folder = None

                # Find the nested data subfolder
                for entry in os.scandir(folder_path):
                    if (
                        entry.is_dir()
                        and entry.name.lower().startswith(base_code_lower)
                        and "STATH" not in entry.name.upper()
                    ):
                        found_data_folder = entry.path
                        break  # Found it

                if found_data_folder:
                    # Look for the CSV file inside the nested folder
                    path_check_2 = os.path.join(found_data_folder, expected_csv_name)
                    if os.path.exists(path_check_2):
                        csv_full_path = path_check_2

            # --- END NEW LOGIC ---

            # Now, process the file if we found it
            if csv_full_path:
                print(f"  Processing {basename} (reading {csv_full_path})...")
                csv_data = pd.read_csv(csv_full_path, encoding="utf-8-sig")
            else:
                # If we couldn't find the file in either location
                print(
                    f"    Error: Could not find {expected_csv_name} in {basename} or its subfolders."
                )
                continue

            if "GeographyCode" not in csv_data.columns:
                print("    Error: 'GeographyCode' column not found.")
                continue

            # --- Filter logic (unchanged) ---
            lsoa_data = csv_data[
                csv_data["GeographyCode"].str.startswith("E01", na=False)
            ].copy()

            if lsoa_data.empty:
                # This is expected for 'WA' (Wales) files, not an error
                if "WA" in basename.upper():
                    print(
                        f"    No 'E01' (England LSOA) data found in this file (as expected for {basename})."
                    )
                else:
                    print(f"    No 'E01' (England LSOA) data found in this file.")
                continue

            print(f"    Found {len(lsoa_data)} LSOA rows.")

            # Rename and set index
            lsoa_data = lsoa_data.rename(columns={"GeographyCode": LSOA_CODE_COL})
            lsoa_data = lsoa_data.set_index(LSOA_CODE_COL)

            # Join to the master dataframe
            if master_df.empty:
                master_df = lsoa_data
            else:
                cols_to_drop = [
                    col for col in lsoa_data.columns if col in master_df.columns
                ]
                if cols_to_drop:
                    lsoa_data = lsoa_data.drop(columns=cols_to_drop)

                master_df = master_df.join(lsoa_data, how="outer")

        except Exception as e:
            print(f"    Error processing {basename}. Details: {e}")

    # --- Final Save ---
    if master_df.empty:
        print("No LSOA data was found in any of the folders. No output file created.")
        return

    print(f"All files processed. Saving master LSOA file to {OUTPUT_FILE}...")
    master_df.to_csv(OUTPUT_FILE)
    print("---")
    print(f"Success! Master LSOA data saved to {OUTPUT_FILE}.")


# --- Run the script ---
if __name__ == "__main__":
    create_lsoa_master_file(CENSUS_ROOT_FOLDER)
