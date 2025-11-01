#!/usr/bin/env python3
"""
Download ALL 2011 Census data for England at LSOA level.
Downloads complete bulk data from Nomis - all Key Statistics and Quick Statistics tables.
"""

import requests
import pandas as pd
from pathlib import Path
import zipfile
import io
import time
from typing import List, Tuple


class CompleteCensusDownloader:
    """Download ALL 2011 Census bulk data tables."""

    def __init__(self, output_dir="census_2011_complete"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.base_url = "https://www.nomisweb.co.uk/output/census/2011"

    def get_all_key_statistics_tables(self) -> List[Tuple[str, str]]:
        """Get all Key Statistics table codes."""
        tables = [
            ("KS101EW", "Usual Resident Population"),
            ("KS102EW", "Age Structure"),
            ("KS103EW", "Marital and Civil Partnership Status"),
            ("KS104EW", "Living Arrangements"),
            ("KS105EW", "Household Composition"),
            ("KS106EW", "Adults Not in Employment"),
            ("KS107EW", "Lone Parent Households With Dependent Children"),
            ("KS201EW", "Ethnic Group"),
            ("KS202EW", "National Identity"),
            ("KS204EW", "Country of Birth"),
            ("KS205EW", "Passport Held"),
            ("KS206EW", "Household Language"),
            ("KS207WA", "Welsh Language Skills"),
            ("KS208WA", "Welsh Language Profile"),
            ("KS209EW", "Religion"),
            ("KS301EW", "Health and Provision of Unpaid Care"),
            ("KS401EW", "Dwellings, Household Spaces and Accommodation Type"),
            ("KS402EW", "Tenure"),
            ("KS403EW", "Rooms, Bedrooms and Central Heating"),
            ("KS404EW", "Car or Van Availability"),
            ("KS405EW", "Communal Establishments and Residents"),
            ("KS501EW", "Qualifications and Students"),
            ("KS601EW", "Economic Activity"),
            ("KS602EW", "Economic Activity - Males"),
            ("KS603EW", "Economic Activity - Females"),
            ("KS604EW", "Hours Worked"),
            ("KS605EW", "Industry"),
            ("KS606EW", "Industry - Males"),
            ("KS607EW", "Industry - Females"),
            ("KS608EW", "Occupation"),
            ("KS609EW", "Occupation - Males"),
            ("KS610EW", "Occupation - Females"),
            ("KS611EW", "NS-SeC"),
            ("KS612EW", "NS-SeC - Males"),
            ("KS613EW", "NS-SeC - Females"),
        ]
        return tables

    def get_all_quick_statistics_tables(self) -> List[Tuple[str, str]]:
        """Get all Quick Statistics table codes."""
        tables = [
            ("QS101EW", "Residence Type"),
            ("QS102EW", "Population Density"),
            ("QS103EW", "Age by Single Year"),
            ("QS104EW", "Sex"),
            ("QS105EW", "Schoolchildren and full-time students at their non term-time"),
            ("QS106EW", "Second Address"),
            ("QS108EW", "Living Arrangements"),
            ("QS110EW", "Adult Lifestage (alternative adult definition)"),
            ("QS111EW", "Household Lifestage"),
            ("QS112EW", "Household Composition - People"),
            ("QS113EW", "Household Composition - Households"),
            (
                "QS114EW",
                "Household Comp. (alternative child/adult definition) - People",
            ),
            (
                "QS115EW",
                "Household Comp. (alternative child/adult definition) - Households",
            ),
            ("QS116EW", "Household Type"),
            ("QS117EW", "People Aged 18 to 64 Living In a One Adult Household"),
            ("QS118EW", "Families with Dependent Children"),
            ("QS119EW", "Households by Deprivation Dimensions"),
            ("QS121EW", "Armed Forces"),
            ("QS201EW", "Ethnic Group"),
            ("QS202EW", "Multiple Ethnic Groups"),
            ("QS203EW", "Country of Birth (detailed)"),
            ("QS204EW", "Main Language"),
            ("QS205EW", "Proficiency in English"),
            ("QS206WA", "Welsh Language Skills"),
            ("QS207WA", "Welsh Language Skills (Detailed)"),
            ("QS208EW", "Religion"),
            ("QS210EW", "Religion (detailed)"),
            ("QS211EW", "Ethnic Group (detailed)"),
            ("QS301EW", "Provision of Unpaid Care"),
            ("QS302EW", "General Health"),
            ("QS303EW", "Long-term Health Problem or Disability"),
            ("QS401EW", "Accommodation Type - People"),
            ("QS402EW", "Accommodation Type - Households"),
            ("QS403EW", "Tenure - People"),
            ("QS404EW", "Tenure - Household Reference Person aged 65 and Over"),
            ("QS405EW", "Tenure - Households"),
            ("QS406EW", "Household Size"),
            ("QS407EW", "Number of Rooms"),
            ("QS408EW", "Occupancy Rating (rooms)"),
            ("QS409EW", "Persons per Room - Households"),
            ("QS410EW", "Persons per Room - people"),
            ("QS411EW", "Number of Bedrooms"),
            ("QS412EW", "Occupancy Rating (Bedrooms)"),
            ("QS413EW", "Persons per Bedroom - Households"),
            ("QS414EW", "Persons per Bedroom - People"),
            ("QS415EW", "Central Heating"),
            ("QS416EW", "Car or Van Availability"),
            ("QS417EW", "Household Spaces"),
            ("QS418EW", "Dwellings"),
            ("QS419EW", "Position in Communal Establishment"),
            ("QS420EW", "Communal establishment management and type"),
            ("QS421EW", "Communal establishment management and type - People"),
            ("QS501EW", "Highest Level of Qualification"),
            ("QS502EW", "Qualifications Gained"),
            ("QS601EW", "Economic Activity"),
            ("QS602EW", "Economic Activity of Household Reference Person"),
            ("QS603EW", "Economic Activity - Full-time Students"),
            ("QS604EW", "Hour Worked"),
            ("QS605EW", "Industry"),
            ("QS606EW", "Occupation (Minor Groups)"),
            ("QS607EW", "NS-Sec"),
            ("QS608EW", "NS-SeC of Household Reference Person - People Aged under 65"),
            ("QS609EW", "NS-Sec of Household Reference Person - People Aged under 65"),
            ("QS610EW", "NS-Sec of Household Reference Person - HRP Aged under 65"),
            ("QS611EW", "Approximated Social Grade"),
            ("QS612EW", "Year Last Worked"),
            ("QS701EW", "Method of Travel to Work"),
            ("QS801EW", "Year of Arrival in UK"),
            ("QS802EW", "Age of Arrival in UK"),
            ("QS803EW", "Length of Residence in UK"),
            ("CT0010", "Ethnic group"),
        ]
        return tables

    def download_bulk_table(
        self, table_code: str, table_name: str, file_type: str = "oa"
    ) -> bool:
        """
        Download a bulk census table.

        Parameters:
        -----------
        table_code : str
            The census table code (e.g., 'KS101EW')
        table_name : str
            Description of the table
        file_type : str
            'oa' for Output Areas (includes LSOAs) or 'wards' for Ward level
        """
        print(f"Downloading {table_code}: {table_name} ({file_type} file)...")

        # Construct the download URL for bulk data
        # Format: https://www.nomisweb.co.uk/output/census/2011/ks101ew_2011_oa.zip
        table_lower = table_code.lower()
        url = f"{self.base_url}/{table_lower}_2011_{file_type}.zip"

        try:
            response = requests.get(url, timeout=300, stream=True)

            if response.status_code == 200:
                # Save zip file
                zip_path = self.output_dir / f"{table_code}_{file_type}.zip"

                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Extract zip file
                extract_dir = self.output_dir / f"{table_code}_{file_type}"
                extract_dir.mkdir(exist_ok=True)

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)

                # Find and filter CSV files for England LSOAs
                csv_files = list(extract_dir.glob("*.csv"))

                if csv_files:
                    for csv_file in csv_files:
                        try:
                            # Read the CSV
                            df = pd.read_csv(csv_file, low_memory=False)

                            # Filter for England (codes starting with 'E')
                            if "GeographyCode" in df.columns:
                                original_len = len(df)
                                df = df[
                                    df["GeographyCode"].str.startswith("E", na=False)
                                ]

                                # Save filtered England data
                                if len(df) > 0:
                                    output_file = (
                                        self.output_dir
                                        / f"{table_code}_england_{csv_file.name}"
                                    )
                                    df.to_csv(output_file, index=False)
                                    print(
                                        f"  ✓ Saved {len(df)} England records (from {original_len} total)"
                                    )
                        except Exception as e:
                            print(f"  ⚠ Error processing {csv_file.name}: {e}")

                # Remove zip file to save space
                zip_path.unlink()
                print(f"  ✓ Extracted and cleaned {table_code}")
                return True

            else:
                print(f"  ✗ Failed: HTTP {response.status_code}")
                return False

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

    def download_all_tables(self):
        """Download all census tables."""
        print("=" * 80)
        print("COMPLETE 2011 Census Bulk Data Download for England (LSOA level)")
        print("=" * 80)
        print()

        # Get all tables
        ks_tables = self.get_all_key_statistics_tables()
        qs_tables = self.get_all_quick_statistics_tables()

        all_tables = ks_tables + qs_tables

        print(f"Total tables to download: {len(all_tables)}")
        print(f"  - Key Statistics: {len(ks_tables)}")
        print(f"  - Quick Statistics: {len(qs_tables)}")
        print()
        print("This will take a while (30-60 minutes)...")
        print()

        successful = 0
        failed = 0
        failed_tables = []

        # Download Output Area files (includes LSOAs)
        for i, (table_code, table_name) in enumerate(all_tables, 1):
            print(f"\n[{i}/{len(all_tables)}] ", end="")

            if self.download_bulk_table(table_code, table_name, file_type="oa"):
                successful += 1
            else:
                failed += 1
                failed_tables.append(table_code)

            # Be polite to the server
            time.sleep(2)

        # Summary
        print("\n" + "=" * 80)
        print("DOWNLOAD COMPLETE!")
        print("=" * 80)
        print(f"Successful: {successful}/{len(all_tables)}")
        print(f"Failed: {failed}/{len(all_tables)}")

        if failed_tables:
            print(f"\nFailed tables: {', '.join(failed_tables)}")

        print(f"\nAll data saved to: {self.output_dir.absolute()}")

        # Show what was downloaded
        csv_files = list(self.output_dir.glob("*_england_*.csv"))
        print(f"\nTotal England CSV files: {len(csv_files)}")

        total_size = sum(f.stat().st_size for f in csv_files)
        print(f"Total data size: {total_size / (1024**3):.2f} GB")

        print("\n" + "=" * 80)


def main():
    downloader = CompleteCensusDownloader()
    downloader.download_all_tables()


if __name__ == "__main__":
    main()
