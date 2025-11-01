#!/bin/bash
#
# run_all_domains.sh
#
# Runs the feature selection and transfer modelling for all IMD domains.
# Assumes 'feature_selection_domains.py' and 'transfer_domains.py' are
# in the same directory.
#
# Make this script executable:
# chmod +x run_all_domains.sh
#
# Then run it:
# ./run_all_domains.sh

echo "Starting IMD Domain Analysis Pipeline..."

# Define all domains as a list of "Column Name|ShortName"
# NOTE: Column names must be *exactly* as they appear in the CSV headers.
DOMAINS=(
    "Index of Multiple Deprivation (IMD) Score|IMD"
    "Income Score (rate)|Income"
    "Employment Score (rate)|Employment"
    "Education, Skills and Training Score|Education"
    "Health Deprivation and Disability Score|Health"
    "Crime Score|Crime"
    "Barriers to Housing and Services Score|Barriers"
    "Living Environment Score|LivingEnv"
    "Income Deprivation Affecting Children Index (IDACI) Score (rate)|IDACI"
    "Income Deprivation Affecting Older People (IDAOPI) Score (rate)|IDAOPI"
    "Children and Young People Sub-domain Score|CYP"
    "Adult Skills Sub-domain Score|AdultSkills"
    "Geographical Barriers Sub-domain Score|GeoBarriers"
    "Wider Barriers Sub-domain Score|WiderBarriers"
    "Indoors Sub-domain Score|Indoors"
    "Outdoors Sub-domain Score|Outdoors"
)

# Create a log file
LOGFILE="domain_pipeline_run.log"
echo "Logging output to $LOGFILE"
> $LOGFILE # Clear logfile

# Loop through each domain
for item in "${DOMAINS[@]}"; do
    # Split the string by the pipe |
    IFS='|' read -r DOMAIN_COL SHORT_NAME <<< "$item"
    
    echo "" | tee -a $LOGFILE
    echo "============================================================" | tee -a $LOGFILE
    echo "Processing Domain: $SHORT_NAME ($DOMAIN_COL)" | tee -a $LOGFILE
    echo "============================================================" | tee -a $LOGFILE

    # --- Step 1: Feature Selection ---
    echo "\n[Step 1/2] Running Feature Selection for $SHORT_NAME..." | tee -a $LOGFILE
    
    # --- NEW FIX: Remove old feature file first ---
    rm -f "selected_features_${SHORT_NAME}.txt"
    # --- END NEW FIX ---
    
    python3 feature_selection_domains.py \
        --domain_col "$DOMAIN_COL" \
        --short_name "$SHORT_NAME" \
        | tee -a $LOGFILE

    # Check if feature selection was successful
    if [ ! -f "selected_features_${SHORT_NAME}.txt" ]; then
        echo "Error: Feature selection for $SHORT_NAME failed. Skipping transfer." | tee -a $LOGFILE
        continue
    fi

    # --- Step 2: Transfer Modelling ---
    echo "\n[Step 2/2] Running Transfer Modelling for $SHORT_NAME..." | tee -a $LOGFILE
    
    python3 transfer_domains.py \
        --domain_col "$DOMAIN_COL" \
        --short_name "$SHORT_NAME" \
        | tee -a $LOGFILE

    echo "\nFinished processing $SHORT_NAME." | tee -a $LOGFILE

done

echo ""
echo "============================================================"
echo "All domains processed. Check $LOGFILE for details."
echo "============================================================"