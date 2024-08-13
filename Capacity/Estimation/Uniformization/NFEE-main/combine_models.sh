#!/bin/bash

# Step 1: Stash any changes to files in the folder temp_data/saved_models/4N
echo "Stashing changes in temp_data/saved_models/4N..."
git stash push -m "Stash before pulling" temp_data/saved_models/4N/*

# Step 2: Add, commit, and push any changes to files in temp_data/CPDSSS_data/*
echo "Committing changes in temp_data/CPDSSS_data/*..."
git add temp_data/CPDSSS_data/*
git commit -m "Update CPDSSS data"
git push

# Step 3: Pull changes (this will include changes to temp_data/saved_models/4N)
echo "Pulling latest changes..."
git pull

# Step 4: Move files from temp_data/saved_models/4N to temp_data/saved_models/4N_old
echo "Moving files from 4N to 4N_old..."
mkdir -p temp_data/saved_models/4N_old
mv temp_data/saved_models/4N/* temp_data/saved_models/4N_old/

# Step 5: Restore the git stash
echo "Restoring stashed changes..."
git stash pop

# Step 6: Activate conda environment and run the Python script
echo "Activating conda environment and running combine_model_files.py..."
conda activate gpuknn_new
python combine_model_files.py

# Step 7: Delete temp_data/saved_models/4N_old
echo "Deleting temp_data/saved_models/4N_old..."
rm -rf temp_data/saved_models/4N_old

# Step 8: Commit and push changes to temp_data/saved_models/4N
echo "Committing and pushing changes in temp_data/saved_models/4N..."
git add temp_data/saved_models/4N/*
git commit -m "Update 4N models after combining"
#git push

echo "Script completed."
