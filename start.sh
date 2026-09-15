#!/bin/bash
# start.sh — Railway Free plan only allows ONE volume per project, but the
# app needs TWO persistent folders (Database/ and Data_Cleaning/). Fix:
# mount the single volume at /app/storage, then symlink Database and
# Data_Cleaning to live inside it. dashboard_kofetala.py and
# build_database.py both use the plain relative names "Database" and
# "Data_Cleaning", so nothing in the Python code needs to change.
#
# On the VERY FIRST boot this also seeds EMPTY CSVs (headers only, zero
# data rows) and builds an EMPTY database (correct schema, zero records)
# so the deployed app starts with a clean slate. Every boot after the
# first just skips straight to launching Streamlit, since the files
# already exist in the volume by then.
#
# NOTE: the DB build is only considered "done" once it finishes
# successfully — tracked via Database/.build_ok — instead of just
# checking whether dinedata.db exists. sqlite3.connect() creates the
# .db file immediately, before any rows are inserted, so a crash
# mid-build used to leave a permanently stuck, half-built database
# that no future code fix or redeploy could repair.

mkdir -p storage/Database storage/Data_Cleaning
ln -sfn storage/Database Database
ln -sfn storage/Data_Cleaning Data_Cleaning

if [ ! -f Data_Cleaning/kofe_tala_sales_data.csv ]; then
    echo "First boot detected — seeding EMPTY (headers-only) CSVs into Data_Cleaning/"
    cp Data_Cleaning_empty/*.csv Data_Cleaning/
fi

if [ ! -f Database/.build_ok ]; then
    echo "No successful build recorded — (re)building database schema"
    rm -f Database/dinedata.db
    python build_database.py && touch Database/.build_ok
fi

# Train the 4 Random Forest models EVERY boot (not just first boot).
# ML_Models/ and Feature_Engineering/ hold source code (not just data), so
# unlike Database/ and Data_Cleaning/ they can't be symlinked into the one
# Railway volume without hiding the .py files that live there too. Instead
# we just retrain on every start — the dataset is small (a few thousand
# rows) so this only takes a few seconds, and it has the nice side effect
# of always reflecting whatever is currently in Data_Cleaning/ (which DOES
# persist, and grows whenever someone uploads new data via the Database
# page), instead of shipping stale/empty models forever.
echo "Training ML models (feature engineering + 4 Random Forest models)..."
(cd Feature_Engineering && python feature_engineering.py) \
  && (cd ML_Models && python ml_models.py) \
  || echo "⚠️  Model training failed — dashboard will show 'Missing' for models and keep running without predictions."

streamlit run dashboard_kofetala.py --server.port "$PORT" --server.address 0.0.0.0
