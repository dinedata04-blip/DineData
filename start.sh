#!/bin/bash
# start.sh — on the VERY FIRST boot only, sets up EMPTY CSVs (headers only,
# zero data rows) and builds an EMPTY database (correct schema, zero
# records) so the deployed app starts with a clean slate. Every boot
# after the first just skips straight to launching Streamlit, since the
# files already exist on the persistent volume by then.

mkdir -p Database Data_Cleaning

if [ ! -f Data_Cleaning/kofe_tala_sales_data.csv ]; then
    echo "First boot detected — seeding EMPTY (headers-only) CSVs into Data_Cleaning/"
    cp Data_Cleaning_empty/*.csv Data_Cleaning/
fi

if [ ! -f Database/dinedata.db ]; then
    echo "First boot detected — building an EMPTY database schema"
    python build_database.py
fi

streamlit run dashboard_kofetala.py --server.port "$PORT" --server.address 0.0.0.0
