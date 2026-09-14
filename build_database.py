"""
build_database.py
Builds dinedata.db — a SQLite database implementing DineData's Galaxy Schema
(3 fact tables + 5 dimension tables) from the existing cleaned CSV files.

Run this after your data pipeline (data_preprocessing -> feature_engineering)
produces the CSVs in Data_Cleaning/. Safe to re-run any time — it drops and
rebuilds all tables from the current CSVs.

Usage:
    python Database/build_database.py
"""
import sqlite3
import pandas as pd
import numpy as np
import os

# ── Paths (relative to project root, same convention as dashboard_kofetala.py) ──
DATA_CLEANING_DIR = 'Data_Cleaning'
DATABASE_DIR      = 'Database'
DB_PATH           = os.path.join(DATABASE_DIR, 'dinedata.db')

SALES_CSV     = os.path.join(DATA_CLEANING_DIR, 'kofe_tala_sales_data.csv')
MENU_CSV      = os.path.join(DATA_CLEANING_DIR, 'kofe_tala_menu_data.csv')
INVENTORY_CSV = os.path.join(DATA_CLEANING_DIR, 'kofe_tala_inventory_data.csv')
WASTE_CSV     = os.path.join(DATA_CLEANING_DIR, 'kofe_tala_waste_data.csv')

os.makedirs(DATABASE_DIR, exist_ok=True)


def build_schema(conn):
    """Create the Galaxy Schema tables (drops existing tables first)."""
    cur = conn.cursor()

    cur.executescript("""
    DROP TABLE IF EXISTS FACT_SALES;
    DROP TABLE IF EXISTS FACT_WASTE;
    DROP TABLE IF EXISTS FACT_INVENTORY;
    DROP TABLE IF EXISTS DIM_DATE;
    DROP TABLE IF EXISTS DIM_ITEM;
    DROP TABLE IF EXISTS DIM_INGREDIENT;
    DROP TABLE IF EXISTS DIM_WASTE_REASON;
    DROP TABLE IF EXISTS DIM_ALERT_LEVEL;

    -- ============================================================
    -- DIMENSION TABLES
    -- ============================================================
    CREATE TABLE DIM_DATE (
        date_key       INTEGER PRIMARY KEY,   -- YYYYMMDD
        date           TEXT NOT NULL,
        day_of_week    INTEGER NOT NULL,      -- 0=Mon .. 6=Sun
        day_name       TEXT NOT NULL,
        week           INTEGER NOT NULL,
        month          INTEGER NOT NULL,
        month_name     TEXT NOT NULL,
        quarter        INTEGER NOT NULL,
        year           INTEGER NOT NULL,
        is_weekend     INTEGER NOT NULL
    );

    CREATE TABLE DIM_ITEM (
        item_key       INTEGER PRIMARY KEY AUTOINCREMENT,
        item_name      TEXT NOT NULL UNIQUE,
        category       TEXT,
        price          REAL,
        cost           REAL,
        profit_margin  REAL
    );

    CREATE TABLE DIM_INGREDIENT (
        ingredient_key INTEGER PRIMARY KEY AUTOINCREMENT,
        ingredient_name TEXT NOT NULL UNIQUE,
        unit           TEXT,
        shelf_life_days INTEGER
    );

    CREATE TABLE DIM_WASTE_REASON (
        reason_key     INTEGER PRIMARY KEY AUTOINCREMENT,
        reason_name    TEXT NOT NULL UNIQUE
    );

    CREATE TABLE DIM_ALERT_LEVEL (
        alert_key      INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_name     TEXT NOT NULL UNIQUE
    );

    -- ============================================================
    -- FACT TABLES
    -- ============================================================
    CREATE TABLE FACT_SALES (
        sale_key       INTEGER PRIMARY KEY AUTOINCREMENT,
        date_key       INTEGER NOT NULL,
        item_key       INTEGER NOT NULL,
        full_datetime  TEXT,
        quantity       REAL,
        price          REAL,
        total          REAL,
        cost           REAL,
        FOREIGN KEY (date_key) REFERENCES DIM_DATE(date_key),
        FOREIGN KEY (item_key) REFERENCES DIM_ITEM(item_key)
    );

    CREATE TABLE FACT_WASTE (
        waste_key       INTEGER PRIMARY KEY AUTOINCREMENT,
        date_key        INTEGER NOT NULL,
        item_key        INTEGER NOT NULL,
        reason_key      INTEGER NOT NULL,
        quantity_wasted REAL,
        cost_per_item   REAL,
        total_waste_cost REAL,
        FOREIGN KEY (date_key) REFERENCES DIM_DATE(date_key),
        FOREIGN KEY (item_key) REFERENCES DIM_ITEM(item_key),
        FOREIGN KEY (reason_key) REFERENCES DIM_WASTE_REASON(reason_key)
    );

    CREATE TABLE FACT_INVENTORY (
        inventory_key   INTEGER PRIMARY KEY AUTOINCREMENT,
        date_key        INTEGER NOT NULL,      -- purchase_date
        ingredient_key  INTEGER NOT NULL,
        alert_key       INTEGER NOT NULL,
        quantity        REAL,
        cost_per_unit   REAL,
        total_cost      REAL,
        expiration_date TEXT,
        shelf_life_days INTEGER,
        days_until_expiration INTEGER,
        FOREIGN KEY (date_key) REFERENCES DIM_DATE(date_key),
        FOREIGN KEY (ingredient_key) REFERENCES DIM_INGREDIENT(ingredient_key),
        FOREIGN KEY (alert_key) REFERENCES DIM_ALERT_LEVEL(alert_key)
    );

    CREATE INDEX idx_sales_date ON FACT_SALES(date_key);
    CREATE INDEX idx_sales_item ON FACT_SALES(item_key);
    CREATE INDEX idx_waste_date ON FACT_WASTE(date_key);
    CREATE INDEX idx_waste_item ON FACT_WASTE(item_key);
    CREATE INDEX idx_inv_date   ON FACT_INVENTORY(date_key);
    CREATE INDEX idx_inv_ingredient ON FACT_INVENTORY(ingredient_key);
    """)
    conn.commit()
    print("Schema created: 5 dimension tables + 3 fact tables")


def date_key(d):
    """Convert a date/Timestamp to an integer YYYYMMDD key."""
    return int(pd.Timestamp(d).strftime('%Y%m%d'))


def populate_dim_date(conn, all_dates):
    """Build DIM_DATE covering the full min-max range across all datasets."""
    min_d, max_d = min(all_dates), max(all_dates)
    full_range = pd.date_range(start=min_d, end=max_d, freq='D')

    rows = []
    for d in full_range:
        rows.append((
            date_key(d), d.strftime('%Y-%m-%d'), int(d.dayofweek), d.day_name(),
            int(d.isocalendar()[1]), int(d.month), d.month_name(),
            int((d.month - 1) // 3 + 1), int(d.year), int(d.dayofweek >= 5)
        ))

    conn.executemany(
        "INSERT OR IGNORE INTO DIM_DATE VALUES (?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    print(f"DIM_DATE populated: {len(rows):,} days ({min_d.date()} to {max_d.date()})")


def populate_dim_item(conn, menu_df, sales_df, waste_df):
    """Build DIM_ITEM from the menu master list, plus any items seen in
    sales/waste but missing from the menu (edge case safety)."""
    items = menu_df[['item_name', 'category', 'price', 'cost']].copy() if menu_df is not None else pd.DataFrame()
    if len(items) > 0:
        items['profit_margin'] = (items['price'] - items['cost']).round(2)

    known_names = set(items['item_name']) if len(items) else set()

    extra_names = set()
    if sales_df is not None:
        col = 'item' if 'item' in sales_df.columns else 'item_name'
        if col in sales_df.columns:
            extra_names |= set(sales_df[col].dropna().unique()) - known_names
    if waste_df is not None and 'item_name' in waste_df.columns:
        extra_names |= set(waste_df['item_name'].dropna().unique()) - known_names

    for name in extra_names:
        items = pd.concat([items, pd.DataFrame([{
            'item_name': name, 'category': 'Other', 'price': None, 'cost': None, 'profit_margin': None
        }])], ignore_index=True)

    items = items.drop_duplicates(subset='item_name')
    items.to_sql('DIM_ITEM', conn, if_exists='append', index=False,
                 method='multi', chunksize=500)
    conn.commit()
    print(f"DIM_ITEM populated: {len(items):,} items")


def populate_dim_ingredient(conn, inventory_df):
    if inventory_df is None or 'ingredient' not in inventory_df.columns:
        print("DIM_INGREDIENT: no inventory data, skipped")
        return
    ing = inventory_df[['ingredient']].drop_duplicates().rename(columns={'ingredient': 'ingredient_name'})
    ing['unit'] = inventory_df.groupby('ingredient')['unit'].first().reindex(ing['ingredient_name']).values if 'unit' in inventory_df.columns else None
    ing['shelf_life_days'] = inventory_df.groupby('ingredient')['shelf_life_days'].first().reindex(ing['ingredient_name']).values if 'shelf_life_days' in inventory_df.columns else None
    ing.to_sql('DIM_INGREDIENT', conn, if_exists='append', index=False, method='multi', chunksize=500)
    conn.commit()
    print(f"DIM_INGREDIENT populated: {len(ing):,} ingredients")


def populate_dim_waste_reason(conn, waste_df):
    if waste_df is None or 'waste_reason' not in waste_df.columns:
        print("DIM_WASTE_REASON: no waste data, skipped")
        return
    reasons = pd.DataFrame({'reason_name': sorted(waste_df['waste_reason'].dropna().unique())})
    reasons.to_sql('DIM_WASTE_REASON', conn, if_exists='append', index=False)
    conn.commit()
    print(f"DIM_WASTE_REASON populated: {len(reasons)} reasons")


def populate_dim_alert_level(conn, inventory_df):
    if inventory_df is None or 'alert_level' not in inventory_df.columns:
        levels = ['Low Alert', 'Medium Alert', 'High Alert', 'Expired', 'Out of Stock']
    else:
        levels = sorted(inventory_df['alert_level'].dropna().unique())
    alert_df = pd.DataFrame({'alert_name': levels})
    alert_df.to_sql('DIM_ALERT_LEVEL', conn, if_exists='append', index=False)
    conn.commit()
    print(f"DIM_ALERT_LEVEL populated: {len(alert_df)} levels")


def populate_fact_sales(conn, sales_df):
    if sales_df is None or len(sales_df) == 0:
        print("FACT_SALES: no sales data, skipped")
        return

    item_map = pd.read_sql("SELECT item_key, item_name FROM DIM_ITEM", conn)
    item_lookup = dict(zip(item_map['item_name'], item_map['item_key']))

    df = sales_df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    item_col = 'item' if 'item' in df.columns else 'item_name'

    out = pd.DataFrame({
        'date_key':      df['date'].apply(date_key),
        'item_key':      df[item_col].map(item_lookup),
        'full_datetime': df['date'].astype(str),
        'quantity':      df.get('quantity'),
        'price':         df.get('price'),
        'total':         df.get('total'),
        'cost':          df.get('cost'),
    }).dropna(subset=['date_key', 'item_key'])

    out.to_sql('FACT_SALES', conn, if_exists='append', index=False, method='multi', chunksize=1000)
    conn.commit()
    print(f"FACT_SALES populated: {len(out):,} rows")


def populate_fact_waste(conn, waste_df):
    if waste_df is None or len(waste_df) == 0:
        print("FACT_WASTE: no waste data, skipped")
        return

    item_map = pd.read_sql("SELECT item_key, item_name FROM DIM_ITEM", conn)
    item_lookup = dict(zip(item_map['item_name'], item_map['item_key']))
    reason_map = pd.read_sql("SELECT reason_key, reason_name FROM DIM_WASTE_REASON", conn)
    reason_lookup = dict(zip(reason_map['reason_name'], reason_map['reason_key']))

    df = waste_df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    out = pd.DataFrame({
        'date_key':         df['date'].apply(date_key),
        'item_key':         df['item_name'].map(item_lookup),
        'reason_key':       df['waste_reason'].map(reason_lookup),
        'quantity_wasted':  df.get('quantity_wasted'),
        'cost_per_item':    df.get('cost_per_item'),
        'total_waste_cost': df.get('total_waste_cost'),
    }).dropna(subset=['date_key', 'item_key', 'reason_key'])

    out.to_sql('FACT_WASTE', conn, if_exists='append', index=False, method='multi', chunksize=1000)
    conn.commit()
    print(f"FACT_WASTE populated: {len(out):,} rows")


def populate_fact_inventory(conn, inventory_df):
    if inventory_df is None or len(inventory_df) == 0:
        print("FACT_INVENTORY: no inventory data, skipped")
        return

    ing_map = pd.read_sql("SELECT ingredient_key, ingredient_name FROM DIM_INGREDIENT", conn)
    ing_lookup = dict(zip(ing_map['ingredient_name'], ing_map['ingredient_key']))
    alert_map = pd.read_sql("SELECT alert_key, alert_name FROM DIM_ALERT_LEVEL", conn)
    alert_lookup = dict(zip(alert_map['alert_name'], alert_map['alert_key']))

    df = inventory_df.copy()
    df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')

    out = pd.DataFrame({
        'date_key':               df['purchase_date'].apply(date_key),
        'ingredient_key':         df['ingredient'].map(ing_lookup),
        'alert_key':              df['alert_level'].map(alert_lookup),
        'quantity':               df.get('quantity'),
        'cost_per_unit':          df.get('cost_per_unit'),
        'total_cost':             df.get('total_cost'),
        'expiration_date':        df.get('expiration_date').astype(str) if 'expiration_date' in df.columns else None,
        'shelf_life_days':        df.get('shelf_life_days'),
        'days_until_expiration':  df.get('days_until_expiration'),
    }).dropna(subset=['date_key', 'ingredient_key', 'alert_key'])

    out.to_sql('FACT_INVENTORY', conn, if_exists='append', index=False, method='multi', chunksize=1000)
    conn.commit()
    print(f"FACT_INVENTORY populated: {len(out):,} rows")


def main():
    print("=" * 60)
    print("Building DineData Galaxy Schema database")
    print("=" * 60)

    sales_df     = pd.read_csv(SALES_CSV)     if os.path.exists(SALES_CSV)     else None
    menu_df      = pd.read_csv(MENU_CSV)      if os.path.exists(MENU_CSV)      else None
    inventory_df = pd.read_csv(INVENTORY_CSV) if os.path.exists(INVENTORY_CSV) else None
    waste_df     = pd.read_csv(WASTE_CSV)     if os.path.exists(WASTE_CSV)     else None

    if sales_df is None and waste_df is None:
        print("ERROR: No CSV data found in Data_Cleaning/. Run the preprocessing "
              "pipeline first (data_preprocessing_kofetala.py -> feature_engineering.py).")
        return

    conn = sqlite3.connect(DB_PATH)
    build_schema(conn)

    # ── Dimensions first (facts reference them) ──
    all_dates = []
    if sales_df is not None and 'date' in sales_df.columns:
        all_dates += list(pd.to_datetime(sales_df['date'], errors='coerce').dropna())
    if waste_df is not None and 'date' in waste_df.columns:
        all_dates += list(pd.to_datetime(waste_df['date'], errors='coerce').dropna())
    if inventory_df is not None and 'purchase_date' in inventory_df.columns:
        all_dates += list(pd.to_datetime(inventory_df['purchase_date'], errors='coerce').dropna())

    if all_dates:
        populate_dim_date(conn, all_dates)
    populate_dim_item(conn, menu_df, sales_df, waste_df)
    populate_dim_ingredient(conn, inventory_df)
    populate_dim_waste_reason(conn, waste_df)
    populate_dim_alert_level(conn, inventory_df)

    # ── Facts ──
    populate_fact_sales(conn, sales_df)
    populate_fact_waste(conn, waste_df)
    populate_fact_inventory(conn, inventory_df)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)
    for table in ['DIM_DATE','DIM_ITEM','DIM_INGREDIENT','DIM_WASTE_REASON','DIM_ALERT_LEVEL',
                  'FACT_SALES','FACT_WASTE','FACT_INVENTORY']:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:<20} {count:>10,} rows")

    conn.close()
    print(f"\nDatabase written to: {DB_PATH}")
    print("Run the dashboard to explore it — go to the 'Database' page.")


if __name__ == '__main__':
    main()
