"""
================================================================================
FEATURE ENGINEERING — DINEDATA
File: feature_engineering.py
Location: Feature_Engineering/ folder
Purpose: Create all domain-specific features for machine learning
         - Rolling averages (7-day, 30-day)
         - Waste percentage estimates
         - Profit margin feature
         - Spoilage risk proxy
         - Calendar/temporal features
Author: BPSU Data Science Team
Date: June 2026

PIPELINE ORDER:
  1. convert_real_peddlr_data.py
  2. data_preprocessing_kofetala.py
  3. exploratory_data_analysis.py
  4. feature_engineering.py           ← YOU ARE HERE
  5. ml_models.py
  6. dashboard_kofetala.py

INPUT FILE:
  ../Data_Cleaning/kofe_tala_sales_data.csv

  Expected input columns:
  | date                | item                  | category | quantity | price  | total  | cost  |
  |---------------------|-----------------------|----------|----------|--------|--------|-------|
  | 2023-02-01 14:27:42 | caramel macchiato     | Coffee   | 2        | 120.00 | 240.00 | 36.00 |
  | 2023-02-01 15:10:33 | Overload lomi         | Food     | 1        | 180.00 | 180.00 | 54.00 |
  | 2023-02-02 09:22:11 | oreo overload Frappe  | Frappé   | 3        | 95.00  | 285.00 | 28.50 |
  | 2023-02-03 08:05:44 | Chicken wings (3pcs)  | Food     | 2        | 150.00 | 300.00 | 45.00 |
  | 2023-02-04 09:15:22 | Spanish latte (iced)  | Coffee   | 1        | 110.00 | 110.00 | 33.00 |

OUTPUT FILE:
  kofe_tala_features_final.csv

  Key engineered features added:
  | Feature              | Type    | Description                                    |
  |----------------------|---------|------------------------------------------------|
  | hour                 | int     | Hour of transaction (0-23)                     |
  | day_of_week          | int     | Day number (0=Mon, 6=Sun)                      |
  | is_weekend           | int     | 1 if Saturday or Sunday                        |
  | is_peak_hour         | int     | 1 if hour in [7,8,9,12,13,15,16]              |
  | profit_margin        | float   | (price - cost) / price                         |
  | waste_rate           | float   | Category waste rate (Coffee=5%, Food=15%)      |
  | estimated_waste_qty  | float   | quantity × waste_rate                          |
  | rolling_7d_avg       | float   | 7-day rolling average sales per item           |
  | rolling_30d_avg      | float   | 30-day rolling average sales per item          |
  | demand_trend         | float   | (7d_avg - 30d_avg) / 30d_avg                  |
  | spoilage_risk_num    | int     | 0=Low, 1=Medium, 2=High                        |
  | menu_performance_num | int     | 0=Reconsider, 1=Improve, 2=Keep                |
================================================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to preprocessed sales data (from Data_Cleaning folder)
DATA_FILE = '../Data_Cleaning/kofe_tala_sales_data.csv'

# Output files
OUTPUT_MAIN       = 'kofe_tala_features_final.csv'
OUTPUT_DAILY      = 'agg_daily.csv'
OUTPUT_ITEM       = 'agg_item.csv'
OUTPUT_CATEGORY   = 'agg_category.csv'
OUTPUT_HOURLY     = 'agg_hourly.csv'

# Waste rate assumptions per category (industry standard proxies)
# Used because historical waste logs were not available before this study
# Waste rates per category based on actual Kôfētala menu
# | Category       | Waste Rate | Reason                              |
# |----------------|------------|-------------------------------------|
# | Coffee Based   | 5%         | Made to order, low waste            |
# | Signature Kôfē | 6%         | Specialty drinks, slightly higher   |
# | Kôfē Frappé    | 8%         | Blended drinks, some over-prep      |
# | Sans Coffee    | 7%         | Non-coffee beverages                |
# | Mini Bites     | 15%        | Food items, highest perishability   |
# | Other          | 10%        | Default rate                        |
WASTE_RATE = {
    'Coffee Based':   0.05,
    'Signature Kôfē': 0.06,
    'Kôfē Frappé':    0.08,
    'Sans Coffee':    0.07,
    'Mini Bites':     0.15,
    'Other':          0.10,
}

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_data():
    print("=" * 80)
    print("LOADING CLEANED DATA")
    print("=" * 80)

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Cannot find: {DATA_FILE}\n"
            "Run: python ../Data_Cleaning/data_preprocessing_kofetala.py first."
        )

    df = pd.read_csv(DATA_FILE)

    # Ensure datetime columns
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Ensure required numeric columns exist
    for col in ['quantity', 'price', 'cost', 'total']:
        if col not in df.columns:
            df[col] = np.nan

    print(f"✅ Loaded {len(df):,} records | {len(df.columns)} columns")
    print(f"   Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"   Columns: {df.columns.tolist()}")
    return df

# ============================================================================
# STEP 2: TEMPORAL FEATURES
# ============================================================================

def create_temporal_features(df):
    """
    Extract calendar-based features from transaction datetime.
    Key for capturing seasonality and weekly demand patterns.
    """
    print("\n" + "=" * 80)
    print("CREATING TEMPORAL FEATURES")
    print("=" * 80)

    df['hour']        = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek          # 0=Monday … 6=Sunday
    df['day_name']    = df['date'].dt.day_name()
    df['month']       = df['date'].dt.month
    df['month_name']  = df['date'].dt.month_name()
    df['quarter']     = df['date'].dt.quarter
    df['week_number'] = df['date'].dt.isocalendar().week.astype(int)
    df['date_only']   = df['date'].dt.date

    # Binary flags
    df['is_weekend']   = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_peak_hour'] = df['hour'].isin([7, 8, 9, 12, 13, 15, 16]).astype(int)

    # Season (Philippines: dry = Dec–May, wet = Jun–Nov)
    df['is_dry_season'] = df['month'].isin([12, 1, 2, 3, 4, 5]).astype(int)

    print("✅ Temporal features created:")
    print("   hour, day_of_week, day_name, month, month_name, quarter,")
    print("   week_number, is_weekend, is_peak_hour, is_dry_season")
    return df

# ============================================================================
# STEP 3: PROFIT MARGIN FEATURES
# ============================================================================

def create_profit_features(df):
    """
    Compute profit margin per transaction.
    Used as a key feature in menu performance classification.
    Formula: profit_margin = (price - cost) / price
    """
    print("\n" + "=" * 80)
    print("CREATING PROFIT MARGIN FEATURES")
    print("=" * 80)

    # Gross profit per unit
    df['gross_profit'] = df['price'] - df['cost']

    # Profit margin ratio (0–1 scale)
    df['profit_margin'] = np.where(
        df['price'] > 0,
        (df['price'] - df['cost']) / df['price'],
        0.0
    ).round(4)

    # Revenue per transaction
    df['revenue'] = df['quantity'] * df['price']

    # Cost per transaction
    df['total_cost'] = df['quantity'] * df['cost']

    # Profit per transaction
    df['transaction_profit'] = df['revenue'] - df['total_cost']

    print("✅ Profit features created:")
    print("   gross_profit, profit_margin, revenue, total_cost, transaction_profit")
    return df

# ============================================================================
# STEP 4: WASTE ESTIMATION FEATURES
# ============================================================================

def create_waste_features(df):
    """
    Estimate waste quantities using category-based waste rates.
    Required because detailed waste logs were not kept historically.
    Used as target variable for waste prediction model.
    """
    print("\n" + "=" * 80)
    print("CREATING WASTE ESTIMATION FEATURES")
    print("=" * 80)

    # Apply category waste rate
    df['waste_rate'] = df['category'].map(WASTE_RATE).fillna(0.10)

    # Estimated waste quantity per transaction
    df['estimated_waste_qty'] = (df['quantity'] * df['waste_rate']).round(2)

    # Estimated waste cost
    df['estimated_waste_cost'] = (df['estimated_waste_qty'] * df['cost']).round(2)

    # Waste percentage label (for classification)
    # High waste = items with waste rate > 10%
    df['is_high_waste_item'] = (df['waste_rate'] > 0.10).astype(int)

    print(f"✅ Waste features created:")
    print(f"   waste_rate, estimated_waste_qty, estimated_waste_cost, is_high_waste_item")
    print(f"\n   Waste rates applied:")
    for cat, rate in WASTE_RATE.items():
        count = (df['category'] == cat).sum()
        print(f"   {cat}: {rate*100:.0f}% ({count:,} records)")
    return df

# ============================================================================
# STEP 5: ROLLING SALES AVERAGE FEATURES
# ============================================================================

def create_rolling_features(df):
    """
    Compute 7-day and 30-day rolling average sales per item.
    These are the most important features for demand forecasting.
    Based on Cerqueira et al. (2025) - lag and rolling window features.
    """
    print("\n" + "=" * 80)
    print("CREATING ROLLING SALES AVERAGE FEATURES")
    print("=" * 80)

    # Aggregate to daily sales per item first
    daily_item = (
        df.groupby(['date_only', 'item'])['quantity']
        .sum()
        .reset_index()
        .rename(columns={'quantity': 'daily_qty'})
    )
    daily_item['date_only'] = pd.to_datetime(daily_item['date_only'])
    daily_item = daily_item.sort_values(['item', 'date_only'])

    # --- 7-day rolling average ---
    daily_item['rolling_7d_avg'] = (
        daily_item.groupby('item')['daily_qty']
        .transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
        .round(2)
    )

    # --- 30-day rolling average ---
    daily_item['rolling_30d_avg'] = (
        daily_item.groupby('item')['daily_qty']
        .transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean())
        .round(2)
    )

    # --- Lag feature: yesterday's sales ---
    daily_item['lag_1d'] = (
        daily_item.groupby('item')['daily_qty']
        .transform(lambda x: x.shift(1))
        .round(2)
    )

    # --- Trend: 7d avg vs 30d avg ---
    daily_item['demand_trend'] = (
        (daily_item['rolling_7d_avg'] - daily_item['rolling_30d_avg'])
        / (daily_item['rolling_30d_avg'] + 1e-6)
    ).round(4)

    # Merge rolling features back into transaction-level dataframe
    df['date_only'] = pd.to_datetime(df['date_only'])
    df = df.merge(
        daily_item[['date_only', 'item', 'daily_qty',
                    'rolling_7d_avg', 'rolling_30d_avg',
                    'lag_1d', 'demand_trend']],
        on=['date_only', 'item'],
        how='left'
    )

    # Fill NaN rolling features for first days with item mean
    for col in ['rolling_7d_avg', 'rolling_30d_avg', 'lag_1d', 'demand_trend']:
        item_means = df.groupby('item')[col].transform('mean')
        df[col] = df[col].fillna(item_means).fillna(0).round(4)

    print("✅ Rolling features created:")
    print("   daily_qty, rolling_7d_avg, rolling_30d_avg, lag_1d, demand_trend")
    return df

# ============================================================================
# STEP 6: SPOILAGE RISK PROXY FEATURES
# ============================================================================

def create_spoilage_features(df):
    """
    Create a proxy spoilage risk label based on category waste rate
    and rolling demand trend.
    Labels: High Risk (2), Medium Risk (1), Low Risk (0)
    Used as target for spoilage classification model.
    """
    print("\n" + "=" * 80)
    print("CREATING SPOILAGE RISK PROXY FEATURES")
    print("=" * 80)

    def spoilage_risk_label(row):
        """Rule-based proxy using waste rate and demand trend."""
        waste  = row.get('waste_rate', 0.10)
        trend  = row.get('demand_trend', 0.0)
        # Falling demand + high waste category = highest risk
        if waste >= 0.15 and trend < -0.10:
            return 2   # High Risk
        elif waste >= 0.10 or trend < 0:
            return 1   # Medium Risk
        else:
            return 0   # Low Risk

    df['spoilage_risk_num']   = df.apply(spoilage_risk_label, axis=1)
    df['spoilage_risk_label'] = df['spoilage_risk_num'].map(
        {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
    )

    dist = df['spoilage_risk_label'].value_counts()
    print("✅ Spoilage risk proxy created:")
    for label, count in dist.items():
        print(f"   {label}: {count:,} records ({count/len(df)*100:.1f}%)")
    return df

# ============================================================================
# STEP 7: MENU PERFORMANCE CLASSIFICATION TARGET
# ============================================================================

def create_menu_performance_target(df):
    """
    Assign menu performance label to each item:
      Keep      – high sales, good margin, low waste
      Improve   – moderate performance
      Reconsider– low sales or low margin
    Used as target for menu classification model.
    """
    print("\n" + "=" * 80)
    print("CREATING MENU PERFORMANCE TARGET")
    print("=" * 80)

    # Item-level summary
    item_summary = df.groupby('item').agg(
        total_qty     = ('quantity',       'sum'),
        avg_margin    = ('profit_margin',  'mean'),
        avg_waste     = ('waste_rate',     'mean'),
        total_revenue = ('revenue',        'sum')
    ).reset_index()

    # Percentile thresholds
    qty_75    = item_summary['total_qty'].quantile(0.75)
    qty_25    = item_summary['total_qty'].quantile(0.25)
    margin_50 = item_summary['avg_margin'].quantile(0.50)

    def classify_menu(row):
        if row['total_qty'] >= qty_75 and row['avg_margin'] >= margin_50:
            return 'Keep'
        elif row['total_qty'] >= qty_25:
            return 'Improve'
        else:
            return 'Reconsider'

    item_summary['menu_performance'] = item_summary.apply(classify_menu, axis=1)

    # Merge back
    df = df.merge(
        item_summary[['item', 'menu_performance']],
        on='item',
        how='left'
    )
    df['menu_performance'] = df['menu_performance'].fillna('Improve')

    # Encode as number for ML
    perf_map = {'Keep': 2, 'Improve': 1, 'Reconsider': 0}
    df['menu_performance_num'] = df['menu_performance'].map(perf_map)

    dist = df['menu_performance'].value_counts()
    print("✅ Menu performance target created:")
    for label, count in dist.items():
        print(f"   {label}: {count:,} records ({count/len(df)*100:.1f}%)")
    return df

# ============================================================================
# STEP 8: CATEGORICAL ENCODING
# ============================================================================

def encode_categoricals(df):
    """One-hot encode category; label-encode item."""
    print("\n" + "=" * 80)
    print("ENCODING CATEGORICAL VARIABLES")
    print("=" * 80)

    # Label encode item (keeps single column, needed for rolling merge)
    df['item_encoded'] = pd.factorize(df['item'])[0]

    # One-hot encode category
    cat_dummies = pd.get_dummies(df['category'], prefix='cat').astype(int)
    df = pd.concat([df, cat_dummies], axis=1)

    print(f"✅ Encoded 'item' → item_encoded")
    print(f"✅ One-hot encoded 'category' → {list(cat_dummies.columns)}")
    return df

# ============================================================================
# STEP 9: AGGREGATIONS
# ============================================================================

def create_aggregations(df):
    """Create summary-level datasets for analysis."""
    print("\n" + "=" * 80)
    print("CREATING AGGREGATION DATASETS")
    print("=" * 80)

    aggs = {}

    # Daily totals
    aggs['daily'] = df.groupby('date_only').agg(
        total_records      = ('item',                    'count'),
        total_qty          = ('quantity',                'sum'),
        total_revenue      = ('revenue',                 'sum'),
        total_cost         = ('total_cost',              'sum'),
        total_waste_cost   = ('estimated_waste_cost',    'sum'),
        avg_profit_margin  = ('profit_margin',           'mean'),
    ).reset_index()

    # Item totals
    aggs['item'] = df.groupby('item').agg(
        category          = ('category',             'first'),
        total_qty         = ('quantity',             'sum'),
        total_revenue     = ('revenue',              'sum'),
        avg_margin        = ('profit_margin',        'mean'),
        avg_waste_rate    = ('waste_rate',           'mean'),
        menu_performance  = ('menu_performance',     'first'),
    ).reset_index()

    # Category totals
    aggs['category'] = df.groupby('category').agg(
        total_qty       = ('quantity',             'sum'),
        total_revenue   = ('revenue',              'sum'),
        avg_margin      = ('profit_margin',        'mean'),
        record_count    = ('item',                 'count'),
    ).reset_index()

    # Hourly totals
    aggs['hourly'] = df.groupby('hour').agg(
        total_qty     = ('quantity',  'sum'),
        record_count  = ('item',      'count'),
    ).reset_index()

    for name, adf in aggs.items():
        print(f"✅ {name}: {len(adf)} rows")

    return aggs

# ============================================================================
# STEP 10: SAVE OUTPUTS
# ============================================================================

def save_outputs(df, aggs):
    print("\n" + "=" * 80)
    print("SAVING OUTPUT FILES")
    print("=" * 80)

    df.to_csv(OUTPUT_MAIN, index=False)
    print(f"✅ {OUTPUT_MAIN}  →  {len(df):,} records, {len(df.columns)} features")

    for name, adf in aggs.items():
        fname = f'agg_{name}.csv'
        adf.to_csv(fname, index=False)
        print(f"✅ agg_{name}.csv  →  {len(adf)} rows")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "█" * 80)
    print("  FEATURE ENGINEERING — DineData Kôfētala Bistro")
    print("█" * 80)

    try:
        df = load_data()
        df = create_temporal_features(df)
        df = create_profit_features(df)
        df = create_waste_features(df)
        df = create_rolling_features(df)
        df = create_spoilage_features(df)
        df = create_menu_performance_target(df)
        df = encode_categoricals(df)

        aggs = create_aggregations(df)
        save_outputs(df, aggs)

        print("\n" + "█" * 80)
        print("✅ FEATURE ENGINEERING COMPLETE!")
        print(f"   Total features in output: {len(df.columns)}")
        print("   Key ML features ready:")
        print("     • Demand target    → daily_qty")
        print("     • Waste target     → estimated_waste_qty")
        print("     • Menu target      → menu_performance / menu_performance_num")
        print("     • Spoilage target  → spoilage_risk_num / spoilage_risk_label")
        print("\n🚀 Next: python ../ML_Models/ml_models.py")
        print("█" * 80)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
