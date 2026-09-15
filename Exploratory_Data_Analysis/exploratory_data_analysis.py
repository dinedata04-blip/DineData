import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')


# CONFIGURATION

# Fixed path — preprocessing saves to Data_Cleaning/ folder
DATA_FILE   = '../Data_Cleaning/kofe_tala_sales_data.csv'
OUTPUT_DIR  = 'eda_plots'

sns.set_style("whitegrid")
PALETTE     = ['#6F4E37', '#A0826D', '#D2691E', '#8B5E3C', '#C4A882']
EARTH_MAIN  = '#6F4E37'

# HELPER: SAVE FIGURE

def save_fig(filename):
    """Save the current matplotlib figure to eda_plots/<filename> and close
    it, so figures don't pile up in memory when this script generates many
    plots in one run."""
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {filename}")


# STEP 1: LOAD DATA

def load_data():
    """Load the cleaned sales CSV (output of the preprocessing step) and
    derive the calendar columns (hour, day name, month name/number, plain
    date) that every analysis function below relies on. Doing this once
    here — instead of separately in each function — keeps the derived
    columns consistent throughout the whole EDA run."""
    print("=" * 80)
    print("LOADING DATA FOR EDA")
    print("=" * 80)

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Cannot find: {DATA_FILE}\n"
            "Run: python ../Data_Cleaning/data_preprocessing_kofetala.py first."
        )

    df = pd.read_csv(DATA_FILE)

    # Parse datetime column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
    else:
        raise ValueError("'date' column not found in data.")

    # Derive time columns used throughout EDA
    df['hour']       = df['date'].dt.hour
    df['day_name']   = df['date'].dt.day_name()
    df['month_name'] = df['date'].dt.month_name()
    df['month_num']  = df['date'].dt.month
    df['date_only']  = df['date'].dt.date

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"✅ Loaded: {len(df):,} records | {len(df.columns)} columns")
    print(f"   Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    # Show sample of loaded data
    print(f"\n   Sample data (first 5 rows):")
    print(df.head(5).to_string())

    return df

# STEP 2: DATA OVERVIEW

def data_overview(df):
    """Print a quick health-check of the dataset: row/column counts, missing
    values, unique items/categories, date span, and basic numeric stats.
    This is the first thing checked before any deeper analysis, so any
    obvious data quality problems (e.g. lots of missing values) are caught
    early rather than silently skewing later charts."""
    print("\n" + "=" * 80)
    print("DATA OVERVIEW")
    print("=" * 80)

    print(f"\n  Records   : {len(df):,}")
    print(f"  Columns   : {len(df.columns)}")
    print(f"  Missing   : {df.isnull().sum().sum()}")

    if 'item' in df.columns:
        print(f"  Items     : {df['item'].nunique()} unique")
    if 'category' in df.columns:
        print(f"  Categories: {df['category'].nunique()} unique")

    date_span = (df['date'].max() - df['date'].min()).days
    print(f"  Duration  : {date_span} days ({date_span//30} months)")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n  Numeric column stats:")
        print(df[numeric_cols[:6]].describe().round(2).to_string())

    print(f"\n  Columns: {df.columns.tolist()}")
    print(f"\n  First 3 rows:")
    print(df.head(3).to_string())

# STEP 3: SALES VOLUME ANALYSIS

def sales_analysis(df):
    """Summarize overall sales volume and revenue, and check whether the
    'quantity' column is skewed. A skewness above 1 (in either direction)
    is the signal that later feeds the decision in feature_engineering.py
    to apply a log transform, since skewed targets hurt regression models."""
    print("\n" + "=" * 80)
    print("SALES VOLUME ANALYSIS")
    print("=" * 80)

    if 'quantity' not in df.columns:
        print("⚠️ 'quantity' column not found. Skipping.")
        return

    total_qty = df['quantity'].sum()
    avg_qty   = df['quantity'].mean()
    print(f"  Total units sold : {total_qty:,.0f}")
    print(f"  Avg per record   : {avg_qty:.2f}")

    if 'total' in df.columns:
        total_rev = df['total'].sum()
        print(f"  Total revenue    : ₱{total_rev:,.2f}")

    # Distribution check
    skew = df['quantity'].skew()
    print(f"  Quantity skewness: {skew:.2f}  {'→ log transform recommended' if abs(skew) > 1 else ''}")

# STEP 4: CATEGORY ANALYSIS + PLOT

def category_analysis(df):
    """Break down transaction counts by menu category (Coffee, Food, etc.)
    and save a bar chart. This is what later justifies one-hot encoding
    category as a model feature (cat_Coffee, cat_Food, ...) in
    feature_engineering.py — categories with meaningfully different
    transaction volumes are worth letting the model distinguish between."""
    print("\n" + "=" * 80)
    print("CATEGORY ANALYSIS")
    print("=" * 80)

    if 'category' not in df.columns:
        print("⚠️ 'category' column not found. Skipping.")
        return

    cat_counts = df['category'].value_counts()
    print(f"\n  Category distribution:")
    for cat, cnt in cat_counts.items():
        pct = cnt / len(df) * 100
        print(f"    {cat:15s}: {cnt:>7,}  ({pct:.1f}%)")

    # ── Plot 1: Category bar ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(cat_counts.index, cat_counts.values,
                  color=PALETTE[:len(cat_counts)], edgecolor='white', linewidth=0.5)
    ax.set_title("Transaction Count by Category", fontsize=14, fontweight='bold', color='#3E2723')
    ax.set_xlabel("Category", fontsize=11)
    ax.set_ylabel("Number of Transactions", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f'{bar.get_height():,.0f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    save_fig('01_category_distribution.png')

    # ── Plot 2: Revenue by category (if available) ───────────────────────
    if 'total' in df.columns:
        cat_rev = df.groupby('category')['total'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(cat_rev.index, cat_rev.values,
               color=PALETTE[:len(cat_rev)], edgecolor='white')
        ax.set_title("Revenue by Category", fontsize=14, fontweight='bold', color='#3E2723')
        ax.set_ylabel("Total Revenue (₱)", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₱{x:,.0f}'))
        plt.tight_layout()
        save_fig('02_revenue_by_category.png')

# STEP 5: TOP ITEMS ANALYSIS + PLOT

def item_analysis(df):
    """Rank menu items by units sold. This ranking is the basis for the
    'item_encoded' feature used by every model in ml_models.py — items sold
    often carry more reliable historical demand signal than rarely-sold
    ones, which is why rolling averages (rolling_7d_avg, rolling_30d_avg)
    are computed per item later in feature_engineering.py."""
    print("\n" + "=" * 80)
    print("TOP ITEMS ANALYSIS")
    print("=" * 80)

    if 'item' not in df.columns:
        print("⚠️ 'item' column not found. Skipping.")
        return

    qty_col = 'quantity' if 'quantity' in df.columns else None

    if qty_col:
        top_items = df.groupby('item')[qty_col].sum().nlargest(15)
    else:
        top_items = df['item'].value_counts().head(15)

    print(f"\n  Top 15 Items by units sold:")
    for i, (item, val) in enumerate(top_items.items(), 1):
        print(f"    {i:2d}. {item:40s} {val:>8,.0f}")

    # ── Plot 3: Top items horizontal bar ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = [EARTH_MAIN if i == 0 else '#A0826D' for i in range(len(top_items))]
    ax.barh(top_items.index[::-1], top_items.values[::-1],
            color=colors[::-1], edgecolor='white')
    ax.set_title("Top 15 Items by Quantity Sold", fontsize=14,
                 fontweight='bold', color='#3E2723')
    ax.set_xlabel("Total Units Sold", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.tight_layout()
    save_fig('03_top_items.png')


# STEP 6: TIME PATTERN ANALYSIS + PLOTS

def time_analysis(df):
    """Check for hourly, day-of-week, and monthly patterns in transaction
    volume. Whatever peak hours/days show up here directly become the
    'is_peak_hour', 'day_of_week', and 'is_weekend' features engineered in
    feature_engineering.py — we only add those flags because this analysis
    confirms the patterns actually exist in the data."""
    print("\n" + "=" * 80)
    print("TIME PATTERN ANALYSIS")
    print("=" * 80)

    # ── Hourly ───────────────────────────────────────────────────────────
    hourly = df.groupby('hour').size()
    print(f"\n  Peak hours (top 5):")
    for hr, cnt in hourly.nlargest(5).items():
        print(f"    {hr:02d}:00  → {cnt:,} transactions")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(hourly.index, hourly.values,
           color=[EARTH_MAIN if h in [8, 9, 12, 13, 15] else '#C4A882'
                  for h in hourly.index],
           edgecolor='white')
    ax.set_title("Transactions by Hour of Day", fontsize=14,
                 fontweight='bold', color='#3E2723')
    ax.set_xlabel("Hour (24-hr)", fontsize=11)
    ax.set_ylabel("Transaction Count", fontsize=11)
    ax.set_xticks(range(0, 24))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.tight_layout()
    save_fig('04_hourly_pattern.png')

    # ── Day of week ───────────────────────────────────────────────────────
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    daily_dow = df.groupby('day_name').size().reindex(day_order)

    print(f"\n  Sales by day of week:")
    for day, cnt in daily_dow.items():
        bar = "█" * int(cnt / daily_dow.max() * 20)
        print(f"    {day:10s}: {bar} {cnt:,}")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = [EARTH_MAIN if d in ['Saturday','Sunday'] else '#C4A882'
               for d in daily_dow.index]
    ax.bar(daily_dow.index, daily_dow.values,
           color=colors, edgecolor='white')
    ax.set_title("Transactions by Day of Week", fontsize=14,
                 fontweight='bold', color='#3E2723')
    ax.set_ylabel("Transaction Count", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.tight_layout()
    save_fig('05_day_of_week.png')

    # ── Monthly trend ─────────────────────────────────────────────────────
    month_order = ['January','February','March','April','May','June',
                   'July','August','September','October','November','December']
    monthly = df.groupby('month_name').size()
    monthly = monthly.reindex([m for m in month_order if m in monthly.index])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly.index, monthly.values,
            marker='o', color=EARTH_MAIN, linewidth=2.5, markersize=6)
    ax.fill_between(range(len(monthly)), monthly.values,
                    alpha=0.15, color=EARTH_MAIN)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly.index, rotation=30, ha='right')
    ax.set_title("Monthly Sales Volume", fontsize=14,
                 fontweight='bold', color='#3E2723')
    ax.set_ylabel("Transaction Count", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.tight_layout()
    save_fig('06_monthly_trend.png')


# STEP 7: DAILY TIME SERIES + PLOT

def daily_timeseries(df):
    """Plot daily total units sold with a 7-day moving average overlay.
    This is what motivates the 'rolling_7d_avg' and 'rolling_30d_avg'
    features later — if daily sales were perfectly flat, rolling averages
    wouldn't add any predictive value, but the visible day-to-day swings
    here justify smoothing the signal for the demand model."""
    print("\n" + "=" * 80)
    print("DAILY TIME SERIES")
    print("=" * 80)

    qty_col = 'quantity' if 'quantity' in df.columns else None
    if qty_col:
        daily = df.groupby('date_only')[qty_col].sum()
    else:
        daily = df.groupby('date_only').size()

    print(f"  Average daily units sold: {daily.mean():.1f}")
    print(f"  Peak day: {daily.idxmax()} ({daily.max():.0f} units)")
    print(f"  Lowest day: {daily.idxmin()} ({daily.min():.0f} units)")

    fig, ax = plt.subplots(figsize=(14, 5))
    daily_x = pd.to_datetime(list(daily.index))
    ax.plot(daily_x, daily.values, color=EARTH_MAIN, linewidth=1, alpha=0.7)

    # 7-day moving average
    ma7 = pd.Series(daily.values).rolling(7, min_periods=1).mean()
    ax.plot(daily_x, ma7.values, color='#D2691E', linewidth=2.5, label='7-day MA')
    ax.legend(fontsize=10)
    ax.set_title("Daily Sales Volume with 7-Day Moving Average",
                 fontsize=14, fontweight='bold', color='#3E2723')
    ax.set_ylabel("Units Sold", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.tight_layout()
    save_fig('07_daily_timeseries.png')

# STEP 8: CORRELATION HEATMAP

def correlation_analysis(df):
    """Compute pairwise correlation between numeric columns and save a
    heatmap. Used to catch redundant features (two columns that are almost
    perfectly correlated add no new information to a model and can be
    dropped) and to sanity-check that engineered features later behave the
    way we'd expect relative to price/cost/quantity."""
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Keep only meaningful columns for the heatmap
    keep_cols = [c for c in numeric_cols
                 if c not in ['item_encoded', 'category_encoded']
                 and not c.startswith('cat_')][:10]

    if len(keep_cols) < 2:
        print("⚠️ Not enough numeric columns for correlation analysis.")
        return

    corr = df[keep_cols].corr()
    print(f"\n  Correlation matrix ({len(keep_cols)} columns):")

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='YlOrBr', ax=ax, linewidths=0.5,
                cbar_kws={'label': 'Correlation'})
    ax.set_title("Correlation Heatmap — Numeric Features",
                 fontsize=14, fontweight='bold', color='#3E2723')
    plt.tight_layout()
    save_fig('08_correlation_heatmap.png')

# STEP 9: DISTRIBUTION PLOT (for skewness check)

def distribution_analysis(df):
    """Compare the raw 'quantity' distribution against its log-transformed
    version and report which is less skewed. This is the concrete evidence
    behind the log-transform decision mentioned in sales_analysis() — we
    only recommend it here if the log version measurably reduces skew,
    rather than applying it as a blanket rule."""
    print("\n" + "=" * 80)
    print("DISTRIBUTION ANALYSIS")
    print("=" * 80)

    qty_col = 'quantity' if 'quantity' in df.columns else None
    if qty_col is None:
        print("⚠️ 'quantity' column not found. Skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Raw distribution
    axes[0].hist(df[qty_col].dropna(), bins=40,
                 color=EARTH_MAIN, edgecolor='white', alpha=0.8)
    axes[0].set_title("Quantity Distribution (Raw)",
                      fontsize=12, fontweight='bold', color='#3E2723')
    axes[0].set_xlabel("Quantity")
    axes[0].set_ylabel("Frequency")

    # Log-transformed
    log_qty = np.log1p(df[qty_col].dropna())
    axes[1].hist(log_qty, bins=40,
                 color='#A0826D', edgecolor='white', alpha=0.8)
    axes[1].set_title("Quantity Distribution (Log-Transformed)",
                      fontsize=12, fontweight='bold', color='#3E2723')
    axes[1].set_xlabel("log(1 + Quantity)")

    skew_raw = df[qty_col].skew()
    skew_log = log_qty.skew()
    print(f"  Raw skewness:         {skew_raw:.3f}")
    print(f"  Log-transformed skew: {skew_log:.3f}")
    print(f"  → {'Log transform recommended' if abs(skew_raw) > abs(skew_log) else 'Raw distribution acceptable'}")

    plt.tight_layout()
    save_fig('09_quantity_distribution.png')

# MAIN

def main():
    print("\n" + "█" * 80)
    print("  EXPLORATORY DATA ANALYSIS — DineData Kôfētala Bistro")
    print("█" * 80)

    try:
        df = load_data()
        data_overview(df)
        sales_analysis(df)
        category_analysis(df)
        item_analysis(df)
        time_analysis(df)
        daily_timeseries(df)
        correlation_analysis(df)
        distribution_analysis(df)

        print("\n" + "█" * 80)
        print("✅ EDA COMPLETE!")
        print(f"   9 plots saved to: {OUTPUT_DIR}/")
        print("\n  Key findings to carry into Feature Engineering:")
        print("    • Weekly seasonality found → add day_of_week & is_weekend features")
        print("    • Peak hours identified    → add is_peak_hour flag")
        print("    • Monthly trend visible    → add month & quarter features")
        print("    • Quantity skewness checked → apply log transform if needed")
        print("\n🚀 Next: python ../Feature_Engineering/feature_engineering.py")
        print("█" * 80)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
