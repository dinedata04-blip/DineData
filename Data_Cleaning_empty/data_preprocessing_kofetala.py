import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 60)
print("CONVERTING REAL KÔFĒTALA PEDDLR DATA")
print("=" * 60)

# LOAD RAW PEDDLR DATA

print("\n📂 Loading real Peddlr data...")

df = pd.read_csv('../Product_Sales_79966_2026-02-17-214352.csv')
print(f"✅ Loaded {len(df):,} sales records")

# Parse DATETIME — format='mixed' suppresses UserWarning
df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='mixed', errors='coerce')
df = df.dropna(subset=['DATETIME'])
print(f"   From: {df['DATETIME'].min()}")
print(f"   To:   {df['DATETIME'].max()}")

# Force numeric columns to numbers — any corrupted/garbled values
# (bad export characters, binary junk, etc.) become NaN instead of crashing
for col in ['PRICE', 'COST', 'QTY']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows where the core numeric fields are unusable
before = len(df)
df = df.dropna(subset=['PRICE', 'QTY'])
dropped = before - len(df)
if dropped > 0:
    print(f"⚠️  Dropped {dropped} corrupted row(s) with unreadable PRICE/QTY/COST data")

# ============================================================================
# CONVERT TO DINEDATA FORMAT
#
# Category assignment rules:
# | Keyword in product name              | Category assigned |
# |--------------------------------------|-------------------|
# | mocha, latte, americano, cappuccino  | Coffee            |
# | frappe, shake, smoothie              | Frappé            |
# | tea, milk tea                        | Non-Coffee        |
# | chicken, fries, bff, ramen, lomi     | Food              |
# | (everything else)                    | Non-Coffee        |
# ============================================================================

print("\n🔄 Converting to DineData format...")

sales_data = []

for idx, row in df.iterrows():
    dt      = row['DATETIME']
    product = str(row['PRODUCT']).lower()
    variant = str(row['VARIANT']).lower() if pd.notna(row['VARIANT']) else ''

    # ── Category rules based on actual Kôfētala menu board ──────────────
    #
    # | Category       | Keywords / Items                                      |
    # |----------------|-------------------------------------------------------|
    # | Coffee Based   | americano, cappuccino, mocha, latte, espresso,        |
    # |                | spanish latte, caramel macchiato                      |
    # | Signature Kôfē | honey shaken, supercharged, whipped, christmas,       |
    # |                | dalgona, matcha espresso fusion                       |
    # | Kôfē Frappé    | frappe, frappé, super mocha, triple caramel,          |
    # |                | blended capp, dirty coffee shake                      |
    # | Sans Coffee    | milk tea, choco dinosaur, oreo overload, strawberry   |
    # |                | latte, matcha supreme, mango banana, sparkling,       |
    # |                | hot cocoa, citron, strawberry reverie, sm tea         |
    # | Mini Bites     | chicken, fries, bff, ramen, lomi, panini, chiz,       |
    # |                | noodles, pops                                         |

    if any(w in product for w in [
        'honey shaken','supercharged','whipped americano',
        'christmas drink','dalgona','matcha espresso fusion']):
        category = 'Signature Kôfē'

    elif any(w in product for w in [
        'frappe','frappé','super mocha','triple caramel',
        'blended capp','dirty coffee','oreo overload frappe',
        'shake']):
        category = 'Kôfē Frappé'

    elif any(w in product for w in [
        'milk tea','choco dinosaur','oreo overload',
        'strawberry latte','matcha supreme','mango banana',
        'sparkling berry','hot cocoa','citron','strawberry reverie',
        'sm tea','simple milk','special blend']):
        category = 'Sans Coffee'

    elif any(w in product for w in [
        'chicken','fries','bff','ramen','lomi','panini',
        'chiz','noodles','pops','batangas','wing']):
        category = 'Mini Bites'

    elif any(w in product for w in [
        'americano','cappuccino','mocha','espresso',
        'spanish latte','caramel macchiato','latte']):
        category = 'Coffee Based'

    else:
        category = 'Sans Coffee'  # default for unmatched beverages

    if variant and variant not in ['nan', 'none']:
        item_name = f"{row['PRODUCT']} ({variant})"
    else:
        item_name = row['PRODUCT']

    # Use 30% of price as cost estimate if COST is missing or zero
    cost_val = float(row['COST']) \
        if pd.notna(row['COST']) and row['COST'] > 0 \
        else float(row['PRICE']) * 0.3

    sales_data.append({
        'date':        dt,
        'day_of_week': dt.strftime('%A'),
        'item':        item_name,
        'item_name':   item_name,
        'category':    category,
        'quantity':    int(row['QTY']),
        'price':       float(row['PRICE']),
        'total':       float(row['TOTAL PRODUCT PRICE']),
        'cost':        cost_val,
        'is_weekend':  dt.strftime('%A') in ['Saturday', 'Sunday'],
    })

sales_df = pd.DataFrame(sales_data)
sales_df['hour']         = sales_df['date'].dt.hour
sales_df['month']        = sales_df['date'].dt.month
sales_df['month_name']   = sales_df['date'].dt.month_name()
sales_df['week']         = sales_df['date'].dt.isocalendar().week
sales_df['is_peak_hour'] = sales_df['hour'].isin([8, 9, 12, 13, 15, 16])

print(f"✅ Converted {len(sales_df):,} sales records")

# ============================================================================
# CREATE MENU REFERENCE
#
# Sample output:
# | item_name             | category | price | cost | profit_margin | prep_time_minutes |
# |-----------------------|----------|-------|------|---------------|-------------------|
# | caramel macchiato     | Coffee   | 120   | 36   | 84            | 5                 |
# | Overload lomi         | Food     | 180   | 54   | 126           | 5                 |
# ============================================================================

print("\n📋 Creating menu reference...")

menu_items = sales_df.groupby('item_name').agg({
    'category': 'first',
    'price':    'mean',
    'cost':     'mean',
    'quantity': 'sum',
}).reset_index()

menu_items['price']             = menu_items['price'].round(0).astype(int)
menu_items['cost']              = menu_items['cost'].round(0).astype(int)
menu_items['profit_margin']     = menu_items['price'] - menu_items['cost']
menu_items['prep_time_minutes'] = 5

menu_df = menu_items[['item_name','category','price','cost',
                       'prep_time_minutes','profit_margin']].copy()

print(f"✅ Created menu with {len(menu_df)} unique items")

# ============================================================================
# GENERATE INVENTORY DATA
#
# Ingredients with shelf life:
# | ingredient    | unit   | shelf_life_days | cost_per_unit |
# |---------------|--------|-----------------|---------------|
# | Coffee Beans  | kg     | 60              | 850           |
# | Milk          | liter  | 7               | 90            |
# | Chicken       | kg     | 5               | 300           |
#
# Alert Level rules:
# | Condition              | Alert Level  |
# |------------------------|--------------|
# | quantity <= 0          | Out of Stock |
# | days_until_expiry < 0  | Expired      |
# | days_until_expiry <= 2 | High Alert   |
# | days_until_expiry <= 7 | Medium Alert |
# | days_until_expiry > 7  | Low Alert    |
# ============================================================================

print("\n📦 Generating sample inventory data...")

ingredients = {
    'Coffee Beans':    {'unit': 'kg',     'shelf_life_days': 60,  'cost_per_unit': 850},
    'Milk':            {'unit': 'liter',  'shelf_life_days': 7,   'cost_per_unit': 90},
    'Fresh Cream':     {'unit': 'liter',  'shelf_life_days': 10,  'cost_per_unit': 180},
    'Chicken':         {'unit': 'kg',     'shelf_life_days': 5,   'cost_per_unit': 300},
    'Cheese':          {'unit': 'kg',     'shelf_life_days': 21,  'cost_per_unit': 480},
    'Bread':           {'unit': 'pack',   'shelf_life_days': 4,   'cost_per_unit': 55},
    'Chocolate Syrup': {'unit': 'bottle', 'shelf_life_days': 90,  'cost_per_unit': 220},
    'Caramel Syrup':   {'unit': 'bottle', 'shelf_life_days': 90,  'cost_per_unit': 220},
    'Matcha Powder':   {'unit': 'kg',     'shelf_life_days': 180, 'cost_per_unit': 950},
    'Strawberry':      {'unit': 'kg',     'shelf_life_days': 5,   'cost_per_unit': 280},
    'Oreo':            {'unit': 'pack',   'shelf_life_days': 180, 'cost_per_unit': 150},
    'Potatoes':        {'unit': 'kg',     'shelf_life_days': 14,  'cost_per_unit': 95},
    'Noodles':         {'unit': 'pack',   'shelf_life_days': 365, 'cost_per_unit': 45},
}

inventory_data = []
recent_dates   = pd.date_range(
    start=sales_df['date'].max() - timedelta(days=3*365),
    end=sales_df['date'].max(), freq='W'
)

n_weeks = max(len(recent_dates) - 1, 1)
for i, date in enumerate(recent_dates):
    # gradual cost inflation + growing purchase volume over the 3-year span
    growth = 1 + 0.35 * (i / n_weeks)          # +35% by the final week
    for ingredient, info in ingredients.items():
        if np.random.random() < 0.7:
            qty             = np.random.uniform(3, 12) * growth * np.random.uniform(0.9, 1.1)
            unit_cost       = round(info['cost_per_unit'] * growth * np.random.uniform(0.97, 1.03), 2)
            purchase_date   = date.date()
            expiration_date = purchase_date + timedelta(days=info['shelf_life_days'])
            inventory_data.append({
                'purchase_date':   purchase_date,
                'ingredient':      ingredient,
                'quantity':        round(qty, 1),
                'unit':            info['unit'],
                'cost_per_unit':   unit_cost,
                'total_cost':      round(qty * unit_cost, 2),
                'expiration_date': expiration_date,
                'shelf_life_days': info['shelf_life_days'],
            })

inventory_df = pd.DataFrame(inventory_data)

today = pd.Timestamp.now().normalize()
inventory_df['days_until_expiration'] = (
    pd.to_datetime(inventory_df['expiration_date']) - today
).dt.days

def classify_alert_level(row):
    if row.get('quantity', 1) <= 0:
        return 'Out of Stock'
    days = row['days_until_expiration']
    if days < 0:    return 'Expired'
    elif days <= 2: return 'High Alert'
    elif days <= 7: return 'Medium Alert'
    else:           return 'Low Alert'

inventory_df['alert_level'] = inventory_df.apply(classify_alert_level, axis=1)
print(f"✅ Generated {len(inventory_df)} inventory records (3-year history)")

# ============================================================================
# GENERATE WASTE DATA
#
# Sample waste data:
# | date       | item_name         | category | quantity_wasted | waste_reason  | total_waste_cost |
# |------------|-------------------|----------|-----------------|---------------|------------------|
# | 2026-01-01 | caramel macchiato | Coffee   | 2               | Over-prepared | 240.00           |
# | 2026-01-01 | Overload lomi     | Food     | 1               | Spoiled       | 180.00           |
#
# Waste reason probabilities:
# | Reason        | Probability |
# |---------------|-------------|
# | Over-prepared | 45%         |
# | Expired       | 25%         |
# | Spoiled       | 20%         |
# | Quality issue | 10%         |
#
# 3-year span with a gradual upward trend (+ seasonality + noise) so waste
# volume actually grows over time instead of flatlining — this is what
# the forecast page extrapolates from.
# ============================================================================

print("\n🗑️ Generating sample waste data (3-year trend)...")

waste_data  = []
recent_days = pd.date_range(
    start=sales_df['date'].max() - timedelta(days=3*365),
    end=sales_df['date'].max(), freq='D'
)

n_days = max(len(recent_days) - 1, 1)
for i, date in enumerate(recent_days):
    day_frac = i / n_days                                    # 0 → 1 across the 3 years
    trend_multiplier    = 1 + 0.6 * day_frac                 # up to +60% waste volume by year 3
    seasonal_multiplier = 1 + 0.20 * np.sin(2 * np.pi * date.dayofyear / 365)  # yearly seasonality
    weekday_multiplier  = 1.25 if date.dayofweek >= 5 else 1.0                 # busier weekends
    noise               = np.random.uniform(0.85, 1.15)

    base_num_waste = 3 * trend_multiplier * seasonal_multiplier * weekday_multiplier * noise
    num_waste      = max(1, int(round(base_num_waste)))

    waste_items = menu_df.sample(n=min(num_waste, len(menu_df)))
    for _, item in waste_items.iterrows():
        qty_wasted = max(1, int(round(np.random.randint(1, 4) * trend_multiplier * np.random.uniform(0.85, 1.15))))
        reason = np.random.choice(
            ['Over-prepared','Expired','Spoiled','Quality issue'],
            p=[0.45, 0.25, 0.20, 0.10]
        )
        waste_data.append({
            'date':             date.date(),
            'item_name':        item['item_name'],
            'category':         item['category'],
            'quantity_wasted':  qty_wasted,
            'waste_reason':     reason,
            'cost_per_item':    item['cost'],
            'total_waste_cost': qty_wasted * item['cost'],
        })

waste_df = pd.DataFrame(waste_data)
waste_df['month']      = pd.to_datetime(waste_df['date']).dt.month
waste_df['month_name'] = pd.to_datetime(waste_df['date']).dt.month_name()
print(f"✅ Generated {len(waste_df)} waste records (3-year history, upward trend)")


# SAVE ALL OUTPUT FILES → Data_Cleaning/ folder

print("\n💾 Saving to Data_Cleaning/ folder...")

sales_df.to_csv('../Data_Cleaning/kofe_tala_sales_data.csv',         index=False)
inventory_df.to_csv('../Data_Cleaning/kofe_tala_inventory_data.csv', index=False)
waste_df.to_csv('../Data_Cleaning/kofe_tala_waste_data.csv',         index=False)
menu_df.to_csv('../Data_Cleaning/kofe_tala_menu_data.csv',           index=False)

print("\n" + "=" * 60)
print("✅ CONVERSION COMPLETE!")
print("=" * 60)
print("Files saved to Data_Cleaning/:")
print("  1. kofe_tala_sales_data.csv")
print("  2. kofe_tala_inventory_data.csv")
print("  3. kofe_tala_waste_data.csv")
print("  4. kofe_tala_menu_data.csv")

print(f"\n📊 DATA SUMMARY:")
print(f"  Total Sales Records : {len(sales_df):,}")
print(f"  Total Revenue       : ₱{sales_df['total'].sum():,.2f}")
print(f"  Date Range          : {sales_df['date'].min().date()} → {sales_df['date'].max().date()}")
print(f"  Unique Items        : {len(menu_df)}")

print("\n📊 TOP 10 BEST SELLERS:")
top = sales_df.groupby('item_name')['quantity'].sum().nlargest(10)
for i, (item, qty) in enumerate(top.items(), 1):
    print(f"  {i:2d}. {item:45s} {int(qty):,} sold")

print("\n🚀 Next: cd ../Feature_Engineering && python feature_engineering.py")
