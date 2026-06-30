import pandas as pd, numpy as np, os

PRICE_LOW, PRICE_PCTL, QUALITY_THRESHOLD = 5_000, 0.99, 5
DROP_COLS = ["payment_terms","agent_name","agency_name","listing_url","scraped_date"]
AMENITY_COLS = ["has_pool","has_solar","has_borehole","has_garage",
                "has_garden","has_security","has_inverter","has_generator"]

def get_house_category(ht):
    if pd.isna(ht): return "House"
    ht = str(ht).lower()
    if "townhouse" in ht or "complex" in ht: return "Townhouse"
    if "flat" in ht or "apartment" in ht: return "Flat"
    if "unit" in ht: return "Unit"
    return "House"

def clean_region(r):
    if pd.isna(r): return np.nan
    r = str(r).strip()
    if "," in r: r = r.split(",")[-1].strip()
    return r

def assign_segment(price):
    if price < 120_000: return "Budget"
    if price < 250_000: return "Mid"
    if price < 500_000: return "Premium"
    return "Luxury"

def clean_dataset(df, price_ceiling=None):
    df = df.copy()
    if "listing_url" in df.columns:
        df = df[~df.duplicated(subset=["listing_url"], keep="first")].reset_index(drop=True)
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
    df = df.dropna(subset=["price"]).reset_index(drop=True)
    df.loc[df.get("bathrooms", pd.Series(dtype=float)) > 50, "bathrooms"] = np.nan
    for col in ["floor_area","land_size"]: df[col] = df[col].replace(0, np.nan)
    df.loc[df["floor_area"] < 30, "floor_area"] = np.nan
    PRICE_HIGH = price_ceiling or df["price"].quantile(PRICE_PCTL)
    df = df[df["price"] >= PRICE_LOW].reset_index(drop=True)
    df["price"] = df["price"].clip(upper=PRICE_HIGH)
    df["house_category"] = df["house_type"].apply(get_house_category)
    hm = df["house_category"] == "House"
    df.loc[hm & (df["bedrooms"] > 12), "bedrooms"] = 12
    df.loc[hm & (df["bathrooms"] > 15), "bathrooms"] = 15
    fm = df.groupby(["bedrooms","house_category"])["floor_area"].median()
    df["floor_area"] = df.apply(lambda r: fm.get((r["bedrooms"],r["house_category"]), df["floor_area"].median()) if pd.isna(r["floor_area"]) else r["floor_area"], axis=1)
    bm = df.groupby("bedrooms")["bathrooms"].median()
    df["bathrooms"] = df.apply(lambda r: bm.get(r["bedrooms"], df["bathrooms"].median()) if pd.isna(r["bathrooms"]) else r["bathrooms"], axis=1)
    df["region"] = df["region"].apply(clean_region)
    lm = df.groupby("region")["land_size"].median()
    df["land_size"] = df.apply(lambda r: lm.get(r["region"], df["land_size"].median()) if pd.isna(r["land_size"]) else r["land_size"], axis=1)
    s2r = df.dropna(subset=["suburb","region"]).groupby("suburb")["region"].agg(lambda x: x.mode()[0]).to_dict()
    df.loc[df["region"].isna(), "region"] = df.loc[df["region"].isna(), "suburb"].map(s2r)
    df["region"] = df["region"].fillna("Unknown")
    df["suburb"] = df["suburb"].fillna("Unknown")
    for col in AMENITY_COLS: df[col] = df[col].astype(int)
    df["furnished"] = df["furnished"].astype(int)
    df["amenity_score"]      = df[["has_pool","has_garden","has_garage","has_security"]].sum(axis=1)
    df["energy_score"]       = df[["has_solar","has_borehole","has_inverter","has_generator"]].sum(axis=1)
    df["maintenance_burden"] = df[["has_pool","has_garden","has_generator"]].sum(axis=1)
    df["modern_amenities"]   = df[["has_solar","has_inverter","has_generator"]].sum(axis=1)
    df["amenities_count"]    = df[AMENITY_COLS].sum(axis=1)
    df["segment"] = df["price"].apply(assign_segment)
    core = ["price","suburb","region","bedrooms","bathrooms","floor_area","land_size"]
    df["data_quality_score"] = df[core].notna().sum(axis=1)
    df = df[df["data_quality_score"] >= QUALITY_THRESHOLD].reset_index(drop=True)
    df["segment_encoded"]        = df["segment"].map({"Budget":0,"Mid":1,"Premium":2,"Luxury":3})
    df["region_price_enc"]       = df.groupby("region")["price"].transform("median")
    df["suburb_price_enc"]       = df.groupby("suburb")["price"].transform("median")
    df["house_category_encoded"] = df["house_category"].map({"House":0,"Townhouse":1,"Flat":2,"Unit":3}).fillna(0).astype(int)
    df["log_price"] = np.log1p(df["price"])
    return df

if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "data/raw/property_phase1_engineered.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/processed/properties_clean.csv"
    df = pd.read_csv(src)
    df_clean = clean_dataset(df)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df_clean.to_csv(out, index=False)
    print(f"Saved {out}: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} cols")