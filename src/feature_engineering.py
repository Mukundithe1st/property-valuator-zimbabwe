import pandas as pd
import numpy as np

SAFE_FEATURES = [
    'bedrooms', 'bathrooms', 'floor_area', 'land_size', 'furnished',
    'has_pool', 'has_solar', 'has_borehole', 'has_garage', 'has_garden',
    'has_security', 'has_inverter', 'has_generator',
    'house_category_encoded',
    'total_amenities', 'energy_score', 'amenity_score', 'modern_amenities',
    'maintenance_burden', 'room_density', 'floor_to_land_ratio',
    'bed_bath_ratio', 'size_ratio',
    'listings_per_region', 'listings_per_suburb'
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    AMENITY_COLS = ['has_pool','has_solar','has_borehole','has_garage',
                    'has_garden','has_security','has_inverter','has_generator']

    df['total_amenities']     = df[AMENITY_COLS].sum(axis=1)
    df['room_density']        = (df['bedrooms'] / df['floor_area']).replace([np.inf,-np.inf], np.nan)
    df['floor_to_land_ratio'] = (df['floor_area'] / df['land_size']).replace([np.inf,-np.inf], np.nan)
    df['bed_bath_ratio']      = (df['bedrooms'] / df['bathrooms'].replace(0, np.nan)).replace([np.inf,-np.inf], np.nan)
    df['listings_per_region'] = df.groupby('region')['price'].transform('count')
    df['listings_per_suburb'] = df.groupby('suburb')['price'].transform('count')
    df['renewable_energy_index'] = df[['has_solar','has_borehole','has_inverter','has_generator']].sum(axis=1) / 4

    if 'house_category_encoded' not in df.columns:
        df['house_category_encoded'] = df.get('house_category', pd.Series('House', index=df.index)).map(
            {'House':0,'Townhouse':1,'Flat':2,'Unit':3}).fillna(0).astype(int)

    for col in SAFE_FEATURES:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    return df

if __name__ == '__main__':
    import sys, os
    src = sys.argv[1] if len(sys.argv) > 1 else 'data/processed/properties_clean.csv'
    out = sys.argv[2] if len(sys.argv) > 2 else 'data/processed/properties_features.csv'
    df = pd.read_csv(src)
    df = engineer_features(df)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f'Saved {out}: {df.shape}')