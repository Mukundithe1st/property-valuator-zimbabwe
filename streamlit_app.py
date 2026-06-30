"""
streamlit_app.py
Zimbabwe Real Estate Price Prediction System — Phase 8
Interactive Automated Valuation Model (AVM)

Run: python -m streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

st.set_page_config(
    page_title="Zimbabwe Property Valuator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG        = "#0F1117"
SURFACE   = "#161925"
SURFACE_2 = "#1E2230"
BORDER    = "#2A2F42"
TEXT      = "#E8E9ED"
MUTED     = "#A8AFC2"   # Lightened for better visibility on dark background
ACCENT    = "#E94560"
ACCENT_D  = "#C2304A"
TEAL      = "#27AE60"
BLUE      = "#2980B9"
AMBER     = "#F39C12"

SEG_COLORS = {"Budget": TEAL, "Mid": BLUE, "Premium": AMBER, "Luxury": ACCENT}
SEGMENT_RANGES = {"Budget": (0, 120_000), "Mid": (120_000, 250_000),
                   "Premium": (250_000, 500_000), "Luxury": (500_000, 1_800_000)}

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif; }}
    .stApp {{ background-color: {BG}; }}

    .topbar {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 1.1rem 1.6rem; margin: -1rem -1rem 1.5rem -1rem;
        background: {SURFACE}; border-bottom: 1px solid {BORDER};
    }}
    .topbar-brand {{ display:flex; align-items:center; gap:0.7rem; }}
    .topbar-brand .logo {{
        width: 36px; height: 36px; border-radius: 9px;
        background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_D} 100%);
        display:flex; align-items:center; justify-content:center; font-size: 1.1rem;
    }}
    .topbar-brand h1 {{ color: {TEXT}; font-size: 1.15rem; font-weight: 600; margin: 0; letter-spacing: -0.01em; }}
    .topbar-brand p  {{ color: {MUTED}; font-size: 0.78rem; margin: 0; }}
    .topbar-meta {{ color: {MUTED}; font-size: 0.8rem; text-align: right; }}
    .topbar-meta .pill {{
        display:inline-block; background: rgba(39,174,96,0.15); color: {TEAL};
        padding: 0.2rem 0.65rem; border-radius: 20px; font-size: 0.72rem;
        font-weight: 600; margin-left: 0.5rem;
    }}

    .kpi-card {{ 
        background: {SURFACE}; 
        border: 1px solid {BORDER}; 
        border-left: 4px solid {BORDER}; 
        border-radius: 12px; 
        padding: 1.1rem 1.3rem; 
    }}
    .kpi-accent {{ 
        border-left-color: {ACCENT}; 
    }}
    .kpi-accent .kpi-value {{ 
        color: {ACCENT}; 
    }}
    .kpi-label {{ color: {MUTED}; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem; }}
    .kpi-value {{ font-family: 'JetBrains Mono', monospace; color: {TEXT}; font-size: 1.55rem; font-weight: 600; line-height: 1.1; }}
    .kpi-sub {{ color: {MUTED}; font-size: 0.74rem; margin-top: 0.3rem; }}

    .panel {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 1.4rem 1.5rem; margin-bottom: 1rem; }}
    .panel-title {{ color: {TEXT}; font-size: 0.95rem; font-weight: 600; margin: 0 0 0.9rem 0; display:flex; align-items:center; gap:0.5rem; }}
    .panel-title .dot {{ width:7px; height:7px; border-radius:50%; background:{ACCENT}; display:inline-block; }}

    .seg-badge {{ display:inline-flex; align-items:center; gap:0.4rem; padding: 0.35rem 0.9rem; border-radius: 8px; font-weight: 600; font-size: 0.85rem; }}

    .insight {{ background: {SURFACE_2}; border-left: 3px solid {BLUE}; border-radius: 0 8px 8px 0; padding: 0.75rem 1rem; margin-bottom: 0.6rem; font-size: 0.84rem; color: {TEXT}; line-height: 1.5; }}
    .insight.warn {{ border-left-color: {AMBER}; }}
    .insight.good {{ border-left-color: {TEAL}; }}
    .insight b {{ color: {TEXT}; }}

    .profile-item {{ margin-bottom: 0.9rem; }}
    .profile-label {{ color: {MUTED}; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing:0.04em; }}
    .profile-value {{ color: {TEXT}; font-size: 0.92rem; font-weight: 500; margin-top:0.15rem; }}

    section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; background: transparent; }}
    .stTabs [data-baseweb="tab"] {{ background: {SURFACE}; border-radius: 8px 8px 0 0; color: {MUTED}; padding: 0.6rem 1.2rem; font-weight: 500; border: 1px solid {BORDER}; border-bottom: none; }}
    .stTabs [aria-selected="true"] {{ background: {SURFACE_2}; color: {ACCENT} !important; }}
    div[data-testid="stMetricValue"] {{ font-family: 'JetBrains Mono', monospace; }}
    .stButton button, .stDownloadButton button {{ background: {ACCENT}; color: white; border: none; border-radius: 8px; font-weight: 600; }}
    .stButton button:hover, .stDownloadButton button:hover {{ background: {ACCENT_D}; }}
    .footer-note {{ text-align:center; color:{MUTED}; font-size: 0.75rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid {BORDER}; }}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": BORDER, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
    "axes.grid": True, "grid.alpha": 0.15, "grid.color": MUTED,
    "font.family": "DejaVu Sans", "savefig.facecolor": SURFACE,
})

SAFE_FEATURES = [
    "bedrooms", "bathrooms", "floor_area", "land_size", "furnished",
    "has_pool", "has_solar", "has_borehole", "has_garage", "has_garden",
    "has_security", "has_inverter", "has_generator",
    "house_category_encoded",
    "total_amenities", "energy_score", "amenity_score", "modern_amenities",
    "maintenance_burden", "room_density", "floor_to_land_ratio",
    "bed_bath_ratio", "size_ratio",
    "listings_per_region", "listings_per_suburb",
]

REGION_SUBURBS = {
    "Harare North": ["Borrowdale", "Borrowdale Brooke", "Helensvale", "Colne Valley",
                      "Glen Lorne", "Chisipite", "Highlands", "Mandara", "Vainona",
                      "Greystone Park", "Mount Pleasant", "Hogerty Hill", "Pomona",
                      "Quinnington", "Gunhill", "Tynwald North"],
    "Harare East": ["Greendale", "Athlone", "Msasa", "Ruwa", "Mabelreign",
                     "Hatfield", "Marlborough", "Newlands", "Braeside",
                     "Mabvuku", "Tafara", "Glen Norah", "Epworth"],
    "Harare West": ["Westgate", "Kuwadzana", "Aspindale Park", "Dzivarasekwa",
                     "Budiriro", "Glen View", "Warren Park", "Kambuzuma", "Sunridge"],
    "Bulawayo East": ["Burnside", "Hillside", "Matsheumhlope", "Suburbs", "Waterford"],
    "Bulawayo West": ["Pumula", "Nketa", "Cowdray Park", "Tshabalala"],
    "Mutare": ["Mutare", "Murambi", "Dangamvura"],
    "Gweru": ["Gweru", "Senga", "Ascot"],
    "Norton": ["Norton"],
}

REGION_LISTINGS = {
    "Harare North": 655, "Harare East": 412, "Harare West": 298,
    "Bulawayo East": 185, "Bulawayo West": 72, "Mutare": 38,
    "Gweru": 31, "Norton": 28,
}

SUBURB_LISTINGS_MAP = {
    "Borrowdale": 142, "Borrowdale Brooke": 58, "Glen Lorne": 64,
    "Greendale": 42, "Westgate": 38, "Burnside": 22, "Hillside": 19,
    "Mount Pleasant": 41, "Marlborough": 29,
}
SUBURB_LISTINGS_DEFAULT = 18


@st.cache_resource
def load_model():
    for p in ["models/xgb_tuned.pkl", "models/best_model.pkl",
              "../models/xgb_tuned.pkl", "../models/best_model.pkl"]:
        if os.path.exists(p):
            return joblib.load(p), p
    return None, None


@st.cache_data
def load_metadata():
    for p in ["models/model_metadata.json", "../models/model_metadata.json"]:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return {}


@st.cache_data
def load_training_data():
    for p in ["data/processed/properties_features.csv",
              "data/processed/properties_clean.csv",
              "../data/processed/properties_features.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


@st.cache_data
def suburb_listing_counts(_df):
    if _df is None or "suburb" not in _df.columns:
        return {}
    return _df["suburb"].value_counts().to_dict()


def build_feature_row(bedrooms, bathrooms, floor_area, land_size, furnished,
                       has_pool, has_solar, has_borehole, has_garage, has_garden,
                       has_security, has_inverter, has_generator,
                       house_category, region, suburb, suburb_counts):
    energy_score       = int(has_solar) + int(has_borehole) + int(has_inverter) + int(has_generator)
    amenity_score      = int(has_pool) + int(has_garden) + int(has_garage) + int(has_security)
    modern_amenities   = int(has_solar) + int(has_inverter) + int(has_generator)
    maintenance_burden = int(has_pool) + int(has_garden) + int(has_generator)
    total_amenities    = sum([has_pool, has_solar, has_borehole, has_garage,
                               has_garden, has_security, has_inverter, has_generator])

    room_density   = bedrooms / floor_area if floor_area > 0 else 0
    floor_to_land  = floor_area / land_size if land_size > 0 else 0
    size_ratio     = floor_area / land_size if land_size > 0 else 0
    bed_bath_ratio = bedrooms / bathrooms if bathrooms > 0 else bedrooms
    house_cat_enc  = {"House": 0, "Townhouse": 1, "Flat": 2, "Unit": 3}.get(house_category, 0)

    listings_region = REGION_LISTINGS.get(region, 50)
    listings_suburb = suburb_counts.get(suburb, SUBURB_LISTINGS_MAP.get(suburb, SUBURB_LISTINGS_DEFAULT))

    row = {
        "bedrooms": bedrooms, "bathrooms": bathrooms, "floor_area": floor_area,
        "land_size": land_size, "furnished": int(furnished),
        "has_pool": int(has_pool), "has_solar": int(has_solar),
        "has_borehole": int(has_borehole), "has_garage": int(has_garage),
        "has_garden": int(has_garden), "has_security": int(has_security),
        "has_inverter": int(has_inverter), "has_generator": int(has_generator),
        "house_category_encoded": house_cat_enc,
        "total_amenities": total_amenities, "energy_score": energy_score,
        "amenity_score": amenity_score, "modern_amenities": modern_amenities,
        "maintenance_burden": maintenance_burden, "room_density": room_density,
        "floor_to_land_ratio": floor_to_land, "bed_bath_ratio": bed_bath_ratio,
        "size_ratio": size_ratio, "listings_per_region": listings_region,
        "listings_per_suburb": listings_suburb,
    }
    return pd.DataFrame([row])[SAFE_FEATURES]


def predict_price(model, feature_row):
    log_pred = model.predict(feature_row)[0]
    price = np.expm1(log_pred)
    if price < 120_000:
        log_std = 0.35
    elif price > 500_000:
        log_std = 0.25
    else:
        log_std = 0.18
    price_low = np.expm1(log_pred - log_std)
    price_hi  = np.expm1(log_pred + log_std)
    return float(price), float(price_low), float(price_hi)


def get_segment(price):
    if price < 120_000: return "Budget"
    if price < 250_000: return "Mid"
    if price < 500_000: return "Premium"
    return "Luxury"


def compute_local_impact(model, feature_row, df_train):
    base_log = model.predict(feature_row)[0]
    base_price = np.expm1(base_log)

    if df_train is not None:
        medians = {f: df_train[f].median() for f in SAFE_FEATURES if f in df_train.columns}
    else:
        medians = {}

    groups = {
        "Floor area":         ["floor_area", "room_density", "floor_to_land_ratio", "size_ratio"],
        "Land size":          ["land_size"],
        "Bed / bath config":  ["bedrooms", "bathrooms", "bed_bath_ratio"],
        "Off-grid utilities": ["has_solar", "has_borehole", "has_inverter", "has_generator", "energy_score", "modern_amenities"],
        "Lifestyle amenities":["has_pool", "has_garden", "has_garage", "has_security", "amenity_score", "maintenance_burden"],
        "Location liquidity": ["listings_per_region", "listings_per_suburb"],
        "Property type":      ["house_category_encoded", "furnished"],
    }

    impacts = {}
    for label, cols in groups.items():
        cols_present = [c for c in cols if c in feature_row.columns]
        if not cols_present:
            continue
        modified = feature_row.copy()
        moved_any = False
        for c in cols_present:
            if c in medians and not pd.isna(medians[c]):
                modified[c] = medians[c]
                moved_any = True
        if not moved_any:
            continue
        new_price = np.expm1(model.predict(modified)[0])
        impacts[label] = float(base_price) - float(new_price)

    return float(base_price), impacts


def plot_value_gauge(price, segment):
    lo, hi = SEGMENT_RANGES[segment]
    color = SEG_COLORS[segment]
    fig, ax = plt.subplots(figsize=(8, 1.3))
    ax.barh([0], [hi - lo], left=lo, height=0.5, color=SURFACE_2, edgecolor=BORDER, linewidth=1)
    ax.barh([0], [min(price, hi) - lo], left=lo, height=0.5, color=color, alpha=0.85)
    ax.scatter([price], [0], s=140, color=color, edgecolor=TEXT, linewidth=1.5, zorder=5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    return fig


def plot_global_importance(model):
    if not hasattr(model, "feature_importances_"):
        return None
    df_imp = pd.DataFrame({"Feature": SAFE_FEATURES, "Gain": model.feature_importances_})
    df_imp = df_imp.sort_values("Gain", ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = [ACCENT if v > df_imp["Gain"].median() else BLUE for v in df_imp["Gain"]]
    ax.barh(df_imp["Feature"], df_imp["Gain"], color=colors, edgecolor="none", alpha=0.9)
    ax.set_xlabel("Relative importance (gain)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)
    plt.tight_layout()
    return fig


def plot_local_impact(impacts):
    df_local = pd.DataFrame(list(impacts.items()), columns=["Component", "Impact"])
    df_local = df_local.sort_values("Impact")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    colors = [TEAL if v >= 0 else ACCENT for v in df_local["Impact"]]
    ax.barh(df_local["Component"], df_local["Impact"], color=colors, alpha=0.9, height=0.55)
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Effect on predicted price vs. typical listing (USD)", fontsize=8.5)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:+.0f}K"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)
    plt.tight_layout()
    return fig


def plot_whatif_curve(model, base_row, vary_param, vary_range, current_val):
    rows = []
    for v in vary_range:
        r = base_row.copy()
        if vary_param == "floor_area":
            r["floor_area"] = v
            r["room_density"] = r["bedrooms"] / v if v > 0 else 0
            r["floor_to_land_ratio"] = v / r["land_size"] if r["land_size"] > 0 else 0
            r["size_ratio"] = v / r["land_size"] if r["land_size"] > 0 else 0
        elif vary_param == "bedrooms":
            r["bedrooms"] = v
            r["room_density"] = v / r["floor_area"] if r["floor_area"] > 0 else 0
            r["bed_bath_ratio"] = v / r["bathrooms"] if r["bathrooms"] > 0 else v
        elif vary_param == "land_size":
            r["land_size"] = v
            r["floor_to_land_ratio"] = r["floor_area"] / v if v > 0 else 0
            r["size_ratio"] = r["floor_area"] / v if v > 0 else 0
        rows.append(r)

    df_vary = pd.DataFrame(rows)[SAFE_FEATURES]
    preds = np.expm1(model.predict(df_vary))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(vary_range, preds / 1e3, color=ACCENT, linewidth=2.5, zorder=3)
    ax.fill_between(vary_range, preds / 1e3, alpha=0.12, color=ACCENT)
    ax.scatter([current_val], [np.interp(current_val, vary_range, preds) / 1e3],
               s=120, color=ACCENT, edgecolor=TEXT, linewidth=1.5, zorder=5, label="Current")
    ax.set_xlabel(vary_param.replace("_", " ").title(), fontsize=9)
    ax.set_ylabel("Predicted price (USD K)", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}K"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, labelcolor=TEXT)
    ax.tick_params(labelsize=8.5)
    plt.tight_layout()
    return fig


def find_similar(df_train, bedrooms, bathrooms, floor_area, price_pred, n=5):
    if df_train is None:
        return None
    mask = (
        (df_train["bedrooms"].between(bedrooms - 1, bedrooms + 1)) &
        (df_train["bathrooms"].between(bathrooms - 1, bathrooms + 1)) &
        (df_train["floor_area"].between(floor_area * 0.7, floor_area * 1.3))
    )
    similar = df_train[mask].copy()
    if len(similar) == 0:
        return None
    similar["price_diff"] = (similar["price"] - price_pred).abs()
    cols = [c for c in ["price", "bedrooms", "bathrooms", "floor_area", "land_size", "segment"]
            if c in similar.columns]
    return similar.nsmallest(n, "price_diff")[cols].reset_index(drop=True)


def location_input(key_prefix, default_region="Harare North"):
    region = st.selectbox("Region", list(REGION_SUBURBS.keys()),
                           index=list(REGION_SUBURBS.keys()).index(default_region),
                           key=f"{key_prefix}_region")
    suburb_options = REGION_SUBURBS.get(region, []) + ["Other (type below)"]
    suburb_choice = st.selectbox("Suburb", suburb_options, key=f"{key_prefix}_suburb_select")
    if suburb_choice == "Other (type below)":
        suburb = st.text_input("Type suburb name", placeholder="e.g. Avondale", key=f"{key_prefix}_suburb_text")
        suburb = suburb.strip() or "Unknown"
    else:
        suburb = suburb_choice
    return region, suburb


def property_inputs(key_prefix, label):
    st.markdown(f'<div class="panel-title"><span class="dot"></span>{label}</div>', unsafe_allow_html=True)
    region, suburb = location_input(key_prefix)

    c1, c2 = st.columns(2)
    with c1:
        house_category = st.selectbox("Property type", ["House", "Townhouse", "Flat", "Unit"], key=f"{key_prefix}_type")
        bedrooms = st.number_input("Bedrooms", 1, 12, 3, key=f"{key_prefix}_beds")
    with c2:
        furnished = st.checkbox("Furnished", value=False, key=f"{key_prefix}_furn")
        bathrooms = st.number_input("Bathrooms", 1, 10, 2, key=f"{key_prefix}_baths")

    c3, c4 = st.columns(2)
    with c3:
        floor_area = st.number_input("Floor area (m²)", 30, 5000, 200, step=10, key=f"{key_prefix}_floor")
    with c4:
        land_size = st.number_input("Land size (m²)", 100, 50000, 1000, step=100, key=f"{key_prefix}_land")

    st.markdown("**Off-grid utilities**")
    c5, c6, c7, c8 = st.columns(4)
    with c5: has_borehole = st.checkbox("Borehole", key=f"{key_prefix}_bh")
    with c6: has_solar = st.checkbox("Solar", key=f"{key_prefix}_solar")
    with c7: has_inverter = st.checkbox("Inverter", key=f"{key_prefix}_inv")
    with c8: has_generator = st.checkbox("Generator", key=f"{key_prefix}_gen")

    st.markdown("**Lifestyle & security**")
    c9, c10, c11, c12 = st.columns(4)
    with c9: has_pool = st.checkbox("Pool", key=f"{key_prefix}_pool")
    with c10: has_garden = st.checkbox("Garden", value=True, key=f"{key_prefix}_garden")
    with c11: has_garage = st.checkbox("Garage", key=f"{key_prefix}_garage")
    with c12: has_security = st.checkbox("Security", key=f"{key_prefix}_sec")

    return dict(
        bedrooms=bedrooms, bathrooms=bathrooms, floor_area=floor_area, land_size=land_size,
        furnished=furnished, has_pool=has_pool, has_solar=has_solar, has_borehole=has_borehole,
        has_garage=has_garage, has_garden=has_garden, has_security=has_security,
        has_inverter=has_inverter, has_generator=has_generator,
        house_category=house_category, region=region, suburb=suburb,
    )


model, model_path = load_model()
metadata  = load_metadata()
df_train  = load_training_data()
suburb_counts = suburb_listing_counts(df_train)

r2_display = metadata.get("test_r2", metadata.get("metrics", {}).get("r2", 0.81))
st.markdown(f"""
<div class="topbar">
    <div class="topbar-brand">
        <div class="logo">🏠</div>
        <div><h1>Property Valuator</h1><p>Zimbabwe Real Estate · AI Valuation Engine</p></div>
    </div>
    <div class="topbar-meta">XGBoost · R² {r2_display:.2f}<span class="pill">● Model Live</span></div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("""
    **Model not found.** Run the Phase 4 and Phase 5 notebooks first to generate
    `models/best_model.pkl` and `models/xgb_tuned.pkl`.
    """)
    st.stop()

mae_display = metadata.get("test_mae_usd", metadata.get("metrics", {}).get("mae_usd", 91650))
n_train = len(df_train) if df_train is not None else 1985

k1, k2, k3, k4 = st.columns(4)
for col, label, value, sub, accent in [
    (k1, "Model accuracy", f"R² {r2_display:.3f}", "On held-out test set", True),
    (k2, "Avg. error",     f"${mae_display:,.0f}", "Mean absolute error", False),
    (k3, "Training data",  f"{n_train:,}", "Property listings", False),
    (k4, "Features used",  f"{len(SAFE_FEATURES)}", "Leakage-free inputs", False),
]:
    with col:
        cls = "kpi-card kpi-accent" if accent else "kpi-card"
        st.markdown(f'<div class="{cls}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["💰 Valuation", "⚖️ Compare Properties", "📈 What-If Analysis", "🛡️ Engine Diagnostics"])

with tab1:
    left, right = st.columns([1, 1.3])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        inputs = property_inputs("v1", "Property Details")
        st.markdown('</div>', unsafe_allow_html=True)

    feat_row = build_feature_row(
        inputs["bedrooms"], inputs["bathrooms"], inputs["floor_area"], inputs["land_size"],
        inputs["furnished"], inputs["has_pool"], inputs["has_solar"], inputs["has_borehole"],
        inputs["has_garage"], inputs["has_garden"], inputs["has_security"],
        inputs["has_inverter"], inputs["has_generator"], inputs["house_category"],
        inputs["region"], inputs["suburb"], suburb_counts,
    )
    price, price_low, price_hi = predict_price(model, feat_row)
    segment = get_segment(price)
    seg_color = SEG_COLORS[segment]

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><span class="dot"></span>Estimated Market Value</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex; align-items:baseline; gap:0.8rem; margin-bottom:0.3rem;">
            <span style="font-family:'JetBrains Mono',monospace; font-size:2.4rem; font-weight:700; color:{TEXT}">${price:,.0f}</span>
            <span class="seg-badge" style="background:{seg_color}22; color:{seg_color}">● {segment}</span>
        </div>
        <div style="color:{MUTED}; font-size:0.85rem; margin-bottom:1rem;">
            Confidence range: ${price_low:,.0f} — ${price_hi:,.0f}
        </div>
        """, unsafe_allow_html=True)

        fig_gauge = plot_value_gauge(price, segment)
        st.pyplot(fig_gauge, use_container_width=True)
        plt.close(fig_gauge)

        m1, m2, m3 = st.columns(3)
        m1.metric("Price / m²", f"${price / inputs['floor_area']:,.0f}")
        m2.metric("Price / Bedroom", f"${price / inputs['bedrooms']:,.0f}")
        energy = int(inputs["has_solar"]) + int(inputs["has_borehole"]) + int(inputs["has_inverter"]) + int(inputs["has_generator"])
        m3.metric("Energy Score", f"{energy}/4")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><span class="dot"></span>Insights</div>', unsafe_allow_html=True)
        seg_desc = {
            "Budget":  "Entry-level market. Price sensitivity is highest in this segment.",
            "Mid":     "Largest buyer pool in Zimbabwe's residential market.",
            "Premium": "Buyers typically expect specific amenities and prime locations.",
            "Luxury":  "Limited buyer pool. Off-grid features are critical value drivers.",
        }
        st.markdown(f'<div class="insight"><b>{segment} segment</b> — {seg_desc[segment]}</div>', unsafe_allow_html=True)
        if energy >= 3:
            st.markdown('<div class="insight good"><b>Strong off-grid profile</b> — energy independence is a major value driver given load-shedding and water supply challenges.</div>', unsafe_allow_html=True)
        elif energy == 0:
            st.markdown('<div class="insight warn"><b>No off-grid features</b> — adding a borehole or solar could meaningfully increase value and buyer interest.</div>', unsafe_allow_html=True)
        size_ratio = inputs["floor_area"] / inputs["land_size"] if inputs["land_size"] > 0 else 0
        if size_ratio < 0.1:
            st.markdown('<div class="insight"><b>Large stand</b> — low floor-to-land ratio suggests development potential or added privacy.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight warn"><b>Accuracy note</b> — R² ≈ {r2_display:.2f} overall; Luxury and Budget tails carry wider error margins. Use alongside a professional valuation.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c_left, c_right = st.columns([1, 1.3])
    with c_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><span class="dot"></span>What\'s Driving This Price</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color:{MUTED}; font-size:0.78rem; margin-bottom:0.6rem;">Each feature group is reset to the training-set median to isolate its effect on this property\'s prediction.</div>', unsafe_allow_html=True)
        base_price, impacts = compute_local_impact(model, feat_row, df_train)
        if impacts:
            fig_local = plot_local_impact(impacts)
            st.pyplot(fig_local, use_container_width=True)
            plt.close(fig_local)
        else:
            st.info("Load training data (data/processed/properties_features.csv) to see per-property impact.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><span class="dot"></span>Similar Properties in Training Data</div>', unsafe_allow_html=True)
        if df_train is not None:
            similar = find_similar(df_train, inputs["bedrooms"], inputs["bathrooms"], inputs["floor_area"], price)
            if similar is not None and len(similar) > 0:
                disp = similar.copy()
                if "price" in disp.columns: disp["price"] = disp["price"].apply(lambda x: f"${x:,.0f}")
                if "floor_area" in disp.columns: disp["floor_area"] = disp["floor_area"].apply(lambda x: f"{x:.0f}m²")
                if "land_size" in disp.columns: disp["land_size"] = disp["land_size"].apply(lambda x: f"{x:,.0f}m²")
                st.dataframe(disp, use_container_width=True, hide_index=True)
            else:
                st.info("No close matches found for this configuration.")
        else:
            st.info("Training data not loaded — place properties_features.csv in data/processed/")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><span class="dot"></span>Property Profile</div>', unsafe_allow_html=True)
    pcols = st.columns(4)
    off_grid = ", ".join([x for x, v in [("Borehole", inputs["has_borehole"]), ("Solar", inputs["has_solar"]),
                                          ("Inverter", inputs["has_inverter"]), ("Generator", inputs["has_generator"])] if v]) or "None"
    leisure = ", ".join([x for x, v in [("Pool", inputs["has_pool"]), ("Garden", inputs["has_garden"]),
                                         ("Garage", inputs["has_garage"]), ("Security", inputs["has_security"])] if v]) or "None"
    profile_items = [
        ("Location", f"{inputs['suburb']}, {inputs['region']}"),
        ("Type", f"{inputs['bedrooms']}BR {inputs['house_category']}"),
        ("Size", f"{inputs['floor_area']:,}m² / {inputs['land_size']:,}m² stand"),
        ("Furnished", "Yes" if inputs["furnished"] else "No"),
        ("Off-grid", off_grid),
        ("Lifestyle", leisure),
    ]
    for i, (lbl, val) in enumerate(profile_items):
        with pcols[i % 4]:
            st.markdown(f'<div class="profile-item"><div class="profile-label">{lbl}</div><div class="profile-value">{val}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Fixed JSON serialization: convert numpy floats to native Python floats ---
    report_data = {
        "predicted_price": float(round(price, 2)),
        "confidence_low": float(round(price_low, 2)),
        "confidence_high": float(round(price_hi, 2)),
        "segment": segment,
        "location": {"region": inputs["region"], "suburb": inputs["suburb"]},
        "attributes": {
            "bedrooms": int(inputs["bedrooms"]),
            "bathrooms": int(inputs["bathrooms"]),
            "floor_area_m2": int(inputs["floor_area"]),
            "land_size_m2": int(inputs["land_size"]),
            "property_type": inputs["house_category"]
        },
    }
    st.download_button(
        "💾 Download valuation report (.json)",
        data=json.dumps(report_data, indent=2),
        file_name=f"valuation_{inputs['suburb'].replace(' ', '_')}.json",
        mime="application/json",
    )


with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        inputs_a = property_inputs("cmp_a", "Property A")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        inputs_b = property_inputs("cmp_b", "Property B")
        st.markdown('</div>', unsafe_allow_html=True)

    feat_a = build_feature_row(
        inputs_a["bedrooms"], inputs_a["bathrooms"], inputs_a["floor_area"], inputs_a["land_size"],
        inputs_a["furnished"], inputs_a["has_pool"], inputs_a["has_solar"], inputs_a["has_borehole"],
        inputs_a["has_garage"], inputs_a["has_garden"], inputs_a["has_security"],
        inputs_a["has_inverter"], inputs_a["has_generator"], inputs_a["house_category"],
        inputs_a["region"], inputs_a["suburb"], suburb_counts,
    )
    feat_b = build_feature_row(
        inputs_b["bedrooms"], inputs_b["bathrooms"], inputs_b["floor_area"], inputs_b["land_size"],
        inputs_b["furnished"], inputs_b["has_pool"], inputs_b["has_solar"], inputs_b["has_borehole"],
        inputs_b["has_garage"], inputs_b["has_garden"], inputs_b["has_security"],
        inputs_b["has_inverter"], inputs_b["has_generator"], inputs_b["house_category"],
        inputs_b["region"], inputs_b["suburb"], suburb_counts,
    )
    price_a, _, _ = predict_price(model, feat_a)
    price_b, _, _ = predict_price(model, feat_b)
    seg_a, seg_b = get_segment(price_a), get_segment(price_b)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><span class="dot"></span>Comparison Result</div>', unsafe_allow_html=True)
    rc1, rc2, rc3 = st.columns([1, 0.3, 1])
    with rc1:
        st.markdown(f"""
        <div style="text-align:center; padding:1rem; background:{SURFACE_2}; border-radius:10px;">
            <div style="color:{MUTED}; font-size:0.78rem; font-weight:600; text-transform:uppercase;">Property A</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1.8rem; font-weight:700; color:{TEXT}; margin:0.3rem 0;">${price_a:,.0f}</div>
            <span class="seg-badge" style="background:{SEG_COLORS[seg_a]}22; color:{SEG_COLORS[seg_a]}">● {seg_a}</span>
        </div>""", unsafe_allow_html=True)
    with rc2:
        diff = price_a - price_b
        diff_pct = (diff / price_b * 100) if price_b > 0 else 0
        arrow = "↑" if diff > 0 else "↓"
        color = TEAL if diff > 0 else ACCENT
        st.markdown(f"""
        <div style="text-align:center; padding-top:1.8rem;">
            <div style="font-size:1.5rem; color:{color}; font-weight:700;">{arrow}</div>
            <div style="font-family:'JetBrains Mono',monospace; color:{color}; font-weight:600; font-size:0.85rem;">{abs(diff_pct):.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with rc3:
        st.markdown(f"""
        <div style="text-align:center; padding:1rem; background:{SURFACE_2}; border-radius:10px;">
            <div style="color:{MUTED}; font-size:0.78rem; font-weight:600; text-transform:uppercase;">Property B</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1.8rem; font-weight:700; color:{TEXT}; margin:0.3rem 0;">${price_b:,.0f}</div>
            <span class="seg-badge" style="background:{SEG_COLORS[seg_b]}22; color:{SEG_COLORS[seg_b]}">● {seg_b}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    compare_df = pd.DataFrame({
        "Attribute": ["Location", "Type", "Bedrooms", "Bathrooms", "Floor area", "Land size",
                      "Energy score", "Amenity score", "Predicted price"],
        "Property A": [
            f"{inputs_a['suburb']}, {inputs_a['region']}", inputs_a["house_category"],
            inputs_a["bedrooms"], inputs_a["bathrooms"], f"{inputs_a['floor_area']}m²", f"{inputs_a['land_size']}m²",
            int(inputs_a["has_solar"])+int(inputs_a["has_borehole"])+int(inputs_a["has_inverter"])+int(inputs_a["has_generator"]),
            int(inputs_a["has_pool"])+int(inputs_a["has_garden"])+int(inputs_a["has_garage"])+int(inputs_a["has_security"]),
            f"${price_a:,.0f}",
        ],
        "Property B": [
            f"{inputs_b['suburb']}, {inputs_b['region']}", inputs_b["house_category"],
            inputs_b["bedrooms"], inputs_b["bathrooms"], f"{inputs_b['floor_area']}m²", f"{inputs_b['land_size']}m²",
            int(inputs_b["has_solar"])+int(inputs_b["has_borehole"])+int(inputs_b["has_inverter"])+int(inputs_b["has_generator"]),
            int(inputs_b["has_pool"])+int(inputs_b["has_garden"])+int(inputs_b["has_garage"])+int(inputs_b["has_security"]),
            f"${price_b:,.0f}",
        ],
    })
    st.dataframe(compare_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


with tab3:
    left_wi, right_wi = st.columns([1, 1.3])
    with left_wi:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        wi_inputs = property_inputs("wi", "Base Property")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><span class="dot"></span>Vary a feature</div>', unsafe_allow_html=True)
        vary_param = st.selectbox("Feature to explore", ["floor_area", "bedrooms", "land_size"],
                                   format_func=lambda x: x.replace("_", " ").title())
        st.markdown('</div>', unsafe_allow_html=True)

    wi_feat = build_feature_row(
        wi_inputs["bedrooms"], wi_inputs["bathrooms"], wi_inputs["floor_area"], wi_inputs["land_size"],
        wi_inputs["furnished"], wi_inputs["has_pool"], wi_inputs["has_solar"], wi_inputs["has_borehole"],
        wi_inputs["has_garage"], wi_inputs["has_garden"], wi_inputs["has_security"],
        wi_inputs["has_inverter"], wi_inputs["has_generator"], wi_inputs["house_category"],
        wi_inputs["region"], wi_inputs["suburb"], suburb_counts,
    )
    wi_price, _, _ = predict_price(model, wi_feat)

    if vary_param == "floor_area":
        rng = np.linspace(50, 800, 40); current = wi_inputs["floor_area"]
    elif vary_param == "bedrooms":
        rng = np.arange(1, 11); current = wi_inputs["bedrooms"]
    else:
        rng = np.linspace(200, 10000, 40); current = wi_inputs["land_size"]

    with right_wi:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(f'<div class="panel-title"><span class="dot"></span>Price sensitivity — {vary_param.replace("_"," ").title()}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color:{MUTED}; font-size:0.85rem; margin-bottom:0.8rem;">Current estimate: <span style="color:{TEXT}; font-weight:600">${wi_price:,.0f}</span> at {vary_param.replace("_"," ")} = {current}</div>', unsafe_allow_html=True)
        fig_wi = plot_whatif_curve(model, wi_feat.iloc[0], vary_param, rng, current)
        st.pyplot(fig_wi, use_container_width=True)
        plt.close(fig_wi)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><span class="dot"></span>Scenario snapshots</div>', unsafe_allow_html=True)
        snap_points = np.percentile(rng, [10, 30, 50, 70, 90])
        if vary_param != "land_size":
            snap_points = snap_points.astype(int)
        snap_rows = []
        for v in snap_points:
            r = wi_feat.iloc[0].copy()
            if vary_param == "floor_area":
                r["floor_area"] = v; r["room_density"] = r["bedrooms"]/v if v > 0 else 0
            elif vary_param == "bedrooms":
                r["bedrooms"] = v; r["room_density"] = v/r["floor_area"] if r["floor_area"] > 0 else 0
            elif vary_param == "land_size":
                r["land_size"] = v; r["floor_to_land_ratio"] = r["floor_area"]/v if v > 0 else 0
            pred = np.expm1(model.predict(pd.DataFrame([r])[SAFE_FEATURES])[0])
            snap_rows.append({vary_param.replace("_"," ").title(): f"{v:,.0f}", "Predicted price": f"${pred:,.0f}"})
        st.dataframe(pd.DataFrame(snap_rows), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)


with tab4:
    diag_left, diag_right = st.columns([1, 1])
    with diag_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><span class="dot"></span>Global Feature Importance</div>', unsafe_allow_html=True)
        fig_global = plot_global_importance(model)
        if fig_global:
            st.pyplot(fig_global, use_container_width=True)
            plt.close(fig_global)
        st.markdown('</div>', unsafe_allow_html=True)

    with diag_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><span class="dot"></span>Model & Validation Details</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="insight"><b>Estimator</b> — XGBoost Regressor</div>
        <div class="insight"><b>Target transform</b> — log(1 + price), reported back in USD</div>
        <div class="insight"><b>Hyperparameter search</b> — Optuna, 100 trials, 5-fold CV</div>
        <div class="insight good"><b>Leakage controls</b> — price-derived columns excluded
        (price_per_sqm_floor, price_per_bed, price_per_bath, price_per_hectare,
        value_density, bed_quality, segment_encoded, region_price_enc, suburb_price_enc)</div>
        <div class="insight warn"><b>Disclaimer</b> — algorithmic estimate only. Not a substitute
        for a licensed on-site valuation.</div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


with st.expander("ℹ️ Model & data information"):
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.markdown("**Model**")
        st.write("Type: XGBoost Regressor")
        st.write("Target: log(1+price)")
        st.write(f"Features: {len(SAFE_FEATURES)} (leakage-free)")
    with fc2:
        st.markdown("**Training data**")
        st.write(f"Records: ~{n_train:,} listings")
        st.write("Source: property.co.zw")
        st.write("Scraped: April 2026")
    with fc3:
        st.markdown("**Performance**")
        st.write(f"R²: {r2_display:.3f} (test set)")
        st.write(f"MAE: ${mae_display:,.0f}")
        st.write("Tuned with: Optuna (100 trials)")

st.markdown('<div class="footer-note">Zimbabwe Real Estate Price Prediction System · Built with Streamlit &amp; XGBoost</div>', unsafe_allow_html=True)