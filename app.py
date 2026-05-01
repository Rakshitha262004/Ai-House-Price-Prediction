# ============================================
# HOUSE PRICE PREDICTION — STREAMLIT DASHBOARD
# Premium Dark Gold Theme
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# --------------------------------------------
# PAGE CONFIG
# --------------------------------------------
st.set_page_config(
    page_title="EstateIQ — House Price Predictor",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------
# CUSTOM CSS — DARK GOLD LUXURY THEME
# --------------------------------------------
st.markdown("""
<style>
/* ---- Google Font Import ---- */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ---- Global Reset ---- */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0C0C0E;
    color: #E8DCC8;
}

/* ---- Main Content Area ---- */
.main .block-container {
    padding: 2rem 3rem;
    max-width: 1400px;
}

/* ---- Hide Streamlit Branding ---- */
#MainMenu, footer, header { visibility: hidden; }

/* ---- Hero Title ---- */
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    background: linear-gradient(135deg, #C9A84C 0%, #F5E6C8 50%, #C9A84C 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #7A6E5F;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.hero-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #C9A84C55, transparent);
    margin: 1.5rem 0 2.5rem 0;
}

/* ---- Metric Cards ---- */
.metric-card {
    background: linear-gradient(135deg, #16141A 0%, #1C1A22 100%);
    border: 1px solid #2A2520;
    border-left: 3px solid #C9A84C;
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}

.metric-label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #7A6E5F;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.8rem;
    font-weight: 600;
    color: #C9A84C;
}

/* ---- Prediction Banner ---- */
.prediction-banner {
    background: linear-gradient(135deg, #16141A, #1C1820);
    border: 1px solid #C9A84C44;
    border-radius: 6px;
    padding: 2rem 2.5rem;
    text-align: center;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
}

.prediction-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #C9A84C, transparent);
}

.prediction-label {
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #7A6E5F;
    margin-bottom: 0.5rem;
}

.prediction-price {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #C9A84C;
    letter-spacing: 0.03em;
}

/* ---- Section Headers ---- */
.section-header {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: #E8DCC8;
    letter-spacing: 0.05em;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #2A2520;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F0D14 0%, #0C0C0E 100%);
    border-right: 1px solid #1E1C24;
}

[data-testid="stSidebar"] .sidebar-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.3rem;
    color: #C9A84C;
    letter-spacing: 0.08em;
    padding: 1rem 0 0.5rem 0;
}

/* ---- Slider Styling ---- */
[data-testid="stSlider"] > div > div > div > div {
    background: #C9A84C !important;
}

/* ---- Button ---- */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #C9A84C, #A8893C);
    color: #0C0C0E;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border: none;
    border-radius: 3px;
    padding: 0.7rem 1.5rem;
    cursor: pointer;
    transition: all 0.3s ease;
    margin-top: 1rem;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #E0C06A, #C9A84C);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px #C9A84C33;
}

/* ---- Dataframe ---- */
[data-testid="stDataFrame"] {
    border: 1px solid #1E1C24;
    border-radius: 4px;
}

/* ---- Info / Warning Boxes ---- */
.info-box {
    background: #14121A;
    border: 1px solid #1E1C24;
    border-left: 3px solid #4A90E2;
    border-radius: 3px;
    padding: 0.8rem 1.2rem;
    font-size: 0.85rem;
    color: #9A8E7F;
    margin: 0.5rem 0 1rem 0;
}

/* ---- Tab Styling ---- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #2A2520;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7A6E5F;
    padding: 0.6rem 1.5rem;
}

[data-testid="stTabs"] [aria-selected="true"] {
    color: #C9A84C !important;
    border-bottom: 2px solid #C9A84C !important;
}

/* ---- Selectbox / Number Input ---- */
[data-testid="stSelectbox"], [data-testid="stNumberInput"] {
    border-color: #2A2520 !important;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------
# MATPLOTLIB DARK GOLD THEME
# --------------------------------------------
def apply_plot_theme():
    mpl.rcParams.update({
        "figure.facecolor":  "#13111A",
        "axes.facecolor":    "#13111A",
        "axes.edgecolor":    "#2A2520",
        "axes.labelcolor":   "#9A8E7F",
        "axes.titlecolor":   "#E8DCC8",
        "xtick.color":       "#6A5E4F",
        "ytick.color":       "#6A5E4F",
        "text.color":        "#E8DCC8",
        "grid.color":        "#1E1C24",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.5,
        "figure.dpi":        140,
        "font.family":       "serif",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

apply_plot_theme()
GOLD = "#C9A84C"
GOLD2 = "#8B6914"


# --------------------------------------------
# LOAD DATA & MODEL
# --------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/train.csv")

@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    columns = joblib.load("models/feature_columns.pkl")
    return model, columns

df = load_data()

model_loaded = False
try:
    model, feature_columns = load_model()
    model_loaded = True
except Exception:
    st.warning("⚠️ No trained model found. Run `python main.py` to train first.")
    model, feature_columns = None, []


# --------------------------------------------
# HERO HEADER
# --------------------------------------------
st.markdown('<div class="hero-title">EstateIQ</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">AI-Powered House Price Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-divider"></div>', unsafe_allow_html=True)

# Dataset stats row
col1, col2, col3, col4 = st.columns(4)
numeric_df = df.select_dtypes(include="number")

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Properties</div>
        <div class="metric-value">{len(df):,}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    avg_price = int(df["SalePrice"].mean()) if "SalePrice" in df.columns else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Average Sale Price</div>
        <div class="metric-value">${avg_price:,}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    med_price = int(df["SalePrice"].median()) if "SalePrice" in df.columns else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Median Price</div>
        <div class="metric-value">${med_price:,}</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Features Available</div>
        <div class="metric-value">{df.shape[1] - 1}</div>
    </div>""", unsafe_allow_html=True)


# --------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------
st.sidebar.markdown('<div class="sidebar-title">🏛️ Property Details</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

lot_area        = st.sidebar.slider("Lot Area (sq ft)",      1000, 20000, 8000, step=100)
overall_qual    = st.sidebar.slider("Overall Quality",        1, 10, 7)
overall_cond    = st.sidebar.slider("Overall Condition",      1, 10, 5)
year_built      = st.sidebar.slider("Year Built",             1900, 2023, 2000)
total_bsmt_sf   = st.sidebar.slider("Basement Area (sq ft)",  0, 3000, 800, step=50)
gr_liv_area     = st.sidebar.slider("Living Area (sq ft)",    500, 5000, 1500, step=50)
bedrooms        = st.sidebar.slider("Bedrooms Above Grade",   0, 8, 3)
full_bath       = st.sidebar.slider("Full Bathrooms",         0, 4, 2)
garage_cars     = st.sidebar.slider("Garage Capacity (cars)", 0, 4, 2)
garage_area     = st.sidebar.slider("Garage Area (sq ft)",    0, 1500, 480, step=20)

st.sidebar.markdown("---")
predict_clicked = st.sidebar.button("✦ Estimate Price")


# --------------------------------------------
# PREDICTION
# --------------------------------------------
if predict_clicked and model_loaded:
    feature_dict = {
        "LotArea":       lot_area,
        "OverallQual":   overall_qual,
        "OverallCond":   overall_cond,
        "YearBuilt":     year_built,
        "TotalBsmtSF":   total_bsmt_sf,
        "GrLivArea":     gr_liv_area,
        "BedroomAbvGr":  bedrooms,
        "FullBath":      full_bath,
        "GarageCars":    garage_cars,
        "GarageArea":    garage_area,
    }

    input_df = pd.DataFrame([np.zeros(len(feature_columns))], columns=feature_columns)
    for col, val in feature_dict.items():
        if col in input_df.columns:
            input_df[col] = val

    prediction = model.predict(input_df)[0]

    st.markdown(f"""
    <div class="prediction-banner">
        <div class="prediction-label">Estimated Market Value</div>
        <div class="prediction-price">${int(prediction):,}</div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------
# TABS — ANALYTICS
# --------------------------------------------
st.markdown('<div class="section-header">Market Analytics</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "  Price Distribution  ",
    "  Correlation Analysis  ",
    "  Feature Intelligence  ",
    "  Raw Dataset  "
])


# ---- TAB 1: Price Distribution ----
with tab1:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor("#13111A")

    # Histogram
    axes[0].hist(df["SalePrice"], bins=50, color=GOLD, alpha=0.85, edgecolor="#0C0C0E", linewidth=0.4)
    axes[0].set_title("Sale Price Distribution", fontsize=13, pad=12, color="#E8DCC8")
    axes[0].set_xlabel("Sale Price ($)", fontsize=9)
    axes[0].set_ylabel("Count", fontsize=9)
    axes[0].yaxis.grid(True, alpha=0.3)

    # Log scale
    axes[1].hist(np.log1p(df["SalePrice"]), bins=50, color=GOLD2, alpha=0.85,
                 edgecolor="#0C0C0E", linewidth=0.4)
    axes[1].set_title("Log(Sale Price) Distribution", fontsize=13, pad=12, color="#E8DCC8")
    axes[1].set_xlabel("log(1 + Sale Price)", fontsize=9)
    axes[1].set_ylabel("Count", fontsize=9)
    axes[1].yaxis.grid(True, alpha=0.3)

    plt.tight_layout(pad=2)
    st.pyplot(fig)
    plt.close(fig)


# ---- TAB 2: Correlation Heatmap ----
with tab2:
    top_n = st.slider("Number of top features to show", 8, 20, 12, key="heatmap_slider")

    corr = numeric_df.corr()
    top_features = corr["SalePrice"].abs().sort_values(ascending=False).head(top_n).index

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#13111A")

    mask = np.zeros_like(corr.loc[top_features, top_features], dtype=bool)
    np.fill_diagonal(mask, True)

    cmap = sns.diverging_palette(220, 40, s=60, l=35, as_cmap=True)
    sns.heatmap(
        corr.loc[top_features, top_features],
        annot=True, fmt=".2f", linewidths=0.5,
        cmap=cmap, center=0, square=True,
        annot_kws={"size": 7, "color": "#E8DCC8"},
        linecolor="#0C0C0E",
        ax=ax, mask=mask,
        cbar_kws={"shrink": 0.7}
    )
    ax.set_title(f"Top {top_n} Feature Correlations", fontsize=14, pad=15, color="#E8DCC8")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---- TAB 3: Feature Importance ----
with tab3:
    fi_path = "images/feature_importance.png"

    if os.path.exists(fi_path):
        st.image(fi_path, use_column_width=True)
    else:
        # Generate bar chart from correlation
        corr = numeric_df.corr()["SalePrice"].drop("SalePrice").abs().sort_values(ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#13111A")

        colors = [GOLD if v > 0.5 else GOLD2 for v in corr.values]
        corr[::-1].plot(kind="barh", ax=ax, color=colors[::-1], edgecolor="none")
        ax.set_title("Feature Correlation with Sale Price (top 20)", fontsize=13, pad=12, color="#E8DCC8")
        ax.set_xlabel("Absolute Correlation", fontsize=9)
        ax.xaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        <div class="info-box">
            ℹ️ Train the model first to see ML-based feature importance from the best model.
        </div>
        """, unsafe_allow_html=True)


# ---- TAB 4: Dataset Preview ----
with tab4:
    search_col = st.text_input("Filter by column name", placeholder="e.g. Garage, Bath...")
    display_df = df
    if search_col:
        matching = [c for c in df.columns if search_col.lower() in c.lower()]
        display_df = df[matching] if matching else df

    st.dataframe(
        display_df.head(100),
        use_container_width=True,
        height=420
    )
    st.markdown(f"*Showing top 100 of {len(df):,} rows · {df.shape[1]} columns*")


# --------------------------------------------
# FOOTER
# --------------------------------------------
st.markdown("""
<div style="text-align:center; margin-top:4rem; padding-top:1.5rem;
            border-top:1px solid #1E1C24; color:#3A3530; font-size:0.75rem;
            letter-spacing:0.12em; text-transform:uppercase;">
    EstateIQ · AI House Price Intelligence
</div>
""", unsafe_allow_html=True)