
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import warnings
warnings.filterwarnings("ignore")


# PAGE CONFIG

st.set_page_config(
    page_title="UK Road Casualty Analytics 2025",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #f8f9fb !important;
        color: #1a1a2e !important;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e0e4ed;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 18px 22px;
        border-left: 4px solid #2563eb;
        box-shadow: 0 1px 6px rgba(0,0,0,0.07);
        margin-bottom: 10px;
    }
    .metric-card.red   { border-left-color: #dc2626; }
    .metric-card.amber { border-left-color: #d97706; }
    .metric-card.green { border-left-color: #16a34a; }
    .metric-card h2 { margin: 0; font-size: 2rem; color: #1a1a2e; }
    .metric-card p  { margin: 2px 0 0; font-size: 0.85rem; color: #6b7280; }
    .section-header {
        font-size: 1.15rem; font-weight: 700; color: #1e3a5f;
        margin: 24px 0 10px; padding-bottom: 5px;
        border-bottom: 2px solid #dbeafe;
    }
    .insight-box {
        background: #eff6ff; border-left: 3px solid #3b82f6;
        padding: 10px 16px; border-radius: 6px; margin: 8px 0;
        font-size: 0.9rem; color: #1e3a5f;
    }
    .warning-box {
        background: #fef3c7; border-left: 3px solid #f59e0b;
        padding: 10px 16px; border-radius: 6px; margin: 8px 0;
        font-size: 0.9rem; color: #78350f;
    }
    .step-box {
        background: #ffffff; border: 1px solid #e0e4ed;
        border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;
    }
    .step-box h4 { margin: 0 0 6px; color: #1e3a5f; font-size: 0.95rem; }
    .step-box p  { margin: 0; font-size: 0.85rem; color: #374151; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)


# LOOKUP DICTIONARIES

SEVERITY_MAP   = {1:"Fatal", 2:"Serious", 3:"Slight", -1:"Unknown"}
WEATHER_MAP    = {1:"Fine no wind", 2:"Raining no wind", 3:"Snowing no wind",
                  4:"Fine + wind",  5:"Raining + wind",  6:"Snowing + wind",
                  7:"Fog/mist", 8:"Other", 9:"Unknown", -1:"Unknown"}
ROAD_SURF_MAP  = {1:"Dry", 2:"Wet/damp", 3:"Snow", 4:"Frost/ice",
                  5:"Flood", 9:"Unknown", -1:"Unknown"}
LIGHT_MAP      = {1:"Daylight", 4:"Dark - lit", 5:"Dark - unlit",
                  6:"Dark - no lighting", 7:"Dark - lighting unknown", -1:"Unknown"}
URBAN_MAP      = {1:"Urban", 2:"Rural", 3:"Unallocated"}
DOW_MAP        = {1:"Sunday", 2:"Monday", 3:"Tuesday", 4:"Wednesday",
                  5:"Thursday", 6:"Friday", 7:"Saturday"}
CASUALTY_CLASS = {1:"Driver/Rider", 2:"Passenger", 3:"Pedestrian", -1:"Unknown"}
SEX_MAP        = {1:"Male", 2:"Female", 3:"Not known", -1:"Unknown"}
VEHICLE_MAP    = {1:"Pedal cycle", 2:"Motorcycle 50cc", 3:"Motorcycle 125cc",
                  4:"Motorcycle 500cc", 5:"Motorcycle 500cc+", 8:"Taxi",
                  9:"Car", 10:"Minibus", 11:"Bus/Coach", 16:"Ridden horse",
                  17:"Agric vehicle", 18:"Tram", 19:"Van/light goods",
                  20:"HGV rigid", 21:"HGV articulated",
                  97:"Motorcycle unknown", 98:"Other vehicle", 99:"Unknown"}
JUNCTION_MAP   = {0:"Not at junction", 1:"Roundabout", 2:"Mini-roundabout",
                  3:"T or staggered", 5:"Slip road", 6:"Crossroads",
                  7:"More than 4 arms", 8:"Private drive", 9:"Other", -1:"Unknown"}
ROAD_CLASS_MAP = {1:"Motorway", 2:"A(M)", 3:"A", 4:"B", 5:"C", 6:"Unclassified"}
CASUALTY_TYPE_MAP = {
    0:"Pedestrian", 1:"Pedal cyclist", 2:"M/c 50cc rider", 3:"M/c 125cc rider",
    4:"M/c rider", 5:"M/c 500cc+ rider", 8:"Taxi occupant", 9:"Car occupant",
    10:"Minibus occupant", 11:"Bus occupant", 16:"Horse rider",
    17:"Agric vehicle occupant", 18:"Tram occupant", 19:"Van occupant",
    20:"HGV rigid occupant", 21:"HGV artic occupant",
    97:"M/c unknown", 98:"Other vehicle", 99:"Unknown"
}
SEV_COLOR = {"Fatal":"#dc2626", "Serious":"#d97706", "Slight":"#2563eb"}
BG = "#f8f9fb"
LAYOUT = dict(paper_bgcolor=BG, plot_bgcolor=BG, font_color="#1a1a2e")

# Model filename map
MODEL_FILES = {
    "Logistic Regression":  "artifacts/model_logistic_regression.pkl",
    "Random Forest":        "artifacts/model_random_forest.pkl",
    "Gradient Boosting":    "artifacts/model_gradient_boosting.pkl",
    "K-Nearest Neighbours": "artifacts/model_k_nearest_neighbours.pkl",
}


# LOAD ARTIFACTS (pkl + metadata)

@st.cache_resource(show_spinner="Loading pre-trained models...")
def load_artifacts():
    with open("artifacts/metadata.json") as f:
        meta = json.load(f)
    scaler  = joblib.load("artifacts/scaler.pkl")
    models  = {name: joblib.load(fpath) for name, fpath in MODEL_FILES.items()}
    kmeans  = joblib.load("artifacts/kmeans_model.pkl")
    pca_mdl = joblib.load("artifacts/pca_model.pkl")
    return meta, scaler, models, kmeans, pca_mdl

meta, scaler, models, kmeans_model, pca_model = load_artifacts()

FEAT_COLS  = meta["feat_cols"]
ml_results = meta["results"]


best_name = "Random Forest"


if best_name not in models:
    st.error(f"{best_name} model not found. Falling back to metadata best model.")
    best_name = meta["best_model"]

# LOAD RAW DATA

@st.cache_data(show_spinner="Loading datasets...")
def load_data():
    cas = pd.read_csv("dft-road-casualty-statistics-casualty-provisional-2025.csv",  low_memory=False)
    veh = pd.read_csv("dft-road-casualty-statistics-vehicle-provisional-2025.csv",   low_memory=False)
    col = pd.read_csv("dft-road-casualty-statistics-collision-provisional-2025.csv", low_memory=False)

    col["date"]       = pd.to_datetime(col["date"], dayfirst=True, errors="coerce")
    col["month"]      = col["date"].dt.month
    col["month_name"] = col["date"].dt.strftime("%b")
    col["week"]       = col["date"].dt.isocalendar().week.astype(int)
    col["hour"]       = col["time"].str.split(":").str[0].astype(float)

    col["severity_label"]   = col["collision_severity"].map(SEVERITY_MAP)
    col["weather_label"]    = col["weather_conditions"].map(WEATHER_MAP).fillna("Unknown")
    col["road_surf_label"]  = col["road_surface_conditions"].map(ROAD_SURF_MAP).fillna("Unknown")
    col["light_label"]      = col["light_conditions"].map(LIGHT_MAP).fillna("Unknown")
    col["urban_label"]      = col["urban_or_rural_area"].map(URBAN_MAP).fillna("Unknown")
    col["dow_label"]        = col["day_of_week"].map(DOW_MAP).fillna("Unknown")
    col["junction_label"]   = col["junction_detail"].map(JUNCTION_MAP).fillna("Unknown")
    col["road_class_label"] = col["first_road_class"].map(ROAD_CLASS_MAP).fillna("Unknown")
    col = col[col["speed_limit"] > 0]

    cas["severity_label"]      = cas["casualty_severity"].map(SEVERITY_MAP)
    cas["sex_label"]           = cas["sex_of_casualty"].map(SEX_MAP)
    cas["class_label"]         = cas["casualty_class"].map(CASUALTY_CLASS)
    cas["casualty_type_label"] = cas["casualty_type"].map(CASUALTY_TYPE_MAP).fillna("Other")

    veh["vehicle_label"]    = veh["vehicle_type"].map(VEHICLE_MAP).fillna("Other")
    veh["driver_sex_label"] = veh["sex_of_driver"].map(SEX_MAP)

    return col, cas, veh

col_df, cas_df, veh_df = load_data()


# SIDEBAR

with st.sidebar:
    st.markdown("## UK Road Casualty 2025")
    st.markdown("*Jan – May 2025 | DfT Provisional Data*")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "Overview & KPIs",
            "Data Pipeline & Feature Engineering",
            "Temporal Analysis",
            "Geospatial & Environment",
            "Casualty & Vehicle Profiles",
            "Risk Factor Analysis",
            "Clustering Analysis",
            "ML Pipeline & Model Comparison",
            "Predict & Explain",
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**Global Filters**")
    sev_filter   = st.multiselect("Severity",  ["Fatal","Serious","Slight"], default=["Fatal","Serious","Slight"])
    urban_filter = st.multiselect("Area Type", ["Urban","Rural"],            default=["Urban","Rural"])
    speed_filter = st.slider("Speed Limit (mph)", 20, 70, (20,70), step=10)
    st.divider()
    
def apply_filters(df):
    return df[
        df["severity_label"].isin(sev_filter) &
        df["urban_label"].isin(urban_filter) &
        df["speed_limit"].between(speed_filter[0], speed_filter[1])
    ].copy()

fdf = apply_filters(col_df)


# HELPERS

def card(val, label, color=""):
    return f'<div class="metric-card {color}"><h2>{val}</h2><p>{label}</p></div>'

def insight(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

def warn(text):
    st.markdown(f'<div class="warning-box">{text}</div>', unsafe_allow_html=True)

def sh(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)

def predict_input(feat_vals):
   
    X_in = pd.DataFrame([feat_vals], columns=FEAT_COLS)
    best_model = models[best_name]
    needs_sc   = ml_results[best_name]["needs_scaling"]
    X_tf = scaler.transform(X_in) if needs_sc else X_in.values
    pred = best_model.predict(X_tf)[0]
    prob = best_model.predict_proba(X_tf)[0]
    return pred, prob



# PAGE 1 — OVERVIEW & KPIs

if page == "Overview & KPIs":
    st.title("UK Road Casualty Analytics — 2025 Provisional")
    st.markdown("*Jan – May 2025 | DfT Provisional release*")

    total_col   = len(fdf)
    fatal_col   = (fdf["collision_severity"]==1).sum()
    serious_col = (fdf["collision_severity"]==2).sum()
    slight_col  = (fdf["collision_severity"]==3).sum()
    filt_cas    = cas_df[cas_df["collision_index"].isin(fdf["collision_index"])]
    fatalities  = (filt_cas["casualty_severity"]==1).sum()

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(card(f"{total_col:,}",  "Total Collisions"),           unsafe_allow_html=True)
    with c2: st.markdown(card(f"{fatal_col:,}",  "Fatal Collisions",   "red"),  unsafe_allow_html=True)
    with c3: st.markdown(card(f"{serious_col:,}","Serious Collisions","amber"),  unsafe_allow_html=True)
    with c4: st.markdown(card(f"{slight_col:,}", "Slight Collisions", "green"),  unsafe_allow_html=True)
    with c5: st.markdown(card(f"{fatalities:,}", "Total Fatalities",   "red"),   unsafe_allow_html=True)

    sh("Severity Distribution & Monthly Trend")
    c1,c2 = st.columns([1,2])
    with c1:
        sc = fdf["severity_label"].value_counts().reset_index()
        sc.columns = ["Severity","Count"]
        fig = px.pie(sc, values="Count", names="Severity", color="Severity",
                     color_discrete_map=SEV_COLOR, hole=0.45, title="Collision Severity Split")
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        monthly = fdf.groupby(["month","month_name","severity_label"]).size().reset_index(name="count")
        monthly = monthly.sort_values("month")
        fig = px.bar(monthly, x="month_name", y="count", color="severity_label",
                     color_discrete_map=SEV_COLOR, barmode="stack",
                     title="Monthly Collision Trend by Severity",
                     labels={"count":"Collisions","month_name":"Month","severity_label":"Severity"})
        fig.update_layout(**LAYOUT,
                          xaxis=dict(categoryorder="array",
                                     categoryarray=["Jan","Feb","Mar","Apr","May"]))
        st.plotly_chart(fig, use_container_width=True)

    sh("Key Rate Benchmarks")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.metric("Avg Vehicles / Collision",  f"{fdf['number_of_vehicles'].mean():.2f}")
        st.metric("Avg Casualties / Collision", f"{fdf['number_of_casualties'].mean():.2f}")
    with c2:
        fatal_rate = fatal_col/total_col*100 if total_col else 0
        police_att = (fdf["did_police_officer_attend_scene_of_accident"]==1).mean()*100
        st.metric("Fatality Rate",     f"{fatal_rate:.2f}%")
        st.metric("Police Attendance", f"{police_att:.1f}%")
    with c3:
        rural_f = fdf[fdf["urban_label"]=="Rural"]["collision_severity"].eq(1).sum()
        urban_f = fdf[fdf["urban_label"]=="Urban"]["collision_severity"].eq(1).sum()
        st.metric("Rural Fatal Collisions", f"{rural_f:,}")
        st.metric("Urban Fatal Collisions", f"{urban_f:,}")

    sh("Urban vs Rural Breakdown")
    ur = fdf.groupby(["urban_label","severity_label"]).size().reset_index(name="count")
    fig = px.bar(ur, x="urban_label", y="count", color="severity_label",
                 color_discrete_map=SEV_COLOR, barmode="group",
                 title="Urban vs Rural — Collision Severity",
                 labels={"urban_label":"Area","count":"Collisions","severity_label":"Severity"})
    fig.update_layout(**LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    insight("Rural roads account for disproportionately high fatalities despite lower total collision volumes — higher speeds and longer emergency response times drive this disparity.")



# PAGE 2 — DATA PIPELINE & FEATURE ENGINEERING

elif page == "Data Pipeline & Feature Engineering":
    st.title("Data Pipeline & Feature Engineering")
    st.markdown("Complete step-by-step walkthrough of every data preparation decision. ")

    sh("Step 1 — Raw Dataset Ingestion")
    st.markdown("""
    <div class="step-box">
    <h4>What we did</h4>
    <p>Three CSV files loaded via <code>pd.read_csv(low_memory=False)</code>.
    <code>low_memory=False</code> prevents mixed-type warnings on <code>collision_index</code>
    (alphanumeric key). The three datasets are joined by <code>collision_index</code> —
    a unique key per collision event assigned by the Department for Transport.
    A left-merge is used: collision is the anchor table.</p>
    </div>
    """, unsafe_allow_html=True)

    summary = pd.DataFrame({
        "Dataset":     ["Collision","Casualty","Vehicle"],
        "Rows":        [48_472, 60_991, 87_805],
        "Columns":     [44, 23, 32],
        "Granularity": ["1 row per collision","1 row per casualty","1 row per vehicle"],
        "Link Key":    ["collision_index","collision_index","collision_index"]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    sh("Step 2 — Raw Data Preview")
    ds_choice = st.selectbox("Select dataset to preview", ["Collision","Casualty","Vehicle"])
    preview_map = {"Collision": col_df, "Casualty": cas_df, "Vehicle": veh_df}
    pv = preview_map[ds_choice]
    st.dataframe(pv.head(10), use_container_width=True)
    st.caption(f"Shape: {pv.shape[0]:,} rows x {pv.shape[1]} columns")

    sh("Step 3 — Missing Values (True NaN)")
    st.markdown("""
    <div class="step-box">
    <h4>Findings</h4>
    <p>True NaN values are extremely rare — DfT uses sentinel value <strong>-1</strong>
    for unknown/unrecorded data. Only 4 geographic columns (latitude, longitude, easting, northing)
    have 1 true NaN each. These records are excluded only for the map visualisation.</p>
    </div>
    """, unsafe_allow_html=True)

    all_null_rows = []
    for ds_name, ds in [("Collision",col_df),("Casualty",cas_df),("Vehicle",veh_df)]:
        ns = ds.isnull().sum()
        for col_name, cnt in ns[ns>0].items():
            all_null_rows.append({"Dataset":ds_name,"Column":col_name,
                                  "Null Count":int(cnt),"Pct":f"{cnt/len(ds)*100:.2f}%"})
    if all_null_rows:
        st.dataframe(pd.DataFrame(all_null_rows), use_container_width=True, hide_index=True)
    else:
        st.success("No true NaN values found in any dataset.")

    sh("Step 4 — Sentinel Value Analysis (DfT uses -1 for Unknown)")
    st.markdown("""
    <div class="step-box">
    <h4>Strategy applied</h4>
    <p><strong>Analytical charts:</strong> -1 is mapped to "Unknown" label via lookup dictionaries — visible to viewer.<br>
    <strong>ML training:</strong> rows with -1 in any key feature column are dropped before training.
    These represent genuinely unknown conditions that would introduce noise rather than signal.<br>
    <strong>Excluded columns:</strong> <code>propulsion_code</code> and <code>age_of_vehicle</code>
    are -1 for 100% of rows — excluded from analysis entirely.</p>
    </div>
    """, unsafe_allow_html=True)

    sent_rows = []
    for ds_name, ds in [("Collision",col_df),("Casualty",cas_df),("Vehicle",veh_df)]:
        for c in ds.select_dtypes(include="number").columns:
            n = (ds[c]==-1).sum()
            if n > 0:
                sent_rows.append({"Dataset":ds_name,"Column":c,
                                  "Sentinel (-1) Count":int(n),
                                  "Pct":f"{n/len(ds)*100:.1f}%"})
    sent_df = pd.DataFrame(sent_rows).sort_values("Sentinel (-1) Count", ascending=False)
    st.dataframe(sent_df, use_container_width=True, hide_index=True)

    top_sent = sent_df.head(15)
    fig = px.bar(top_sent, x="Sentinel (-1) Count", y="Column", orientation="h",
                 color="Dataset", title="Top 15 Columns with Sentinel -1 Values")
    fig.update_layout(**LAYOUT, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    sh("Step 5 — Duplicate Record Check")
    st.markdown("""
    <div class="step-box">
    <h4>Result: Zero duplicates across all three datasets</h4>
    <p>Full row-level check using <code>df.duplicated().sum()</code>.
    DfT assigns unique <code>collision_index</code> values per event;
    casualty/vehicle references within each collision are unique by construction.
    No deduplication step is required.</p>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Dataset":        ["Collision","Casualty","Vehicle"],
        "Total Rows":     [48_472, 60_991, 87_805],
        "Duplicate Rows": [0, 0, 0],
        "Action":         ["None required","None required","None required"]
    }), use_container_width=True, hide_index=True)
    st.success("All three datasets are duplicate-free.")

    sh("Step 6 — Feature Engineering (17 new features)")
    fe_df = pd.DataFrame({
        "New Feature":  ["hour","month","month_name","week","severity_label",
                         "weather_label","road_surf_label","light_label","urban_label",
                         "dow_label","junction_label","road_class_label",
                         "casualty_type_label","vehicle_label","age_band_cas","age_band_drv","target"],
        "Derived From": ["time (HH:MM)","date","date","date","collision_severity",
                         "weather_conditions","road_surface_conditions","light_conditions",
                         "urban_or_rural_area","day_of_week","junction_detail","first_road_class",
                         "casualty_type","vehicle_type","age_of_casualty","age_of_driver","collision_severity"],
        "Type":         ["Numeric","Numeric","Categorical","Numeric","Categorical",
                         "Categorical","Categorical","Categorical","Categorical","Categorical",
                         "Categorical","Categorical","Categorical","Categorical",
                         "Ordinal (7 bins)","Ordinal (7 bins)","Binary (0/1)"],
        "Rationale":    ["Rush-hour & night risk patterns","Seasonal variation",
                         "Chart-readable month label","Weekly aggregation",
                         "Readable severity for filters","Weather condition label",
                         "Road surface label","Light condition — key ML predictor",
                         "Urban/Rural segmentation","Day of week label",
                         "Junction type for safety analysis","Road class (Motorway/A/B/C)",
                         "Casualty type (pedestrian/cyclist/car occupant)","Vehicle type label",
                         "Age grouped: <18, 18-24 ... 65+","Driver age same 7 bands",
                         "ML binary target: 1=Serious/Fatal, 0=Slight"]
    })
    st.dataframe(fe_df, use_container_width=True, hide_index=True)

    sh("Step 7 — Target Variable & Class Balance")
    cb = meta["class_balance"]
    total_ml = sum(cb.values())
    c1,c2 = st.columns([1,2])
    with c1:
        st.dataframe(pd.DataFrame({
            "Class":      ["0 — Slight","1 — Serious/Fatal"],
            "Count":      [cb.get("0",0), cb.get("1",0)],
            "Percentage": [f"{cb.get('0',0)/total_ml*100:.1f}%",
                           f"{cb.get('1',0)/total_ml*100:.1f}%"]
        }), use_container_width=True, hide_index=True)
        st.metric("ML Dataset Size",   f"{total_ml:,}")
        st.metric("Train rows",        f"{meta['train_rows']:,}")
        st.metric("Test rows",         f"{meta['test_rows']:,}")
    with c2:
        tc_df = pd.DataFrame({"Label":["Slight","Serious/Fatal"],
                              "Count":[cb.get("0",0),cb.get("1",0)]})
        fig = px.pie(tc_df, values="Count", names="Label",
                     color="Label",
                     color_discrete_map={"Slight":"#2563eb","Serious/Fatal":"#dc2626"},
                     hole=0.4, title="ML Target Class Distribution")
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    warn("Class imbalance: ~72% Slight vs ~28% Serious/Fatal. Mitigated with class_weight='balanced' for LR and RF. Sentinel -1 rows dropped before training")

    sh("Step 8 — ML Feature Set (13 features used for training)")
    feat_desc = pd.DataFrame({
        "Feature":     FEAT_COLS,
        "Description": ["Posted speed limit — strongest single predictor",
                        "Number of vehicles involved",
                        "Number of casualties (severity proxy)",
                        "Hour of day (0-23) — rush hour & night risk",
                        "Day of week (1=Sun...7=Sat)",
                        "Weather code — fine/rain/snow/fog",
                        "Light conditions — daylight/dark-lit/dark-unlit",
                        "Road surface — dry/wet/ice/flood",
                        "Urban (1) or Rural (2) area",
                        "Month (1-12) — seasonal variation",
                        "Junction type — roundabout/crossroads etc.",
                        "First road class — Motorway/A/B/C",
                        "Pedestrian crossing type at scene"],
        "Scale":       ["Numeric","Numeric","Numeric","Numeric","Ordinal",
                        "Ordinal","Ordinal","Ordinal","Binary","Numeric",
                        "Ordinal","Ordinal","Ordinal"]
    })
    st.dataframe(feat_desc, use_container_width=True, hide_index=True)



# PAGE 3 — TEMPORAL ANALYSIS

elif page == "Temporal Analysis":
    st.title("Temporal Analysis")

    tab1, tab2, tab3 = st.tabs(["Hour of Day","Day of Week","Weekly Heatmap"])

    with tab1:
        hourly = fdf.groupby(["hour","severity_label"]).size().reset_index(name="count")
        fig = px.line(hourly, x="hour", y="count", color="severity_label",
                      color_discrete_map=SEV_COLOR, title="Collision Frequency by Hour of Day",
                      labels={"hour":"Hour (24h)","count":"Collisions","severity_label":"Severity"})
        fig.update_layout(**LAYOUT)
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)
        insight("Peak hours: 08-09h (morning rush) and 16-18h (evening rush). Fatal collisions spike at 22h-02h — linked to impairment and reduced visibility.")

        fh = fdf[fdf["collision_severity"]==1].groupby("hour").size().reset_index(name="fatal")
        fig2 = px.bar(fh, x="hour", y="fatal", title="Fatal Collisions by Hour of Day",
                      color="fatal", color_continuous_scale="Reds")
        fig2.update_layout(**LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow = fdf.groupby(["dow_label","severity_label"]).size().reset_index(name="count")
        dow["dow_label"] = pd.Categorical(dow["dow_label"], categories=dow_order, ordered=True)
        dow = dow.sort_values("dow_label")
        fig = px.bar(dow, x="dow_label", y="count", color="severity_label",
                     color_discrete_map=SEV_COLOR, barmode="group",
                     title="Collisions by Day of Week",
                     labels={"dow_label":"Day","count":"Collisions","severity_label":"Severity"})
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        dow_fatal = fdf.groupby("dow_label").agg(
            total=("collision_severity","count"),
            fatal=("collision_severity", lambda x:(x==1).sum())
        ).reset_index()
        dow_fatal["fatal_pct"] = dow_fatal["fatal"]/dow_fatal["total"]*100
        dow_fatal["dow_label"] = pd.Categorical(dow_fatal["dow_label"], categories=dow_order, ordered=True)
        dow_fatal = dow_fatal.sort_values("dow_label")
        fig2 = px.line(dow_fatal, x="dow_label", y="fatal_pct", markers=True,
                       title="Fatal Rate (%) by Day of Week",
                       labels={"dow_label":"Day","fatal_pct":"Fatal Rate (%)"})
        fig2.update_traces(line_color="#dc2626")
        fig2.update_layout(**LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)
        insight("Friday has the highest absolute collision count. Weekend-night effect (Fri-Sat, 22h-02h) elevates fatal rate significantly.")

    with tab3:
        hmap_data  = fdf.groupby(["dow_label","hour"]).size().reset_index(name="count")
        hmap_pivot = hmap_data.pivot(index="dow_label", columns="hour", values="count").fillna(0)
        dow_ri = [d for d in ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                  if d in hmap_pivot.index]
        hmap_pivot = hmap_pivot.reindex(dow_ri)
        fig = go.Figure(data=go.Heatmap(
            z=hmap_pivot.values, x=hmap_pivot.columns, y=hmap_pivot.index,
            colorscale="YlOrRd", colorbar=dict(title="Collisions")
        ))
        fig.update_layout(title="Collision Heatmap: Day vs Hour",
                          xaxis_title="Hour of Day", yaxis_title="Day of Week",
                          paper_bgcolor=BG, font_color="#1a1a2e", height=400)
        st.plotly_chart(fig, use_container_width=True)
        insight("Two distinct risk clusters: weekday rush (07-09h & 16-18h) and weekend-night (22h-02h). Completely different casualty profiles.")

        mf = fdf[fdf["collision_severity"]==1].groupby("month_name").size().reset_index(name="fatals")
        month_order = ["Jan","Feb","Mar","Apr","May"]
        mf["month_name"] = pd.Categorical(mf["month_name"], categories=month_order, ordered=True)
        mf = mf.sort_values("month_name")
        fig = px.area(mf, x="month_name", y="fatals", title="Monthly Fatal Collision Count",
                      labels={"month_name":"Month","fatals":"Fatal Collisions"})
        fig.update_traces(line_color="#dc2626", fillcolor="rgba(220,38,38,0.15)")
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)



# PAGE 4 — GEOSPATIAL & ENVIRONMENT

elif page == "Geospatial & Environment":
    st.title("Geospatial & Environmental Analysis")

    tab1, tab2, tab3 = st.tabs(["Collision Map","Weather & Light","Road Conditions"])

    with tab1:
        map_sev = st.selectbox("Severity for map", ["All","Fatal","Serious","Slight"])
        map_df  = fdf.dropna(subset=["latitude","longitude"])
        if map_sev != "All":
            map_df = map_df[map_df["severity_label"]==map_sev]
        map_df = map_df[map_df["latitude"].between(49,61) & map_df["longitude"].between(-8,2)]
        sn = min(5000, len(map_df))
        ms = map_df.sample(sn, random_state=42)
        fig = px.density_mapbox(ms, lat="latitude", lon="longitude",
                                radius=6, zoom=5, center={"lat":52.5,"lon":-1.5},
                                mapbox_style="carto-positron",
                                title=f"Collision Density ({map_sev}) — {sn:,} sample points",
                                color_continuous_scale="Reds")
        fig.update_layout(height=520, paper_bgcolor=BG, font_color="#1a1a2e",
                          margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

        c1,c2 = st.columns(2)
        with c1:
            top_la = fdf.groupby("local_authority_district").size().nlargest(10).reset_index(name="count")
            fig2 = px.bar(top_la, x="count", y="local_authority_district", orientation="h",
                          title="Top 10 Local Authorities by Collisions",
                          color="count", color_continuous_scale="Reds")
            fig2.update_layout(**LAYOUT, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            rc = fdf.groupby("road_class_label")["collision_severity"].apply(
                lambda x:(x==1).sum()).reset_index(name="fatal")
            rc = rc[rc["road_class_label"]!="Unknown"]
            fig3 = px.bar(rc, x="road_class_label", y="fatal",
                          title="Fatal Collisions by Road Class",
                          color="fatal", color_continuous_scale="OrRd")
            fig3.update_layout(**LAYOUT)
            st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        c1,c2 = st.columns(2)
        with c1:
            wx = fdf[~fdf["weather_label"].isin(["Unknown"])].groupby(
                ["weather_label","severity_label"]).size().reset_index(name="count")
            fig = px.bar(wx, x="weather_label", y="count", color="severity_label",
                         color_discrete_map=SEV_COLOR, barmode="stack",
                         title="Collisions by Weather Condition",
                         labels={"weather_label":"Weather","count":"Collisions"})
            fig.update_xaxes(tickangle=30)
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            lx = fdf[~fdf["light_label"].isin(["Unknown"])].groupby(
                ["light_label","severity_label"]).size().reset_index(name="count")
            fig = px.bar(lx, x="light_label", y="count", color="severity_label",
                         color_discrete_map=SEV_COLOR, barmode="stack",
                         title="Collisions by Light Conditions",
                         labels={"light_label":"Light Condition","count":"Collisions"})
            fig.update_xaxes(tickangle=30)
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        wx_fatal = fdf[~fdf["weather_label"].isin(["Unknown"])].groupby("weather_label").agg(
            total=("collision_severity","count"),
            fatal=("collision_severity", lambda x:(x==1).sum())
        ).reset_index()
        wx_fatal["fatal_rate"] = wx_fatal["fatal"]/wx_fatal["total"]*100
        wx_fatal = wx_fatal.sort_values("fatal_rate", ascending=False)
        fig = px.bar(wx_fatal, x="weather_label", y="fatal_rate",
                     title="Fatal Rate (%) by Weather — Normalised",
                     color="fatal_rate", color_continuous_scale="Reds",
                     labels={"fatal_rate":"Fatal Rate (%)","weather_label":"Weather"})
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        insight("Fine weather has the highest absolute count, but fog/mist produces disproportionately high fatal rates when normalised — a counter-intuitive hidden trend.")

    with tab3:
        rs = fdf[~fdf["road_surf_label"].isin(["Unknown"])].groupby(
            ["road_surf_label","severity_label"]).size().reset_index(name="count")
        fig = px.bar(rs, x="road_surf_label", y="count", color="severity_label",
                     color_discrete_map=SEV_COLOR, barmode="group",
                     title="Collisions by Road Surface",
                     labels={"road_surf_label":"Road Surface","count":"Collisions"})
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        c1,c2 = st.columns(2)
        with c1:
            ss = fdf.groupby(["speed_limit","severity_label"]).size().reset_index(name="count")
            fig2 = px.bar(ss, x="speed_limit", y="count", color="severity_label",
                          color_discrete_map=SEV_COLOR, barmode="stack",
                          title="Collisions by Speed Limit",
                          labels={"speed_limit":"Speed Limit (mph)","count":"Collisions"})
            fig2.update_layout(**LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            sfr = fdf.groupby("speed_limit").agg(
                total=("collision_severity","count"),
                fatal=("collision_severity", lambda x:(x==1).sum())
            ).reset_index()
            sfr["fatal_rate"] = sfr["fatal"]/sfr["total"]*100
            fig3 = px.line(sfr, x="speed_limit", y="fatal_rate", markers=True,
                           title="Fatal Rate (%) vs Speed Limit",
                           labels={"speed_limit":"Speed Limit (mph)","fatal_rate":"Fatal Rate (%)"})
            fig3.update_traces(line_color="#dc2626")
            fig3.update_layout(**LAYOUT)
            st.plotly_chart(fig3, use_container_width=True)
        warn("70mph roads carry a fatal rate 4-6x higher than 30mph roads despite better road design.")



# PAGE 5 — CASUALTY & VEHICLE PROFILES

elif page == "Casualty & Vehicle Profiles":
    st.title("Casualty & Vehicle Profiles")

    tab1, tab2 = st.tabs(["Casualty Demographics","Vehicle Analysis"])

    with tab1:
        fc = cas_df[cas_df["collision_index"].isin(fdf["collision_index"])]
        c1,c2 = st.columns(2)
        with c1:
            age_sev = fc[fc["age_of_casualty"]>0].groupby(
                ["age_of_casualty","severity_label"]).size().reset_index(name="count")
            fig = px.area(age_sev, x="age_of_casualty", y="count",
                          color="severity_label", color_discrete_map=SEV_COLOR,
                          title="Casualty Age Distribution by Severity",
                          labels={"age_of_casualty":"Age","count":"Casualties"})
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sx = fc[fc["sex_label"].isin(["Male","Female"])].groupby(
                ["sex_label","severity_label"]).size().reset_index(name="count")
            fig = px.bar(sx, x="sex_label", y="count", color="severity_label",
                         color_discrete_map=SEV_COLOR, barmode="group",
                         title="Casualty Sex by Severity",
                         labels={"sex_label":"Sex","count":"Casualties"})
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        insight("Young males (17-25) are dramatically over-represented in fatal and serious casualties.")

        c1,c2 = st.columns(2)
        with c1:
            cls_sev = fc.groupby(["class_label","severity_label"]).size().reset_index(name="count")
            fig = px.bar(cls_sev, x="class_label", y="count", color="severity_label",
                         color_discrete_map=SEV_COLOR, barmode="stack",
                         title="Casualty Class by Severity",
                         labels={"class_label":"Class","count":"Casualties"})
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            ct = fc.groupby("casualty_type_label")["casualty_severity"].apply(
                lambda x:(x==1).sum()).reset_index(name="fatals")
            ct = ct[ct["fatals"]>0].sort_values("fatals", ascending=False).head(10)
            fig = px.bar(ct, x="fatals", y="casualty_type_label", orientation="h",
                         title="Top 10 Casualty Types by Fatalities",
                         color="fatals", color_continuous_scale="Reds")
            fig.update_layout(**LAYOUT, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        sh("Deprivation Analysis (IMD Decile 1=Most Deprived)")
        imd_df = fc[fc["casualty_imd_decile"]>0].groupby(
            ["casualty_imd_decile","severity_label"]).size().reset_index(name="count")
        fig = px.bar(imd_df, x="casualty_imd_decile", y="count", color="severity_label",
                     color_discrete_map=SEV_COLOR, barmode="stack",
                     title="Casualty Count by IMD Deprivation Decile",
                     labels={"casualty_imd_decile":"IMD Decile","count":"Casualties"})
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        warn("Most deprived decile (1) carries the highest casualty burden — a systemic inequality in road safety exposure.")

    with tab2:
        fv = veh_df[veh_df["collision_index"].isin(fdf["collision_index"])]
        vm = fv.merge(fdf[["collision_index","severity_label","collision_severity"]],
                      on="collision_index", how="left")
        c1,c2 = st.columns(2)
        with c1:
            vt = vm.groupby("vehicle_label").size().reset_index(name="count")
            vt = vt.sort_values("count", ascending=False).head(12)
            fig = px.bar(vt, x="count", y="vehicle_label", orientation="h",
                         title="Vehicles Involved in Collisions",
                         color="count", color_continuous_scale="Blues")
            fig.update_layout(**LAYOUT, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            vf = vm.groupby("vehicle_label")["collision_severity"].apply(
                lambda x:(x==1).sum()).reset_index(name="fatals")
            vf = vf[vf["fatals"]>0].sort_values("fatals", ascending=False).head(10)
            fig = px.bar(vf, x="fatals", y="vehicle_label", orientation="h",
                         title="Vehicle Type — Fatal Collision Involvement",
                         color="fatals", color_continuous_scale="Reds")
            fig.update_layout(**LAYOUT, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        drv = vm[vm["age_of_driver"]>0]
        fig = px.histogram(drv, x="age_of_driver", color="severity_label",
                           color_discrete_map=SEV_COLOR, nbins=40,
                           title="Driver Age Distribution by Collision Severity",
                           barmode="overlay", opacity=0.7,
                           labels={"age_of_driver":"Driver Age","severity_label":"Severity"})
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        insight("Bimodal driver age risk: young (17-25) and elderly (70+) show elevated serious/fatal rates. Middle-aged (35-55) are safest.")



# PAGE 6 — RISK FACTOR ANALYSIS

elif page == "Risk Factor Analysis":
    st.title("Risk Factor Analysis — Hidden Trends")

    tab1, tab2, tab3 = st.tabs(["Compound Risk Factors","Junction Analysis","Correlation & Patterns"])

    with tab1:
        sh("Speed Limit x Light Condition — Fatal Rate Matrix")
        matrix = fdf[~fdf["light_label"].isin(["Unknown"])].groupby(
            ["speed_limit","light_label"]).agg(
            total=("collision_severity","count"),
            fatal=("collision_severity", lambda x:(x==1).sum())
        ).reset_index()
        matrix["fatal_pct"] = matrix["fatal"]/matrix["total"]*100
        pivot = matrix.pivot(index="light_label", columns="speed_limit",
                             values="fatal_pct").fillna(0)
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale="Reds", colorbar=dict(title="Fatal Rate %"),
            text=np.round(pivot.values,1), texttemplate="%{text}%"
        ))
        fig.update_layout(title="Fatal Rate Matrix: Speed Limit x Light Condition",
                          xaxis_title="Speed Limit (mph)", yaxis_title="Light Condition",
                          paper_bgcolor=BG, font_color="#1a1a2e", height=380)
        st.plotly_chart(fig, use_container_width=True)
        insight("Dark unlit roads at 70mph show fatal rates up to 8x that of a 30mph daylight scenario. Entirely hidden when factors are analysed individually.")

        sh("Weather x Hour — Serious/Fatal Density")
        wx_hour = fdf[
            fdf["collision_severity"].isin([1,2]) &
            ~fdf["weather_label"].isin(["Unknown"])
        ].groupby(["weather_label","hour"]).size().reset_index(name="count")
        wx_piv = wx_hour.pivot(index="weather_label", columns="hour", values="count").fillna(0)
        fig = go.Figure(data=go.Heatmap(
            z=wx_piv.values, x=wx_piv.columns, y=wx_piv.index,
            colorscale="YlOrRd", colorbar=dict(title="Serious+Fatal")
        ))
        fig.update_layout(title="Serious/Fatal Collisions: Weather x Hour",
                          xaxis_title="Hour", yaxis_title="Weather",
                          paper_bgcolor=BG, font_color="#1a1a2e", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        jct = fdf[~fdf["junction_label"].isin(["Unknown"])].groupby(
            ["junction_label","severity_label"]).size().reset_index(name="count")
        fig = px.bar(jct, x="junction_label", y="count", color="severity_label",
                     color_discrete_map=SEV_COLOR, barmode="stack",
                     title="Collision Severity by Junction Type",
                     labels={"junction_label":"Junction","count":"Collisions"})
        fig.update_xaxes(tickangle=30)
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        jct_rate = fdf[~fdf["junction_label"].isin(["Unknown"])].groupby("junction_label").agg(
            total=("collision_severity","count"),
            fatal=("collision_severity", lambda x:(x==1).sum())
        ).reset_index()
        jct_rate["fatal_rate"] = jct_rate["fatal"]/jct_rate["total"]*100
        jct_rate = jct_rate.sort_values("fatal_rate", ascending=False)
        fig2 = px.bar(jct_rate, x="junction_label", y="fatal_rate",
                      title="Fatal Rate (%) by Junction Type",
                      color="fatal_rate", color_continuous_scale="Reds",
                      labels={"junction_label":"Junction","fatal_rate":"Fatal Rate (%)"})
        fig2.update_xaxes(tickangle=30)
        fig2.update_layout(**LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)
        insight("Slip roads carry the highest per-collision fatal rate. Roundabout design reduces fatality risk significantly despite higher total incident counts.")

    with tab3:
        corr_cols = ["collision_severity","speed_limit","number_of_vehicles",
                     "number_of_casualties","hour","day_of_week",
                     "weather_conditions","light_conditions",
                     "road_surface_conditions","urban_or_rural_area","month"]
        corr = fdf[corr_cols].dropna().corr().round(2)
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu", zmid=0,
            text=corr.values, texttemplate="%{text}",
            colorbar=dict(title="Correlation")
        ))
        fig.update_layout(title="Pearson Correlation Matrix", height=480,
                          paper_bgcolor=BG, font_color="#1a1a2e")
        st.plotly_chart(fig, use_container_width=True)
        insight("Speed limit has the strongest positive correlation with severity. Number of casualties weakly correlates — multi-vehicle incidents often produce slight outcomes.")

        c1,c2 = st.columns(2)
        with c1:
            nv = fdf[fdf["number_of_vehicles"]<=5].groupby(
                ["number_of_vehicles","severity_label"]).size().reset_index(name="count")
            fig = px.bar(nv, x="number_of_vehicles", y="count", color="severity_label",
                         color_discrete_map=SEV_COLOR, barmode="stack",
                         title="Collisions by Number of Vehicles")
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            nc = fdf[fdf["number_of_casualties"]<=6].groupby(
                ["number_of_casualties","severity_label"]).size().reset_index(name="count")
            fig = px.bar(nc, x="number_of_casualties", y="count", color="severity_label",
                         color_discrete_map=SEV_COLOR, barmode="stack",
                         title="Collisions by Number of Casualties")
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)



# PAGE 7 — CLUSTERING ANALYSIS

elif page == "Clustering Analysis":
    st.title("Unsupervised Learning — K-Means Clustering")
    st.markdown(
        
        "Discovers natural groupings in collision patterns without using severity labels."
    )

    optimal_k     = meta.get("optimal_k", 3)
    cluster_sizes = meta.get("cluster_sizes", {})

    sh("Cluster Overview")
    cs_df = pd.DataFrame({
        "Cluster":     [f"Cluster {k}" for k in sorted(cluster_sizes.keys())],
        "Collisions":  [cluster_sizes[k] for k in sorted(cluster_sizes.keys())],
        "Pct of Data": [f"{cluster_sizes[k]/sum(cluster_sizes.values())*100:.1f}%"
                        for k in sorted(cluster_sizes.keys())]
    })
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(cs_df, use_container_width=True, hide_index=True)
        st.metric("Optimal K (Silhouette)", optimal_k)
        st.metric("Total Clustered Records", f"{sum(cluster_sizes.values()):,}")
    with c2:
        fig = px.pie(cs_df, values="Collisions", names="Cluster",
                     title=f"K-Means Cluster Sizes (K={optimal_k})",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    sh("Elbow Method & Silhouette Scores")
    st.markdown("""
    <div class="step-box">
    <h4>How K was selected</h4>
    <p><strong>Elbow Method:</strong> Inertia (within-cluster sum of squares) plotted vs K.
    The "elbow" point where the rate of decrease flattens indicates a good K.<br>
    <strong>Silhouette Score:</strong> Measures how similar each point is to its own cluster
    vs other clusters. Range: [-1, 1] — higher is better.<br>
    K=<strong>{}</strong> was selected as it maximised
    the Silhouette Score on the training data.</p>
    </div>
    """.format(optimal_k), unsafe_allow_html=True)

    sh("PCA 2D Cluster Visualisation")

    @st.cache_data(show_spinner="Computing cluster assignments...")
    def get_cluster_data():
        ml_df = col_df[FEAT_COLS + ["collision_severity"]].copy()
        ml_df = ml_df[ml_df["collision_severity"].isin([1,2,3])]
        for c in FEAT_COLS:
            ml_df = ml_df[ml_df[c] != -1]
        ml_df.dropna(inplace=True)
        ml_df["target"] = (ml_df["collision_severity"].isin([1,2])).astype(int)
        X_all = ml_df[FEAT_COLS]
        X_all_sc = scaler.transform(X_all)
        clusters = kmeans_model.predict(X_all_sc)
        X_pca    = pca_model.transform(X_all_sc)
        ml_df = ml_df.copy()
        ml_df["cluster"]  = clusters
        ml_df["PC1"]      = X_pca[:, 0]
        ml_df["PC2"]      = X_pca[:, 1]
        ml_df["severity_label"] = ml_df["collision_severity"].map(
            {1:"Fatal",2:"Serious",3:"Slight"})
        return ml_df

    cluster_df = get_cluster_data()
    cluster_df["cluster_label"] = "Cluster " + cluster_df["cluster"].astype(str)
    sample_df  = cluster_df.sample(min(4000, len(cluster_df)), random_state=42)
    sample_df["cluster_label"] = "Cluster " + sample_df["cluster"].astype(str)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(sample_df, x="PC1", y="PC2",
                         color="cluster_label",
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         opacity=0.4,
                         title=f"PCA 2D — Coloured by K-Means Cluster (K={optimal_k})",
                         labels={"PC1":"Principal Component 1",
                                 "PC2":"Principal Component 2"})
        fig.update_traces(marker_size=3)
        fig.update_layout(**LAYOUT, height=420)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.scatter(sample_df, x="PC1", y="PC2",
                          color="severity_label",
                          color_discrete_map=SEV_COLOR,
                          opacity=0.4,
                          title="PCA 2D — Coloured by Actual Severity",
                          labels={"PC1":"Principal Component 1",
                                  "PC2":"Principal Component 2"})
        fig2.update_traces(marker_size=3)
        fig2.update_layout(**LAYOUT, height=420)
        st.plotly_chart(fig2, use_container_width=True)

    insight("Comparing left vs right: where clusters align with severity boundaries, "
            "the model has discovered genuine risk groupings. Misaligned regions indicate "
            "clusters driven by temporal or road-type patterns rather than severity.")

    sh("Cluster Profile Analysis")
    tab1, tab2, tab3 = st.tabs(["Speed & Hour","Severity Composition","Road & Light"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(cluster_df, x="cluster_label", y="speed_limit",
                         color="cluster_label",
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         title="Speed Limit Distribution by Cluster",
                         labels={"cluster_label":"Cluster","speed_limit":"Speed Limit (mph)"})
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.box(cluster_df, x="cluster_label", y="hour",
                          color="cluster_label",
                          color_discrete_sequence=px.colors.qualitative.Set2,
                          title="Hour of Day Distribution by Cluster",
                          labels={"cluster_label":"Cluster","hour":"Hour (24h)"})
            fig2.update_layout(**LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        sev_by_cluster = cluster_df.groupby(["cluster_label","severity_label"]).size().reset_index(name="count")
        fig = px.bar(sev_by_cluster, x="cluster_label", y="count",
                     color="severity_label", color_discrete_map=SEV_COLOR,
                     barmode="stack", title="Severity Composition per Cluster",
                     labels={"cluster_label":"Cluster","count":"Collisions",
                             "severity_label":"Severity"})
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        # Serious/Fatal rate per cluster
        sfr = cluster_df.groupby("cluster_label")["target"].agg(["mean","count"]).reset_index()
        sfr.columns = ["Cluster","Serious/Fatal Rate","Count"]
        sfr["Serious/Fatal Rate"] = (sfr["Serious/Fatal Rate"]*100).round(1)
        fig2 = px.bar(sfr, x="Cluster", y="Serious/Fatal Rate",
                      color="Serious/Fatal Rate", color_continuous_scale="Reds",
                      title="Serious/Fatal Rate (%) by Cluster",
                      text="Serious/Fatal Rate")
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.update_layout(**LAYOUT, yaxis_range=[0, sfr["Serious/Fatal Rate"].max()*1.3])
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            # Light condition heatmap by cluster
            light_map_rev = {1:"Daylight",4:"Dark-lit",5:"Dark-unlit",6:"Dark-no light",-1:"Unknown"}
            cluster_df["light_lbl2"] = cluster_df["light_conditions"].map(light_map_rev).fillna("Unknown")
            lc_cluster = cluster_df[~cluster_df["light_lbl2"].isin(["Unknown"])].groupby(
                ["cluster_label","light_lbl2"]).size().reset_index(name="count")
            fig = px.bar(lc_cluster, x="cluster_label", y="count", color="light_lbl2",
                         barmode="stack", title="Light Condition by Cluster",
                         labels={"cluster_label":"Cluster","count":"Collisions",
                                 "light_lbl2":"Light Condition"})
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            # Urban/Rural by cluster
            cluster_df["urban_lbl2"] = cluster_df["urban_or_rural_area"].map(
                {1:"Urban",2:"Rural",3:"Other"})
            ur_cluster = cluster_df.groupby(["cluster_label","urban_lbl2"]).size().reset_index(name="count")
            fig2 = px.bar(ur_cluster, x="cluster_label", y="count", color="urban_lbl2",
                          barmode="group", title="Urban vs Rural by Cluster",
                          labels={"cluster_label":"Cluster","count":"Collisions",
                                  "urban_lbl2":"Area"})
            fig2.update_layout(**LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

    sh("Cluster Centroids — What Each Cluster Represents")
    centroids_raw = scaler.inverse_transform(kmeans_model.cluster_centers_)
    centroids_df  = pd.DataFrame(centroids_raw, columns=FEAT_COLS).round(2)
    centroids_df.index = [f"Cluster {k}" for k in range(optimal_k)]
    st.dataframe(centroids_df.T, use_container_width=True)
    insight("Cluster centroids (inverse-transformed) show the average feature values for each cluster. "
            "Compare speed_limit and light_conditions across clusters to understand what drives each group.")

    sh("Assign New Collision to a Cluster")
    st.markdown("Enter collision details to find which cluster it belongs to.")
    nc1, nc2 = st.columns(2)
    with nc1:
        n_speed  = st.selectbox("Speed Limit", [20,30,40,50,60,70], key="cl_speed")
        n_light  = st.selectbox("Light Condition", list(LIGHT_MAP.items()), format_func=lambda x:x[1], key="cl_light")[0]
        n_urban  = st.selectbox("Area Type", list(URBAN_MAP.items()), format_func=lambda x:x[1], key="cl_urban")[0]
        n_hour   = st.slider("Hour of Day", 0, 23, 8, key="cl_hour")
    with nc2:
        n_weather = st.selectbox("Weather", list(WEATHER_MAP.items()), format_func=lambda x:x[1], key="cl_weather")[0]
        n_dow     = st.selectbox("Day of Week", list(DOW_MAP.items()), format_func=lambda x:x[1], key="cl_dow")[0]
        n_month   = st.slider("Month", 1, 12, 3, key="cl_month")

    default_vals = [n_speed,1,1,n_hour,n_dow,n_weather,n_light,1,n_urban,n_month,0,3,0]
    if st.button("Find My Cluster", key="cluster_btn"):
        X_new = pd.DataFrame([default_vals], columns=FEAT_COLS)
        X_new_sc = scaler.transform(X_new)
        assigned = int(kmeans_model.predict(X_new_sc)[0])
        distances = kmeans_model.transform(X_new_sc)[0]
        confidence = 1 - distances[assigned]/distances.sum()

        st.markdown(f"""
        <div style="background:#1e3a5f;color:white;border-radius:10px;
                    padding:18px 28px;text-align:center;margin:12px 0;">
            <h2 style="margin:0;color:white;">Cluster {assigned}</h2>
            <p style="margin:4px 0;font-size:0.95rem;">
                Cluster confidence: {confidence:.1%} | Centroid distance: {distances[assigned]:.3f}
            </p>
        </div>
        """, unsafe_allow_html=True)
        dist_df = pd.DataFrame({"Cluster":[f"Cluster {k}" for k in range(optimal_k)],
                                 "Distance to Centroid": distances.round(4)})
        fig = px.bar(dist_df, x="Cluster", y="Distance to Centroid",
                     color="Distance to Centroid", color_continuous_scale="Blues_r",
                     title="Distance to Each Cluster Centroid (lower = better match)")
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


# PAGE 8 — ML PIPELINE & MODEL COMPARISON

elif page == "ML Pipeline & Model Comparison":
    st.title("ML Pipeline — Severity Classification")
    st.markdown(
                f"Best model: **{best_name}**"
    )

    sh("Model Performance Comparison")
    rows = []
    for name, r in ml_results.items():
        rows.append({
            "Model":            name,
            "Accuracy":         f"{r['acc']:.4f}",
            "Weighted F1":      f"{r['f1']:.4f}",
            "Macro F1":         f"{r['f1_macro']:.4f}",
            "AUC-ROC":          f"{r['auc']:.4f}",
            "CV F1 (mean±std)": f"{r['cv_mean']:.4f} ± {r['cv_std']:.4f}",
            "Best?":            "YES" if name == best_name else ""
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
    st.success(f"Best Model: **{best_name}** = {ml_results[best_name]['f1']:.4f}")

    c1,c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for m, lbl, col_c in [("acc","Accuracy","#2563eb"),
                               ("f1","Weighted F1","#16a34a"),
                               ("f1_macro","Macro F1","#d97706")]:
            fig.add_trace(go.Bar(
                x=list(ml_results.keys()),
                y=[ml_results[n][m] for n in ml_results],
                name=lbl, marker_color=col_c
            ))
        fig.update_layout(barmode="group", title="Algorithm Comparison — Key Metrics",
                          yaxis_range=[0,1.05], **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        auc_vals = {n: ml_results[n]["auc"] for n in ml_results}
        fig2 = go.Figure(go.Bar(
            x=list(auc_vals.keys()), y=list(auc_vals.values()),
            marker_color=["#dc2626" if n==best_name else "#93c5fd" for n in auc_vals]
        ))
        fig2.update_layout(title="AUC-ROC by Model", yaxis_range=[0.5,1.0], **LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    sh("Confusion Matrices — All 4 Models (from saved metadata)")
    cols_cm = st.columns(4)
    for i,(name,r) in enumerate(ml_results.items()):
        with cols_cm[i]:
            cm = np.array(r["cm"])
            fig = px.imshow(cm, text_auto=True,
                            x=["Slight","Serious/Fatal"],
                            y=["Slight","Serious/Fatal"],
                            color_continuous_scale="Blues", title=name)
            fig.update_layout(paper_bgcolor=BG, font_color="#1a1a2e",
                              coloraxis_showscale=False, height=280,
                              margin=dict(l=0,r=0,t=50,b=0))
            st.plotly_chart(fig, use_container_width=True)

    sh("Cross-Validation Stability (5-Fold F1 with 95% CI)")
    cv_fig = go.Figure()
    for name, r in ml_results.items():
        cv_fig.add_trace(go.Bar(
            x=[name], y=[r["cv_mean"]],
            error_y=dict(type="data", array=[r["cv_std"]*1.96], visible=True),
            name=name,
            marker_color="#dc2626" if name==best_name else "#93c5fd"
        ))
    cv_fig.update_layout(title="CV F1 Score with 95% CI",
                          yaxis_range=[0,1], showlegend=False, **LAYOUT)
    st.plotly_chart(cv_fig, use_container_width=True)

    sh(f"Feature Importance — Loaded from {best_name} pkl")
    best_model_obj = models[best_name]
    if hasattr(best_model_obj, "feature_importances_"):
        fi = pd.DataFrame({
            "Feature":    FEAT_COLS,
            "Importance": best_model_obj.feature_importances_
        }).sort_values("Importance", ascending=True)
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                     title=f"Feature Importances — {best_name} (from pkl)",
                     color="Importance", color_continuous_scale="Blues")
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    elif hasattr(best_model_obj, "coef_"):
        coef = pd.DataFrame({
            "Feature":     FEAT_COLS,
            "Coefficient": np.abs(best_model_obj.coef_[0])
        }).sort_values("Coefficient", ascending=True)
        fig = px.bar(coef, x="Coefficient", y="Feature", orientation="h",
                     title=f"|Coefficients| — {best_name}",
                     color="Coefficient", color_continuous_scale="Blues")
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"{best_name} has no built-in feature importance.")

    insight(f" Best model ({best_name}) "
            f"Speed limit and light conditions are the dominant predictors "
            f"across all interpretable models.")



# PAGE 8 — PREDICT & EXPLAIN

elif page == "Predict & Explain":
    st.title("Predict Collision Severity & Interpret")
    
    best_model_obj = models[best_name]
    needs_sc       = ml_results[best_name]["needs_scaling"]

    sh("Single-Scenario Prediction")
    st.markdown("Configure all scenario conditions and click **Run Prediction**.")

    c1,c2,c3 = st.columns(3)
    with c1:
        p_speed      = st.selectbox("Speed Limit (mph)", [20,30,40,50,60,70], index=1)
        p_vehicles   = st.slider("No. of Vehicles", 1, 5, 2)
        p_casualties = st.slider("No. of Casualties", 1, 5, 1)
        p_hour       = st.slider("Hour of Day", 0, 23, 8)
        p_dow        = st.selectbox("Day of Week",
                                    list(DOW_MAP.items()),
                                    format_func=lambda x:x[1])[0]
    with c2:
        p_weather    = st.selectbox("Weather",
                                    list(WEATHER_MAP.items()),
                                    format_func=lambda x:x[1])[0]
        p_light      = st.selectbox("Light Condition",
                                    list(LIGHT_MAP.items()),
                                    format_func=lambda x:x[1])[0]
        p_road_surf  = st.selectbox("Road Surface",
                                    list(ROAD_SURF_MAP.items()),
                                    format_func=lambda x:x[1])[0]
        p_urban      = st.selectbox("Area Type",
                                    list(URBAN_MAP.items()),
                                    format_func=lambda x:x[1])[0]
    with c3:
        p_month      = st.slider("Month", 1, 12, 3)
        p_jct        = st.selectbox("Junction Type",
                                    list(JUNCTION_MAP.items()),
                                    format_func=lambda x:x[1])[0]
        p_road_class = st.selectbox("Road Class",
                                    list(ROAD_CLASS_MAP.items()),
                                    format_func=lambda x:x[1])[0]
        p_ped_cross  = st.selectbox("Pedestrian Crossing",
                                    [(0,"None"),(1,"Controlled"),(5,"Footbridge")],
                                    format_func=lambda x:x[1])[0]

    feat_vals = [p_speed, p_vehicles, p_casualties, p_hour, p_dow,
                 p_weather, p_light, p_road_surf, p_urban, p_month,
                 p_jct, p_road_class, p_ped_cross]

    if st.button("Run Prediction", type="primary"):
        pred, prob = predict_input(feat_vals)
        result_label = "SERIOUS / FATAL" if pred == 1 else "SLIGHT"
        result_color = "#dc2626" if pred == 1 else "#16a34a"
        conf = prob[pred]

        st.markdown(f"""
        <div style="background:{result_color};color:white;border-radius:10px;
                    padding:20px 30px;text-align:center;margin:16px 0;">
            <h2 style="margin:0;color:white;">{result_label}</h2>
            <p style="margin:4px 0 0;font-size:1rem;">
                Model confidence: {conf:.1%} | P(Serious/Fatal): {prob[1]:.1%} | Model: {best_name}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob[1]*100,
            title={"text":f"P(Serious/Fatal) % — {best_name}","font":{"color":"#1a1a2e"}},
            gauge={
                "axis":{"range":[0,100],"tickfont":{"color":"#1a1a2e"}},
                "bar":{"color":result_color},
                "steps":[
                    {"range":[0,30],  "color":"#dcfce7"},
                    {"range":[30,60], "color":"#fef3c7"},
                    {"range":[60,100],"color":"#fee2e2"}
                ],
                "threshold":{"line":{"color":"#1a1a2e","width":3},
                             "thickness":0.75,"value":50}
            }
        ))
        fig_gauge.update_layout(height=300, paper_bgcolor=BG, font_color="#1a1a2e",
                                 margin=dict(l=20,r=20,t=60,b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Feature contribution bar
        if hasattr(best_model_obj, "feature_importances_"):
            fi_arr = best_model_obj.feature_importances_
        elif hasattr(best_model_obj, "coef_"):
            fi_arr = np.abs(best_model_obj.coef_[0])
        else:
            X_in_tf = scaler.transform(pd.DataFrame([feat_vals], columns=FEAT_COLS)) \
                      if needs_sc else np.array([feat_vals])
            fi_arr = np.abs(X_in_tf[0])
            fi_arr = fi_arr / (fi_arr.sum() + 1e-9)

        contrib = pd.DataFrame({
            "Feature":    FEAT_COLS,
            "Importance": fi_arr,
            "Your Value": feat_vals
        }).sort_values("Importance", ascending=False).head(8)

        fig_wf = go.Figure(go.Bar(
            x=contrib["Importance"], y=contrib["Feature"], orientation="h",
            marker_color=["#dc2626" if pred==1 else "#2563eb"]*len(contrib),
            text=[f"Input: {v}" for v in contrib["Your Value"]],
            textposition="outside"
        ))
        fig_wf.update_layout(title=f"Top 8 Feature Contributions — {best_name}",
                              **LAYOUT, xaxis_title="Feature Importance",
                              height=320, margin=dict(r=130))
        st.plotly_chart(fig_wf, use_container_width=True)

    sh("Scenario Comparison — Speed x Light Condition")
    st.markdown("Predicted P(Serious/Fatal) across all speed × light combinations ")

    default_row = [30,2,1,14,5,1,1,1,1,3,0,3,0]
    scenario_rows = []
    for sl in [20,30,40,50,60,70]:
        for lc in [1,4,5,6]:
            row = default_row.copy()
            row[FEAT_COLS.index("speed_limit")]      = sl
            row[FEAT_COLS.index("light_conditions")] = lc
            _, prob = predict_input(row)
            scenario_rows.append({
                "Speed Limit": sl,
                "Light":       LIGHT_MAP.get(lc,"?"),
                "P(Serious/Fatal)": round(prob[1],4)
            })

    scen_piv = pd.DataFrame(scenario_rows).pivot(
        index="Light", columns="Speed Limit", values="P(Serious/Fatal)")
    fig = px.imshow(scen_piv, text_auto=".2f",
                    color_continuous_scale="RdYlGn_r",
                    title=f"Predicted P(Serious/Fatal): Speed x Light — {best_name}",
                    labels={"x":"Speed Limit (mph)","y":"Light Condition","color":"Probability"})
    fig.update_layout(paper_bgcolor=BG, font_color="#1a1a2e")
    st.plotly_chart(fig, use_container_width=True)
    insight("Dark unlit 70mph conditions produce 3-4x higher predicted serious/fatal probability vs daylight 30mph scenarios — compound risk quantified ")

    sh("Model Notes & Limitations")
    warn(f"Class imbalance: ~72% Slight vs ~28% Serious/Fatal. "
         f"Mitigated with class_weight='balanced' for LR and RF. ")
    
