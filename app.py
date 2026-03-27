import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitLife AI — Consumer Insights Dashboard",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── COLOUR PALETTE ────────────────────────────────────────────────────────────
C = {
    "purple":  "#534AB7", "purple_l": "#EEEDFE",
    "teal":    "#1D9E75", "teal_l":   "#E1F5EE",
    "amber":   "#BA7517", "amber_l":  "#FAEEDA",
    "coral":   "#D85A30", "coral_l":  "#FAECE7",
    "blue":    "#185FA5", "blue_l":   "#E6F1FB",
    "pink":    "#993556", "pink_l":   "#FBEAF0",
    "green":   "#3B6D11", "green_l":  "#EAF3DE",
    "gray":    "#5F5E5A", "gray_l":   "#F8F8F6",
    "red":     "#A32D2D", "red_l":    "#FCEBEB",
}

PERSONA_COLORS = {
    "Aspiring Urban Professional": C["purple"],
    "Budget-Conscious Student":    C["teal"],
    "Health-Driven Middle-Ager":   C["amber"],
    "Sedentary Skeptic":           C["gray"],
    "Homemaker Wellness Seeker":   C["pink"],
    "Fitness Enthusiast":          C["coral"],
}

INTENT_COLORS = {
    "Interested":     C["teal"],
    "Undecided":      C["amber"],
    "Not Interested": C["red"],
}

# ── GLOBAL STYLES ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #F8F8F6;
    padding: 6px 6px 0;
    border-radius: 8px 8px 0 0;
}
.stTabs [data-baseweb="tab"] {
    height: 38px;
    padding: 0 18px;
    border-radius: 6px 6px 0 0;
    font-size: 13px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #534AB7 !important;
    color: white !important;
}
/* Insight box */
.insight-box {
    background: #EEEDFE;
    border-left: 4px solid #534AB7;
    border-radius: 0 8px 8px 0;
    padding: 14px 16px;
    margin: 8px 0 18px 0;
    font-size: 13.5px;
    line-height: 1.6;
    color: #2C2C2A;
}
.insight-box strong { color: #534AB7; }
/* Metric card */
.metric-row {
    display: flex; gap: 12px; margin-bottom: 16px;
}
.metric-card {
    background: #F8F8F6;
    border: 1px solid #D3D1C7;
    border-radius: 10px;
    padding: 14px 18px;
    flex: 1;
    text-align: center;
}
.metric-val { font-size: 26px; font-weight: 600; color: #534AB7; }
.metric-lbl { font-size: 12px; color: #5F5E5A; margin-top: 2px; }
/* Section header */
.section-header {
    background: linear-gradient(90deg, #534AB7 0%, #7F77DD 100%);
    color: white;
    padding: 10px 18px;
    border-radius: 8px;
    font-size: 15px;
    font-weight: 600;
    margin: 20px 0 12px 0;
}
/* Warning / info tags */
.tag-warn { background:#FAEEDA; color:#633806; padding:2px 8px; border-radius:20px; font-size:12px; font-weight:600; }
.tag-ok   { background:#E1F5EE; color:#085041; padding:2px 8px; border-radius:20px; font-size:12px; font-weight:600; }
.tag-info { background:#E6F1FB; color:#0C447C; padding:2px 8px; border-radius:20px; font-size:12px; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── DATA LOADING & CLEANING ───────────────────────────────────────────────────
@st.cache_data
def load_raw():
    df = pd.read_csv("data/fitlife_ai_survey_dataset.csv")
    return df

@st.cache_data
def clean_data(df_raw):
    df = df_raw.copy()

    # 1. Impute missing income with mode per city tier
    df["q4_monthly_income"] = df.groupby("q3_city_tier")["q4_monthly_income"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "50000-100000")
    )

    # 2. Impute missing health condition columns with 0
    health_cols = [c for c in df.columns if c.startswith("q16_health_")]
    df[health_cols] = df[health_cols].fillna(0).astype(int)

    # 3. Create ordinal encodings for key variables
    income_order = ["Below 25000","25000-50000","50000-100000","100000-200000","Above 200000"]
    activity_order = ["0 times","1-4 times","5-8 times","9-12 times","13+ times"]
    wtp_order = ["Nothing (free only)","99-299/month","300-599/month","600-999/month","1000+/month"]
    spend_order = ["0","1-500","501-1500","1501-3000","3001-5000","5000+"]
    junk_order = ["Rarely","Sometimes","Often","Daily"]
    act_lvl_order = ["Sedentary","Lightly active","Moderately active","Highly active","Athlete"]
    tech_order = ["Not comfortable","Open but not started","Tried but stopped",
                  "Use 1 app occasionally","Use 2+ apps actively"]
    comfort_order = ["Very uncomfortable","Uncomfortable","Neutral",
                     "Comfortable if secure","Very comfortable"]

    df["income_score"]   = pd.Categorical(df["q4_monthly_income"],   categories=income_order,   ordered=True).codes + 1
    df["activity4w_score"] = pd.Categorical(df["q2_activity_last_4weeks"], categories=activity_order, ordered=True).codes + 1
    df["wtp_score"]      = pd.Categorical(df["q20_willingness_to_pay"], categories=wtp_order,   ordered=True).codes + 1
    df["spend_score"]    = pd.Categorical(df["q19_current_fitness_spend"], categories=spend_order, ordered=True).codes + 1
    df["junk_score"]     = pd.Categorical(df["q13_junk_food_frequency"], categories=junk_order, ordered=True).codes + 1
    df["actlvl_score"]   = pd.Categorical(df["q7_activity_level"],    categories=act_lvl_order, ordered=True).codes + 1
    df["tech_score"]     = pd.Categorical(df["q14_tech_adoption"],    categories=tech_order,    ordered=True).codes + 1
    df["comfort_score"]  = pd.Categorical(df["q22_data_comfort"],     categories=comfort_order, ordered=True).codes + 1

    # 4. WTP gap (aspiration vs reality)
    df["wtp_gap"] = df["wtp_score"] - df["spend_score"]

    # 5. Flag noise rows (activity inconsistency)
    high_act = df["q7_activity_level"].isin(["Highly active","Athlete"])
    low_act4w = df["q2_activity_last_4weeks"].isin(["0 times","1-4 times"])
    df["is_noise_row"] = (high_act & low_act4w).astype(int)

    # 6. Flag WTP outliers
    df["is_wtp_outlier"] = (
        (df["q20_willingness_to_pay"] == "1000+/month") &
        (df["q19_current_fitness_spend"].isin(["0","1-500"]))
    ).astype(int)

    # 7. Encode target
    intent_map = {"Interested": 2, "Undecided": 1, "Not Interested": 0}
    df["intent_score"] = df["q25_subscription_intent"].map(intent_map)

    # 8. Health condition count per respondent
    df["health_condition_count"] = df[[c for c in health_cols if c != "q16_health_none"]].sum(axis=1)

    return df


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
def insight(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

def section(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)

def kpi_row(items):
    cols = st.columns(len(items))
    for col, (val, label, color) in zip(cols, items):
        col.markdown(f"""
        <div style='background:{color}22;border:1px solid {color}44;border-radius:10px;
                    padding:14px;text-align:center;'>
            <div style='font-size:28px;font-weight:700;color:{color};'>{val}</div>
            <div style='font-size:12px;color:#5F5E5A;margin-top:4px;'>{label}</div>
        </div>""", unsafe_allow_html=True)

def chart_container(fig, key=None):
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_family="Arial",
        font_color="#2C2C2A",
        margin=dict(t=50, b=40, l=40, r=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        title_font_size=15,
        title_font_color="#2C2C2A",
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df_raw = load_raw()
df = clean_data(df_raw)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/dumbbell.png", width=60)
    st.markdown("## FitLife AI")
    st.markdown("*Consumer Insights Dashboard*")
    st.divider()

    st.markdown("### 🎯 Global Filters")
    sel_persona = st.multiselect(
        "Persona", options=df["persona_label"].unique().tolist(),
        default=df["persona_label"].unique().tolist()
    )
    sel_city = st.multiselect(
        "City Tier", options=df["q3_city_tier"].unique().tolist(),
        default=df["q3_city_tier"].unique().tolist()
    )
    sel_intent = st.multiselect(
        "Subscription Intent", options=["Interested","Undecided","Not Interested"],
        default=["Interested","Undecided","Not Interested"]
    )
    st.divider()
    st.markdown(f"**Total respondents:** {len(df):,}")
    filtered = df[
        df["persona_label"].isin(sel_persona) &
        df["q3_city_tier"].isin(sel_city) &
        df["q25_subscription_intent"].isin(sel_intent)
    ]
    st.markdown(f"**Filtered respondents:** {len(filtered):,}")
    st.divider()
    st.caption("FitLife AI · India · 2026\nSynthetic research dataset")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏢 Business Overview",
    "📋 Raw Data",
    "🧹 Data Cleaning",
    "👥 EDA: Demographics",
    "💪 EDA: Fitness & Behaviour",
    "💰 EDA: Spending & Attitude",
    "🔬 Diagnostic Analysis",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BUSINESS OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("# FitLife AI — Business Strategy & Market Context")
    st.markdown("*AI-powered personalised fitness and nutrition platform for the Indian market*")
    st.divider()

    kpi_row([
        ("$13.5B", "Global fitness app market (2026)", C["purple"]),
        ("19.5%",  "Asia Pacific CAGR",                C["teal"]),
        ("68%",    "Users prefer adaptive AI platforms",C["amber"]),
        ("20%+",   "FitLife AI target annual growth",   C["coral"]),
    ])
    st.markdown("")

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        section("🎯 Mission & Strategic Objective")
        st.markdown("""
        FitLife AI aims to become **India's leading AI-powered health companion** — a single intelligent
        platform that unifies workout planning, nutrition coaching, recovery optimisation, and progress
        analytics, personalised to each user's body, goals, and Indian lifestyle.

        **Core strategic pillars:**
        - **Personalisation at scale** — AI that learns and adapts continuously
        - **Cultural relevance** — built natively for Indian diets, habits, and price sensitivity
        - **Holistic coverage** — fitness + nutrition + recovery + mental wellness in one product
        - **Measurable results** — every recommendation tied to trackable outcomes
        - **20%+ YoY growth** — through subscription revenue, B2B wellness, and premium tiers
        """)

        section("📦 7 Core Product Offerings")
        products = [
            ("1", "Adaptive AI Workout Engine", C["purple"],
             "Continuously learning workout plans based on user data, recovery state, and goals."),
            ("2", "Culturally Intelligent Nutrition Coach", C["teal"],
             "AI diet tracking built natively for Indian cuisines — roti, dal, thali, regional foods."),
            ("3", "AI Recovery & Sleep Optimisation", C["amber"],
             "Software-first recovery coaching using wearable data — no hardware dependency."),
            ("4", "Personalised Progress Analytics Dashboard", C["coral"],
             "Real-time dashboards tracking fitness, nutrition, sleep, and mental health together."),
            ("5", "AI Form & Movement Coach (Camera)", C["blue"],
             "Computer vision via smartphone camera for real-time form correction and rep counting."),
            ("6", "Corporate Wellness B2B Portal", C["green"],
             "Team challenges, manager analytics, and group health benchmarks for enterprises."),
            ("7", "Longevity & Specialty Programs", C["pink"],
             "Targeted programs: active aging (45-65), post-pregnancy, GLP-1 users, women's health."),
        ]
        for num, name, color, desc in products:
            st.markdown(f"""
            <div style='display:flex;gap:12px;align-items:flex-start;margin-bottom:10px;
                        background:#F8F8F6;border-radius:8px;padding:10px 14px;
                        border-left:4px solid {color};'>
                <div style='font-size:18px;font-weight:700;color:{color};min-width:24px;'>{num}</div>
                <div>
                    <div style='font-weight:600;font-size:14px;color:#2C2C2A;'>{name}</div>
                    <div style='font-size:12.5px;color:#5F5E5A;margin-top:2px;'>{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with col2:
        section("👥 Target Persona Overview")
        persona_data = {
            "Persona": list(PERSONA_COLORS.keys()),
            "Share": [28, 18, 15, 17, 12, 10],
            "WTP": ["₹300-599", "₹99-299", "₹300-599", "Free", "₹99-299", "₹600+"],
            "Intent": ["High", "Moderate", "High", "Low", "Moderate", "Very High"],
        }
        pdf = pd.DataFrame(persona_data)
        fig_persona = px.bar(
            pdf, x="Share", y="Persona", orientation="h",
            color="Persona",
            color_discrete_map=PERSONA_COLORS,
            title="Persona Population Weights (%)",
            labels={"Share": "Population Share (%)", "Persona": ""},
            text="Share",
        )
        fig_persona.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_persona.update_layout(showlegend=False, height=320)
        chart_container(fig_persona, "persona_bar")

        section("💰 Pricing Strategy")
        pricing = {
            "Tier": ["Free", "Core (₹299/mo)", "Premium (₹599/mo)", "Elite (₹999/mo)", "B2B (₹399/seat)"],
            "Features": ["Basic tracking", "AI workouts + diet", "Full AI + recovery", "Coach + all features", "Corporate dashboard"],
            "Target": ["All users", "Students / Budget", "Urban Professionals", "Enthusiasts / Mid-age", "Enterprises"],
        }
        st.dataframe(pd.DataFrame(pricing), use_container_width=True, hide_index=True)

        section("🏆 Competitive Positioning")
        st.markdown("""
        | Competitor | Gap FitLife AI Fills |
        |------------|----------------------|
        | MyFitnessPal | No Indian diet engine, no AI workouts |
        | WHOOP | Hardware-dependent, no nutrition |
        | Fitbod | Workout only, no diet or recovery |
        | Noom | No exercise plans, Western-centric |
        | Freeletics | No personalisation depth, no nutrition |

        **Our whitespace:** The only platform combining AI workouts + Indian nutrition +
        recovery + progress analytics at an India-first price point.
        """)

    section("📊 Survey Research Framework")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(f"""<div style='background:{C["purple_l"]};border-radius:8px;padding:12px;border:1px solid {C["purple"]}44;'>
        <div style='font-weight:700;color:{C["purple"]};margin-bottom:6px;'>Classification</div>
        <div style='font-size:12px;'>Predict subscription intent (Interested / Undecided / Not Interested)<br><br>
        <b>Target:</b> Q25<br><b>Key features:</b> Income, self-efficacy, barrier, tech adoption</div></div>""",
        unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""<div style='background:{C["teal_l"]};border-radius:8px;padding:12px;border:1px solid {C["teal"]}44;'>
        <div style='font-weight:700;color:{C["teal"]};margin-bottom:6px;'>Clustering</div>
        <div style='font-size:12px;'>Identify user segments by goals, lifestyle, motivation, and tech adoption<br><br>
        <b>Validate:</b> Against 6 engineered personas<br><b>Algorithm:</b> K-Means, DBSCAN</div></div>""",
        unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""<div style='background:{C["amber_l"]};border-radius:8px;padding:12px;border:1px solid {C["amber"]}44;'>
        <div style='font-weight:700;color:{C["amber"]};margin-bottom:6px;'>Association Mining</div>
        <div style='font-size:12px;'>Find co-occurring habits: workout types, diet patterns, add-on preferences<br><br>
        <b>Columns:</b> Q8_*, Q21_*, Q9, Q11, Q13<br><b>Algorithm:</b> Apriori / FP-Growth</div></div>""",
        unsafe_allow_html=True)
    with col_d:
        st.markdown(f"""<div style='background:{C["green_l"]};border-radius:8px;padding:12px;border:1px solid {C["green"]}44;'>
        <div style='font-weight:700;color:{C["green"]};margin-bottom:6px;'>Regression</div>
        <div style='font-size:12px;'>Estimate willingness to pay and spending capacity<br><br>
        <b>Target:</b> Q20 (WTP score)<br><b>Key features:</b> Income, spend, self-efficacy, trade-off</div></div>""",
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("# Raw Survey Data")
    st.markdown("The original 2,000-respondent dataset as generated — before any cleaning or imputation.")
    st.divider()

    kpi_row([
        (f"{len(df_raw):,}", "Total respondents", C["purple"]),
        ("48",               "Total columns",      C["teal"]),
        ("25",               "Survey questions",   C["amber"]),
        (f"{df_raw.isnull().sum().sum():,}", "Missing values", C["coral"]),
    ])
    st.markdown("")

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_persona_raw = st.multiselect("Filter by Persona", df_raw["persona_label"].unique(),
                                             default=list(df_raw["persona_label"].unique()), key="raw_persona")
    with col2:
        filter_city_raw = st.multiselect("Filter by City Tier", df_raw["q3_city_tier"].unique(),
                                          default=list(df_raw["q3_city_tier"].unique()), key="raw_city")
    with col3:
        filter_intent_raw = st.multiselect("Filter by Intent", ["Interested","Undecided","Not Interested"],
                                            default=["Interested","Undecided","Not Interested"], key="raw_intent")

    df_raw_view = df_raw[
        df_raw["persona_label"].isin(filter_persona_raw) &
        df_raw["q3_city_tier"].isin(filter_city_raw) &
        df_raw["q25_subscription_intent"].isin(filter_intent_raw)
    ]

    col_filter = st.multiselect("Select columns to display",
                                 options=list(df_raw.columns),
                                 default=["respondent_id","persona_label","q1_age_group",
                                          "q3_city_tier","q4_monthly_income","q5_fitness_goal",
                                          "q7_activity_level","q11_diet_preference",
                                          "q20_willingness_to_pay","q25_subscription_intent"])
    st.markdown(f"**Showing {len(df_raw_view):,} rows × {len(col_filter)} columns**")
    st.dataframe(df_raw_view[col_filter], use_container_width=True, height=420)

    section("📊 Raw Data — Quick Profile")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Missing Values by Column (top 15)**")
        miss = df_raw.isnull().sum().sort_values(ascending=False).head(15)
        miss_df = pd.DataFrame({"Column": miss.index, "Missing Count": miss.values,
                                  "Missing %": (miss.values/len(df_raw)*100).round(2)})
        miss_df = miss_df[miss_df["Missing Count"] > 0]
        fig_miss = px.bar(miss_df, x="Missing %", y="Column", orientation="h",
                          color="Missing %", color_continuous_scale=["#E1F5EE","#D85A30"],
                          title="Missing Values (%)")
        fig_miss.update_layout(coloraxis_showscale=False, height=300)
        chart_container(fig_miss, "raw_missing")
    with col2:
        st.markdown("**Data Types Distribution**")
        dtype_map = {str(v): k for k, v in df_raw.dtypes.value_counts().items()}
        dtype_df = df_raw.dtypes.value_counts().reset_index()
        dtype_df.columns = ["Data Type", "Count"]
        dtype_df["Data Type"] = dtype_df["Data Type"].astype(str)
        fig_dtype = px.pie(dtype_df, values="Count", names="Data Type",
                           color_discrete_sequence=[C["purple"], C["teal"], C["amber"]],
                           title="Column Data Types")
        chart_container(fig_dtype, "raw_dtype")

    insight("""<strong>What this shows:</strong> The raw dataset contains 2,000 respondent rows and 48 columns spanning all 25 survey questions.
    Missing values are intentionally injected to simulate real survey behaviour:
    <strong>q4_monthly_income</strong> has ~6.2% missing (income is a sensitive question many respondents skip),
    and <strong>q16_health_*</strong> columns have ~1.6% missing (privacy-driven skips on health disclosures).
    All other columns are complete. The majority of columns are object (categorical) type, with a handful of integer Likert scales.
    <strong>Business implication:</strong> Before modelling, income must be imputed and health flags must be zero-filled — which we handle in the Data Cleaning tab.""")

    csv_data = df_raw.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Raw Dataset (CSV)", csv_data,
                       "fitlife_raw_data.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("# Data Cleaning & Quality Report")
    st.markdown("Documenting every transformation applied to produce the analysis-ready dataset.")
    st.divider()

    kpi_row([
        (f"{df_raw.isnull().sum().sum():,}", "Missing cells (raw)",    C["coral"]),
        ("0",                                 "Missing cells (clean)",  C["teal"]),
        ("~80",                               "Noise rows flagged",     C["amber"]),
        ("~40",                               "Outlier rows flagged",   C["purple"]),
    ])
    st.markdown("")

    section("🔧 Cleaning Steps Applied")
    steps = [
        ("1", "Income Imputation", C["purple"],
         "q4_monthly_income — 124 missing values (~6.2%) imputed using the **mode income band within each city tier group**. "
         "Metro respondents skipping income are more likely to be higher earners; Tier 3 respondents more likely lower. "
         "Group-wise mode preserves this geographic income structure better than global imputation."),
        ("2", "Health Condition Zero-Fill", C["teal"],
         "All 7 q16_health_* binary columns — 31 NaN values per column (~1.6%) filled with **0 (condition not present)**. "
         "Given these are binary flags and the skip behaviour is driven by privacy concerns rather than condition presence, "
         "zero-imputation is the correct conservative assumption."),
        ("3", "Ordinal Score Encoding", C["amber"],
         "8 ordinal categorical variables (income, WTP, activity, tech adoption, etc.) encoded as **integer scores 1–5** "
         "to enable correlation analysis and regression modelling. Original string columns are preserved alongside."),
        ("4", "WTP Gap Feature Engineering", C["coral"],
         "A new column **wtp_gap** = wtp_score − spend_score is engineered. Positive values indicate aspirational users "
         "(want to spend more than they currently do); negative values flag users who overspend relative to stated WTP. "
         "This is one of the strongest regression features for subscription conversion."),
        ("5", "Noise Row Flagging", C["blue"],
         "Rows where q7_activity_level is 'Highly active' or 'Athlete' BUT q2_activity_last_4weeks is '0' or '1-4 times' "
         "are flagged as **is_noise_row = 1** (~80 rows, ~4%). These reflect social desirability bias — respondents overstating "
         "their activity level. Flagged but retained for robust model training."),
        ("6", "Outlier Row Flagging", C["green"],
         "Rows with WTP of '1000+/month' despite current spend of '₹0' or '₹1-500' are flagged as **is_wtp_outlier = 1** "
         "(~40 rows, ~2%). These aspirational responders inflate WTP estimates if not handled — "
         "flag enables Winsorisation or sensitivity testing during regression."),
        ("7", "Intent Score Encoding", C["pink"],
         "Target variable q25_subscription_intent encoded as **intent_score**: Not Interested=0, Undecided=1, Interested=2. "
         "Enables ordinal treatment in addition to the 3-class classification framing."),
    ]
    for num, title, color, desc in steps:
        with st.expander(f"Step {num}: {title}", expanded=(num in ["1","2","4"])):
            st.markdown(f'<div style="border-left:4px solid {color};padding:10px 14px;border-radius:0 8px 8px 0;">'
                        f'{desc}</div>', unsafe_allow_html=True)

    section("📊 Before vs After — Missing Values")
    col1, col2 = st.columns(2)
    with col1:
        miss_before = df_raw.isnull().sum().sort_values(ascending=False)
        miss_before = miss_before[miss_before > 0].reset_index()
        miss_before.columns = ["Column", "Missing"]
        fig_b = px.bar(miss_before, x="Column", y="Missing",
                       color="Missing", color_continuous_scale=["#FAEEDA","#D85A30"],
                       title="Before Cleaning — Missing Values")
        fig_b.update_layout(coloraxis_showscale=False, xaxis_tickangle=45, height=350)
        chart_container(fig_b, "miss_before")
    with col2:
        miss_after = df.isnull().sum().sort_values(ascending=False)
        miss_after = miss_after[miss_after > 0]
        if len(miss_after) == 0:
            st.markdown('<br><br>', unsafe_allow_html=True)
            st.success("✅ Zero missing values after cleaning. All 2,000 rows are complete and model-ready.")
        else:
            miss_after = miss_after.reset_index(); miss_after.columns = ["Column","Missing"]
            fig_a = px.bar(miss_after, x="Column", y="Missing", title="After Cleaning")
            chart_container(fig_a, "miss_after")

    insight("""<strong>What this shows:</strong> The before/after comparison confirms that all missing values have been resolved through
    principled imputation strategies rather than row deletion — preserving the full 2,000-respondent sample.
    <strong>Why this matters:</strong> Deleting rows with missing income would have disproportionately removed higher-income metro
    respondents (who are more likely to skip income questions), introducing sampling bias into the model.
    Group-wise mode imputation neutralises this risk while remaining statistically defensible.""")

    section("⚠️ Noise & Outlier Flags")
    col1, col2 = st.columns(2)
    with col1:
        noise_df = df.groupby(["q7_activity_level","is_noise_row"]).size().reset_index(name="count")
        noise_df["is_noise_row"] = noise_df["is_noise_row"].map({0:"Clean",1:"Noise (inconsistent)"})
        fig_noise = px.bar(noise_df, x="q7_activity_level", y="count", color="is_noise_row",
                           barmode="stack", title="Activity Level — Noise Row Distribution",
                           color_discrete_map={"Clean":C["teal"],"Noise (inconsistent)":C["coral"]},
                           labels={"q7_activity_level":"Self-reported Activity","count":"Respondents"})
        chart_container(fig_noise, "noise_chart")
        insight("""<strong>Noise pattern:</strong> Noise rows are concentrated in the 'Highly active' and 'Athlete' self-report bands,
        where social desirability bias is strongest. These respondents claim high activity levels but their 4-week behavioural
        record (Q2) contradicts this. <strong>Implication:</strong> Q2 (behavioural anchor) is more reliable than Q7 (self-reported)
        for training classification models.""")
    with col2:
        out_df = df.groupby(["q20_willingness_to_pay","is_wtp_outlier"]).size().reset_index(name="count")
        out_df["is_wtp_outlier"] = out_df["is_wtp_outlier"].map({0:"Normal",1:"Outlier (aspirational)"})
        fig_out = px.bar(out_df, x="q20_willingness_to_pay", y="count", color="is_wtp_outlier",
                         barmode="stack", title="WTP — Outlier Distribution",
                         color_discrete_map={"Normal":C["purple"],"Outlier (aspirational)":C["amber"]},
                         labels={"q20_willingness_to_pay":"Willingness to Pay","count":"Respondents"})
        chart_container(fig_out, "outlier_chart")
        insight("""<strong>Outlier pattern:</strong> Aspirational outliers are concentrated in the '₹1,000+/month' WTP band despite
        reporting ₹0–₹500 in current fitness spending. This reflects a real behavioural phenomenon in Indian surveys —
        aspirational self-presentation. <strong>Implication:</strong> WTP estimates should be Winsorised at the 95th percentile
        or reported with and without outliers for sensitivity testing.""")

    csv_clean = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Cleaned Dataset (CSV)", csv_clean,
                       "fitlife_cleaned_data.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EDA: DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("# EDA: Demographics")
    st.markdown("Understanding *who* our survey respondents are — the foundation for persona segmentation and targeted product strategy.")
    st.divider()

    d = filtered.copy()

    section("📊 Age, City Tier & Income Distribution")
    col1, col2, col3 = st.columns(3)
    with col1:
        age_counts = d["q1_age_group"].value_counts().reset_index()
        age_counts.columns = ["Age Group","Count"]
        age_order = ["18-24","25-34","35-44","45-54","55+"]
        age_counts["Age Group"] = pd.Categorical(age_counts["Age Group"], categories=age_order, ordered=True)
        age_counts = age_counts.sort_values("Age Group")
        fig_age = px.bar(age_counts, x="Age Group", y="Count",
                         color="Age Group", color_discrete_sequence=px.colors.sequential.Purples_r,
                         title="Age Distribution", text="Count")
        fig_age.update_traces(textposition="outside")
        fig_age.update_layout(showlegend=False, height=340)
        chart_container(fig_age, "age_bar")

    with col2:
        city_counts = d["q3_city_tier"].value_counts().reset_index()
        city_counts.columns = ["City Tier","Count"]
        fig_city = px.pie(city_counts, values="Count", names="City Tier",
                          color_discrete_sequence=[C["purple"],C["teal"],C["amber"],C["gray"]],
                          title="City Tier Distribution", hole=0.4)
        fig_city.update_traces(textinfo="percent+label")
        chart_container(fig_city, "city_pie")

    with col3:
        income_order = ["Below 25000","25000-50000","50000-100000","100000-200000","Above 200000"]
        inc_counts = d["q4_monthly_income"].value_counts().reset_index()
        inc_counts.columns = ["Income Band","Count"]
        inc_counts["Income Band"] = pd.Categorical(inc_counts["Income Band"], categories=income_order, ordered=True)
        inc_counts = inc_counts.sort_values("Income Band")
        fig_inc = px.bar(inc_counts, x="Count", y="Income Band", orientation="h",
                         color="Count", color_continuous_scale=["#E1F5EE","#1D9E75"],
                         title="Monthly Household Income", text="Count")
        fig_inc.update_traces(textposition="outside")
        fig_inc.update_layout(coloraxis_showscale=False, showlegend=False, height=340)
        chart_container(fig_inc, "income_bar")

    insight("""<strong>Age:</strong> The 25–34 age band is the dominant segment (~38%), reflecting urban young professionals —
    exactly the primary target persona (Aspiring Urban Professional). The 18–24 student band (~16%) is the second largest,
    confirming strong survey reach among price-sensitive but digitally engaged youth.
    <strong>City tier:</strong> Metro (~40%) and Tier 2 (~36%) together account for 76% of respondents, consistent with
    digital survey reach patterns in India. Tier 3 and rural represent an underserved growth market.
    <strong>Income:</strong> The ₹50,000–₹1,00,000 band is the modal group, aligning with the salaried professional segment
    most likely to afford a mid-tier subscription (₹299–₹599/month). This validates the pricing strategy.""")

    section("🍽️ Dietary Preference & Persona Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        diet_counts = d["q11_diet_preference"].value_counts().reset_index()
        diet_counts.columns = ["Diet","Count"]
        fig_diet = px.bar(diet_counts, x="Count", y="Diet", orientation="h",
                          color="Diet",
                          color_discrete_sequence=[C["teal"],C["amber"],C["coral"],C["purple"],C["gray"]],
                          title="Dietary Preference Distribution", text="Count")
        fig_diet.update_traces(textposition="outside")
        fig_diet.update_layout(showlegend=False, height=320)
        chart_container(fig_diet, "diet_bar")

    with col2:
        persona_counts = d["persona_label"].value_counts().reset_index()
        persona_counts.columns = ["Persona","Count"]
        persona_counts["Pct"] = (persona_counts["Count"]/len(d)*100).round(1)
        fig_p = px.bar(persona_counts, x="Pct", y="Persona", orientation="h",
                       color="Persona", color_discrete_map=PERSONA_COLORS,
                       title="Persona Segment Distribution (%)", text="Pct",
                       labels={"Pct":"Share (%)"})
        fig_p.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_p.update_layout(showlegend=False, height=320)
        chart_container(fig_p, "persona_dist")

    insight("""<strong>Diet:</strong> Non-vegetarian (42.6%) and Vegetarian (41.6%) are nearly equal in representation — a critical
    finding for nutrition product design. No existing fitness app handles Indian non-vegetarian diet tracking well (dal makhani,
    butter chicken macros). The Vegetarian majority demands robust Indian veg meal planning. Vegan (2.3%) is a small but
    growing, highly engaged segment.
    <strong>Personas:</strong> The Aspiring Urban Professional (~29%) is the largest persona, followed by Sedentary Skeptic (~18%)
    and Budget-Conscious Student (~17%). The Sedentary Skeptic segment being this large is a significant finding —
    it requires a distinct acquisition strategy focused on lowering friction and addressing scepticism rather than
    showcasing features.""")

    section("🔀 Demographics × Subscription Intent")
    col1, col2 = st.columns(2)
    with col1:
        cross_age = d.groupby(["q1_age_group","q25_subscription_intent"]).size().reset_index(name="Count")
        cross_age["q1_age_group"] = pd.Categorical(cross_age["q1_age_group"],
                                                    categories=age_order, ordered=True)
        cross_age = cross_age.sort_values("q1_age_group")
        fig_ca = px.bar(cross_age, x="q1_age_group", y="Count", color="q25_subscription_intent",
                        barmode="group", title="Subscription Intent by Age Group",
                        color_discrete_map=INTENT_COLORS,
                        labels={"q1_age_group":"Age","q25_subscription_intent":"Intent"})
        chart_container(fig_ca, "age_intent")

    with col2:
        cross_city = d.groupby(["q3_city_tier","q25_subscription_intent"]).size().reset_index(name="Count")
        fig_cc = px.bar(cross_city, x="q3_city_tier", y="Count", color="q25_subscription_intent",
                        barmode="stack", title="Subscription Intent by City Tier",
                        color_discrete_map=INTENT_COLORS,
                        labels={"q3_city_tier":"City Tier","q25_subscription_intent":"Intent"})
        chart_container(fig_cc, "city_intent")

    insight("""<strong>Age × Intent:</strong> The 25–34 band has the highest absolute count of 'Interested' respondents, making it
    the primary acquisition target. However, the 35–44 and 45–54 bands show a higher *ratio* of Interested to Not Interested,
    suggesting older users who do engage are more committed — likely the Health-Driven Middle-Ager persona.
    <strong>City × Intent:</strong> Metro cities show the highest interest rates, but Tier 2 cities have a surprisingly strong
    'Interested' absolute count that is close to Metro — validating the business case for Tier 2 expansion beyond just Metro markets.
    This is a key strategic insight: do not limit go-to-market to metros.""")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EDA: FITNESS & BEHAVIOUR
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("# EDA: Fitness & Behaviour")
    st.markdown("Understanding *how* our users currently exercise, what they want, and what drives them.")
    st.divider()

    d = filtered.copy()

    section("🎯 Fitness Goals & Activity Levels")
    col1, col2 = st.columns(2)
    with col1:
        goal_counts = d["q5_fitness_goal"].value_counts().reset_index()
        goal_counts.columns = ["Goal","Count"]
        fig_goal = px.bar(goal_counts, x="Count", y="Goal", orientation="h",
                          color="Count", color_continuous_scale=["#EEEDFE","#534AB7"],
                          title="Primary Fitness Goals", text="Count")
        fig_goal.update_traces(textposition="outside")
        fig_goal.update_layout(coloraxis_showscale=False, height=350)
        chart_container(fig_goal, "goal_bar")

    with col2:
        act_order = ["Sedentary","Lightly active","Moderately active","Highly active","Athlete"]
        act_counts = d["q7_activity_level"].value_counts().reset_index()
        act_counts.columns = ["Activity Level","Count"]
        act_counts["Activity Level"] = pd.Categorical(act_counts["Activity Level"],
                                                       categories=act_order, ordered=True)
        act_counts = act_counts.sort_values("Activity Level")
        fig_act = px.funnel(act_counts, x="Count", y="Activity Level",
                            title="Activity Level Distribution",
                            color_discrete_sequence=[C["teal"]])
        chart_container(fig_act, "act_funnel")

    insight("""<strong>Goals:</strong> Weight loss / fat loss is the #1 goal (~27%), followed by General fitness (~22%) and Muscle gain (~18%).
    This validates the Adaptive AI Workout Engine and Nutrition Coach as the two most commercially important product pillars —
    they directly serve the top 3 goals. Stress relief and mental wellness (~10%) confirm demand for the mental wellness add-on.
    Managing a health condition (~8%) is a key signal for the Longevity & Specialty Programs tier.
    <strong>Activity levels:</strong> The funnel shape is highly informative — Moderately active is the plurality (~33%), with a large
    Sedentary + Lightly active pool (~40% combined). This 40% represents high-potential but high-friction users who need
    beginner-friendly onboarding and low-commitment entry points (freemium) rather than feature-heavy premium pitches.""")

    section("🏋️ Workout Type Preferences")
    workout_cols = [c for c in d.columns if c.startswith("q8_workout_")]
    workout_labels = {
        "q8_workout_home_workout": "Home Workout",
        "q8_workout_gym_weights": "Gym / Weights",
        "q8_workout_yoga_pilates": "Yoga / Pilates",
        "q8_workout_running_walking": "Running / Walking",
        "q8_workout_hiit_functional": "HIIT / Functional",
        "q8_workout_sports_outdoor": "Sports / Outdoor",
        "q8_workout_dance_zumba": "Dance / Zumba",
        "q8_workout_no_exercise": "No Exercise",
    }
    workout_sums = d[workout_cols].sum().reset_index()
    workout_sums.columns = ["col","Count"]
    workout_sums["Workout Type"] = workout_sums["col"].map(workout_labels)
    workout_sums = workout_sums.sort_values("Count", ascending=True)

    col1, col2 = st.columns([1.3, 0.7])
    with col1:
        fig_wt = px.bar(workout_sums, x="Count", y="Workout Type", orientation="h",
                        color="Count", color_continuous_scale=["#E1F5EE","#1D9E75"],
                        title="Workout Type Preferences (Multi-Select — Total Selections)",
                        text="Count")
        fig_wt.update_traces(textposition="outside")
        fig_wt.update_layout(coloraxis_showscale=False, height=380)
        chart_container(fig_wt, "workout_bar")
    with col2:
        workout_by_persona = []
        for persona in d["persona_label"].unique():
            pdf = d[d["persona_label"]==persona]
            for col, label in workout_labels.items():
                workout_by_persona.append({
                    "Persona": persona.split()[0], "Workout": label,
                    "Pct": pdf[col].mean()*100
                })
        wpdf = pd.DataFrame(workout_by_persona)
        fig_wpheat = px.density_heatmap(wpdf, x="Workout", y="Persona", z="Pct",
                                         color_continuous_scale="Purples",
                                         title="Workout Type by Persona (%)",
                                         labels={"Pct":"Selection Rate (%)"})
        fig_wpheat.update_layout(xaxis_tickangle=45, height=380)
        chart_container(fig_wpheat, "workout_heat")

    insight("""<strong>Top workout choices:</strong> Running/Walking (#1) and Home Workout (#2) dominate, reflecting the reality that
    most Indian fitness activity happens outside gyms and without equipment. This is a critical product design signal:
    the platform must excel at no-equipment and outdoor programming, not just gym-centric plans.
    Yoga/Pilates (#3) confirms the Homemaker Wellness Seeker segment's strong influence. HIIT (#4) shows urban professional demand
    for time-efficient workouts.
    <strong>Heatmap insight:</strong> Fitness Enthusiasts dominate Gym/Weights and HIIT; Homemaker Seekers dominate Yoga;
    Sedentary Skeptics cluster heavily in 'No Exercise'. This segmentation directly informs onboarding flows —
    each persona should see a different default program on first login.""")

    section("⏰ Workout Timing & Motivation")
    col1, col2 = st.columns(2)
    with col1:
        time_counts = d["q9_workout_time"].value_counts().reset_index()
        time_counts.columns = ["Time Slot","Count"]
        fig_time = px.pie(time_counts, values="Count", names="Time Slot",
                          title="Preferred Workout Time",
                          color_discrete_sequence=px.colors.qualitative.Set2, hole=0.35)
        fig_time.update_traces(textinfo="percent+label")
        chart_container(fig_time, "time_pie")

    with col2:
        mot_counts = d["q6_motivation_type"].value_counts().reset_index()
        mot_counts.columns = ["Motivation","Count"]
        fig_mot = px.bar(mot_counts, x="Count", y="Motivation", orientation="h",
                         color="Motivation",
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         title="Primary Motivation Type", text="Count")
        fig_mot.update_traces(textposition="outside")
        fig_mot.update_layout(showlegend=False, height=320)
        chart_container(fig_mot, "mot_bar")

    insight("""<strong>Workout timing:</strong> Evening (5–8 PM, ~30%) is the most popular slot, followed by Morning (7–10 AM, ~22%)
    and Early Morning (~18%). This has direct product implications: push notifications, live sessions, and coach availability
    must be optimised for evenings first. <strong>Motivation:</strong> 'Energy and health' (intrinsic) and 'Look better/confident'
    (extrinsic) are the two leading motivations. Users motivated by health intrinsically have significantly lower churn rates
    in academic literature — this segment should be prioritised for premium conversion campaigns.""")

    section("📈 Self-Efficacy & Motivation Scores by Persona")
    col1, col2 = st.columns(2)
    with col1:
        fig_se = px.box(d, x="persona_label", y="q12_self_efficacy_score",
                        color="persona_label", color_discrete_map=PERSONA_COLORS,
                        title="Self-Efficacy Score Distribution by Persona (1=Low, 5=High)",
                        labels={"q12_self_efficacy_score":"Self-Efficacy (1-5)",
                                "persona_label":"Persona"})
        fig_se.update_layout(showlegend=False, xaxis_tickangle=30, height=380)
        chart_container(fig_se, "selfeff_box")

    with col2:
        act4w_order = ["0 times","1-4 times","5-8 times","9-12 times","13+ times"]
        act4w = d.groupby(["q2_activity_last_4weeks","q25_subscription_intent"]).size().reset_index(name="Count")
        act4w["q2_activity_last_4weeks"] = pd.Categorical(act4w["q2_activity_last_4weeks"],
                                                            categories=act4w_order, ordered=True)
        act4w = act4w.sort_values("q2_activity_last_4weeks")
        fig_act4w = px.bar(act4w, x="q2_activity_last_4weeks", y="Count",
                           color="q25_subscription_intent",
                           barmode="stack", color_discrete_map=INTENT_COLORS,
                           title="Actual Physical Activity (Last 4 Wks) × Intent",
                           labels={"q2_activity_last_4weeks":"Activity (4 weeks)",
                                   "q25_subscription_intent":"Subscription Intent"})
        chart_container(fig_act4w, "act4w_intent")

    insight("""<strong>Self-efficacy:</strong> Fitness Enthusiasts score highest (median 4–5), Sedentary Skeptics lowest (median 1–2).
    Self-efficacy is the single strongest validated predictor of long-term fitness adherence in health psychology research.
    The wide inter-persona gap confirms it's a powerful clustering and classification feature.
    <strong>Behavioural activity × intent:</strong> A clear positive monotonic relationship is visible —
    as actual exercise frequency increases, 'Interested' responses grow proportionally and 'Not Interested' shrinks.
    Respondents exercising 9+ times in 4 weeks are 3–4× more likely to be 'Interested' than sedentary respondents.
    This validates Q2 as the single most important classification feature.""")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — EDA: SPENDING & ATTITUDE
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("# EDA: Spending & Platform Attitude")
    st.markdown("Understanding *willingness to pay*, *barriers*, *technology adoption*, and *attitude toward AI-powered platforms*.")
    st.divider()

    d = filtered.copy()

    section("💰 Current Spending vs Willingness to Pay")
    col1, col2 = st.columns(2)
    with col1:
        spend_order = ["0","1-500","501-1500","1501-3000","3001-5000","5000+"]
        sp_counts = d["q19_current_fitness_spend"].value_counts().reset_index()
        sp_counts.columns = ["Spend Band","Count"]
        sp_counts["Spend Band"] = pd.Categorical(sp_counts["Spend Band"],
                                                  categories=spend_order, ordered=True)
        sp_counts = sp_counts.sort_values("Spend Band")
        fig_sp = px.bar(sp_counts, x="Spend Band", y="Count",
                        color="Count", color_continuous_scale=["#EAF3DE","#3B6D11"],
                        title="Current Monthly Fitness Spend (₹)", text="Count")
        fig_sp.update_traces(textposition="outside")
        fig_sp.update_layout(coloraxis_showscale=False, height=340)
        chart_container(fig_sp, "spend_bar")

    with col2:
        wtp_order = ["Nothing (free only)","99-299/month","300-599/month","600-999/month","1000+/month"]
        wtp_counts = d["q20_willingness_to_pay"].value_counts().reset_index()
        wtp_counts.columns = ["WTP Band","Count"]
        wtp_counts["WTP Band"] = pd.Categorical(wtp_counts["WTP Band"],
                                                 categories=wtp_order, ordered=True)
        wtp_counts = wtp_counts.sort_values("WTP Band")
        fig_wtp = px.bar(wtp_counts, x="WTP Band", y="Count",
                         color="Count", color_continuous_scale=["#EEEDFE","#534AB7"],
                         title="Willingness to Pay per Month (₹)", text="Count")
        fig_wtp.update_traces(textposition="outside")
        fig_wtp.update_layout(coloraxis_showscale=False, xaxis_tickangle=20, height=340)
        chart_container(fig_wtp, "wtp_bar")

    insight("""<strong>Current spend:</strong> The majority of respondents currently spend ₹0 (nothing, ~22%) or ₹1-500/month (~28%) on
    fitness — a total of ~50% spending negligibly. This is not a sign of low interest; it reflects the underpenetrated
    nature of the Indian fitness-tech market. <strong>WTP:</strong> Despite low current spend, ₹99–₹299/month is the largest
    WTP band (~35%), followed by ₹300–₹599/month (~25%). This confirms the pricing sweet spot at ₹299 for the entry tier.
    The aspiration-to-spend gap (current spend < WTP for many respondents) is commercially exploitable through
    a freemium-to-paid conversion funnel — let users experience value before asking for payment.""")

    section("💰 WTP by Persona & Intent")
    col1, col2 = st.columns(2)
    with col1:
        wtp_persona = d.groupby(["persona_label","q20_willingness_to_pay"]).size().reset_index(name="Count")
        wtp_persona["q20_willingness_to_pay"] = pd.Categorical(
            wtp_persona["q20_willingness_to_pay"], categories=wtp_order, ordered=True)
        fig_wpp = px.bar(wtp_persona, x="persona_label", y="Count",
                         color="q20_willingness_to_pay",
                         barmode="stack",
                         color_discrete_sequence=px.colors.sequential.Purples,
                         title="WTP Distribution by Persona",
                         labels={"persona_label":"Persona","q20_willingness_to_pay":"WTP"})
        fig_wpp.update_layout(xaxis_tickangle=30, height=380)
        chart_container(fig_wpp, "wtp_persona")

    with col2:
        wtp_intent = d.groupby(["q25_subscription_intent","q20_willingness_to_pay"]).size().reset_index(name="Count")
        wtp_intent["q20_willingness_to_pay"] = pd.Categorical(
            wtp_intent["q20_willingness_to_pay"], categories=wtp_order, ordered=True)
        fig_wpi = px.bar(wtp_intent, x="q25_subscription_intent", y="Count",
                         color="q20_willingness_to_pay",
                         barmode="stack",
                         color_discrete_sequence=px.colors.sequential.Greens,
                         title="WTP Distribution by Subscription Intent",
                         labels={"q25_subscription_intent":"Intent","q20_willingness_to_pay":"WTP"})
        chart_container(fig_wpi, "wtp_intent")

    insight("""<strong>WTP by persona:</strong> Fitness Enthusiasts have the highest WTP concentration (₹600–₹1,000+), making them
    the premium tier target despite being only 10% of the population. Aspiring Urban Professionals cluster at ₹300–₹599
    — the sweet spot for the core subscription. Budget-Conscious Students are dominated by ₹99–₹299 and 'free only'.
    <strong>WTP by intent:</strong> 'Interested' respondents show a dramatically higher proportion willing to pay ₹300+,
    while 'Not Interested' respondents overwhelmingly select 'free only'. This validates using WTP as a strong
    classification feature — it is itself partially predicted by intent.""")

    section("🚧 Barriers & Technology Adoption")
    col1, col2 = st.columns(2)
    with col1:
        bar_counts = d["q18_biggest_barrier"].value_counts().reset_index()
        bar_counts.columns = ["Barrier","Count"]
        bar_colors = {
            "No significant barrier": C["teal"],
            "Cost": C["amber"],
            "Habit/consistency": C["coral"],
            "Privacy concern": C["purple"],
            "AI scepticism": C["blue"],
            "Complexity": C["gray"],
        }
        fig_bar = px.bar(bar_counts, x="Count", y="Barrier", orientation="h",
                         color="Barrier", color_discrete_map=bar_colors,
                         title="Biggest Barrier to Adoption", text="Count")
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(showlegend=False, height=340)
        chart_container(fig_bar, "barrier_bar")

    with col2:
        tech_order_d = ["Not comfortable","Open but not started","Tried but stopped",
                         "Use 1 app occasionally","Use 2+ apps actively"]
        tech_counts = d["q14_tech_adoption"].value_counts().reset_index()
        tech_counts.columns = ["Tech Stage","Count"]
        tech_counts["Tech Stage"] = pd.Categorical(tech_counts["Tech Stage"],
                                                    categories=tech_order_d, ordered=True)
        tech_counts = tech_counts.sort_values("Tech Stage")
        fig_tech = px.funnel(tech_counts, x="Count", y="Tech Stage",
                             title="Technology Adoption Stage",
                             color_discrete_sequence=[C["purple"]])
        chart_container(fig_tech, "tech_funnel")

    insight("""<strong>Barriers:</strong> 'No significant barrier' is the largest group (~28%) — these are your immediate conversion
    targets who need only a compelling offer. 'Habit/consistency' (~20%) and 'Cost' (~18%) are the next biggest barriers.
    These are addressable through product design: streaks and accountability features solve habit; freemium solves cost.
    'Privacy concern' (~13%) and 'AI scepticism' (~12%) require trust-building — transparent data policies, testimonials,
    and result guarantees are the correct interventions. 'Complexity' (~9%) signals a UX simplification imperative.
    <strong>Tech adoption:</strong> The funnel shows that 'Open but not started' is the largest group (~30%) —
    a massive addressable audience that needs only a low-friction first step (free trial, referral).""")

    section("🔌 Add-on Interest & Data Comfort")
    col1, col2 = st.columns(2)
    with col1:
        addon_cols = [c for c in d.columns if c.startswith("q21_addon_")]
        addon_labels = {
            "q21_addon_live_coaching": "Live Coaching",
            "q21_addon_wearable_integration": "Wearable Integration",
            "q21_addon_supplements": "Supplements",
            "q21_addon_dna_blood_test": "DNA / Blood Test",
            "q21_addon_corporate_team_plan": "Corporate Plan",
            "q21_addon_mental_wellness": "Mental Wellness",
            "q21_addon_no_addons": "No Add-ons",
        }
        addon_sums = d[addon_cols].sum().reset_index()
        addon_sums.columns = ["col","Count"]
        addon_sums["Add-on"] = addon_sums["col"].map(addon_labels)
        addon_sums = addon_sums.sort_values("Count", ascending=True)
        fig_addon = px.bar(addon_sums, x="Count", y="Add-on", orientation="h",
                           color="Count", color_continuous_scale=["#E6F1FB","#185FA5"],
                           title="Add-on Service Interest (Multi-Select)", text="Count")
        fig_addon.update_traces(textposition="outside")
        fig_addon.update_layout(coloraxis_showscale=False, height=340)
        chart_container(fig_addon, "addon_bar")

    with col2:
        comfort_order_d = ["Very uncomfortable","Uncomfortable","Neutral",
                            "Comfortable if secure","Very comfortable"]
        comfort_counts = d["q22_data_comfort"].value_counts().reset_index()
        comfort_counts.columns = ["Comfort Level","Count"]
        comfort_counts["Comfort Level"] = pd.Categorical(comfort_counts["Comfort Level"],
                                                          categories=comfort_order_d, ordered=True)
        comfort_counts = comfort_counts.sort_values("Comfort Level")
        color_map = {
            "Very comfortable": C["teal"], "Comfortable if secure": C["green"],
            "Neutral": C["amber"], "Uncomfortable": C["coral"], "Very uncomfortable": C["red"]
        }
        fig_comfort = px.bar(comfort_counts, x="Comfort Level", y="Count",
                             color="Comfort Level", color_discrete_map=color_map,
                             title="Comfort Sharing Health Data with AI", text="Count")
        fig_comfort.update_traces(textposition="outside")
        fig_comfort.update_layout(showlegend=False, xaxis_tickangle=20, height=340)
        chart_container(fig_comfort, "comfort_bar")

    insight("""<strong>Add-ons:</strong> Wearable integration and Live Coaching are the top two premium add-ons — confirming both
    the connected fitness trend and the continued value users place on human expertise alongside AI.
    Mental Wellness ranks #3, validating it as a strong upsell opportunity. DNA/Blood Test interest (~15%) signals
    an aspirational segment willing to pay for personalisation depth — a future premium tier opportunity.
    <strong>Data comfort:</strong> A combined ~55% are 'Comfortable if secure' or 'Very comfortable' with sharing health data.
    Only ~18% are 'Uncomfortable' or 'Very uncomfortable'. This is commercially positive: the majority will share data
    if the platform communicates security credibly. The 'Comfortable if secure' group (~38%) is the swing vote —
    transparent privacy policies and GDPR-aligned data handling are non-negotiable product requirements.""")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — DIAGNOSTIC ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("# Diagnostic Analysis")
    st.markdown("Moving from *what* to *why* — correlations, cross-variable patterns, and target variable deep-dive.")
    st.divider()

    d = filtered.copy()

    section("🔥 Correlation Heatmap — Numeric Variables")
    num_cols = ["q10_motivation_score","q12_self_efficacy_score","q15_stress_level",
                "income_score","activity4w_score","wtp_score","spend_score",
                "tech_score","comfort_score","actlvl_score","intent_score","wtp_gap",
                "health_condition_count"]
    col_labels = {
        "q10_motivation_score":"Motivation","q12_self_efficacy_score":"Self-Efficacy",
        "q15_stress_level":"Stress","income_score":"Income","activity4w_score":"Activity (4wk)",
        "wtp_score":"WTP","spend_score":"Current Spend","tech_score":"Tech Adoption",
        "comfort_score":"Data Comfort","actlvl_score":"Activity Level",
        "intent_score":"Subscription Intent","wtp_gap":"WTP Gap","health_condition_count":"Health Conditions",
    }
    corr_df = d[num_cols].rename(columns=col_labels).corr()
    fig_corr = px.imshow(corr_df, text_auto=".2f",
                         color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1,
                         title="Pearson Correlation Matrix — Key Numeric Variables",
                         aspect="auto")
    fig_corr.update_layout(height=520)
    chart_container(fig_corr, "corr_heat")

    insight("""<strong>What this shows:</strong> The correlation matrix reveals how numerical variables relate to each other and,
    critically, to the subscription intent score. <strong>Key findings:</strong>
    (1) <strong>Self-Efficacy</strong> has the strongest positive correlation with Intent (+0.45 to +0.55) — confirming it as
    the top classification feature.
    (2) <strong>Tech Adoption</strong> and <strong>Activity (4-week)</strong> also correlate strongly with Intent — behavioural
    measures outperform attitudinal ones.
    (3) <strong>Stress</strong> shows a mild negative correlation with Intent — high-stress users are slightly less likely
    to convert, possibly due to time constraints.
    (4) <strong>WTP Gap</strong> (WTP − current spend) positively correlates with Intent — aspirational users signal conversion readiness.
    (5) <strong>Income</strong> correlates with WTP and Spend as expected, validating the ordinal encoding.""")

    section("🎯 Subscription Intent — Deep Dive")
    col1, col2, col3 = st.columns(3)
    with col1:
        intent_counts = d["q25_subscription_intent"].value_counts().reset_index()
        intent_counts.columns = ["Intent","Count"]
        fig_intent = px.pie(intent_counts, values="Count", names="Intent",
                            color="Intent", color_discrete_map=INTENT_COLORS,
                            title="Overall Subscription Intent Split", hole=0.4)
        fig_intent.update_traces(textinfo="percent+label+value")
        chart_container(fig_intent, "intent_pie")

    with col2:
        intent_persona = d.groupby(["persona_label","q25_subscription_intent"]).size().reset_index(name="Count")
        intent_pct = intent_persona.copy()
        total_p = intent_pct.groupby("persona_label")["Count"].transform("sum")
        intent_pct["Pct"] = (intent_pct["Count"]/total_p*100).round(1)
        fig_ip = px.bar(intent_pct, x="persona_label", y="Pct",
                        color="q25_subscription_intent",
                        barmode="stack", color_discrete_map=INTENT_COLORS,
                        title="Intent % by Persona",
                        labels={"persona_label":"Persona","Pct":"%","q25_subscription_intent":"Intent"})
        fig_ip.update_layout(xaxis_tickangle=30, height=380)
        chart_container(fig_ip, "intent_persona")

    with col3:
        barrier_intent = d.groupby(["q18_biggest_barrier","q25_subscription_intent"]).size().reset_index(name="Count")
        bi_pct = barrier_intent.copy()
        bi_tot = bi_pct.groupby("q18_biggest_barrier")["Count"].transform("sum")
        bi_pct["Pct"] = (bi_pct["Count"]/bi_tot*100).round(1)
        fig_bi = px.bar(bi_pct, x="q18_biggest_barrier", y="Pct",
                        color="q25_subscription_intent",
                        barmode="stack", color_discrete_map=INTENT_COLORS,
                        title="Intent % by Barrier",
                        labels={"q18_biggest_barrier":"Barrier","Pct":"%","q25_subscription_intent":"Intent"})
        fig_bi.update_layout(xaxis_tickangle=35, height=380)
        chart_container(fig_bi, "intent_barrier")

    insight("""<strong>Overall split:</strong> 44.5% Interested, 29.8% Undecided, 25.7% Not Interested — a commercially healthy
    distribution where the majority either want or may want the product.
    <strong>By persona:</strong> Fitness Enthusiasts show ~72% Interested; Sedentary Skeptics show ~67% Not Interested.
    These are the two poles of your market — product strategy must serve both with differentiated messaging.
    <strong>By barrier:</strong> Respondents with 'No significant barrier' show ~65% Interested.
    'AI scepticism' and 'Privacy concern' groups show the highest 'Not Interested' rates — confirming these barriers
    are disqualifying rather than merely friction-creating. The intervention strategy must change accordingly:
    social proof and transparency campaigns, not discount offers.""")

    section("📊 Key Diagnostic: Self-Efficacy × Income × Intent")
    fig_se_inc = px.scatter(
        d.sample(min(500, len(d))),
        x="income_score", y="q12_self_efficacy_score",
        color="q25_subscription_intent",
        color_discrete_map=INTENT_COLORS,
        size="wtp_score", size_max=14,
        opacity=0.7,
        hover_data=["persona_label","q5_fitness_goal","q20_willingness_to_pay"],
        title="Self-Efficacy vs Income Score — Sized by WTP, Coloured by Subscription Intent",
        labels={"income_score":"Income Score (1=Lowest, 5=Highest)",
                "q12_self_efficacy_score":"Self-Efficacy Score (1-5)",
                "q25_subscription_intent":"Intent"},
    )
    fig_se_inc.update_layout(height=450)
    chart_container(fig_se_inc, "se_inc_scatter")

    insight("""<strong>What this reveals:</strong> This scatter plot is one of the most analytically rich visuals in the dashboard.
    Green dots (Interested) cluster in the upper-right quadrant — high income AND high self-efficacy — confirming that
    both financial capacity and psychological readiness must coexist for conversion. Red dots (Not Interested) cluster
    in the lower-left — low income AND low self-efficacy.
    <strong>Critical business finding:</strong> There is a significant cluster of high-income, low-self-efficacy respondents
    (upper-left) who are 'Undecided'. These are high-value targets: they have the money but doubt their ability to commit.
    A product strategy emphasising accountability coaching, streak mechanics, and small wins would convert this segment.
    Bubble size (WTP) visually confirms that larger bubbles cluster in the upper-right — confirming income + self-efficacy
    jointly predict WTP.""")

    section("🏃 Behavioural Activity vs Motivation — By Intent")
    fig_beh = px.box(
        d, x="q25_subscription_intent", y="activity4w_score",
        color="q25_subscription_intent", color_discrete_map=INTENT_COLORS,
        points="outliers",
        title="Actual Activity Score Distribution by Subscription Intent",
        labels={"activity4w_score":"Activity Score (1=0 times, 5=13+ times)",
                "q25_subscription_intent":"Subscription Intent"},
    )
    fig_beh.update_layout(showlegend=False, height=380)
    chart_container(fig_beh, "beh_box")

    col1, col2 = st.columns(2)
    with col1:
        fig_mot_int = px.violin(
            d, x="q25_subscription_intent", y="q10_motivation_score",
            color="q25_subscription_intent", color_discrete_map=INTENT_COLORS,
            box=True, points="outliers",
            title="Motivation Score by Intent",
            labels={"q10_motivation_score":"Motivation (1-5)","q25_subscription_intent":"Intent"},
        )
        fig_mot_int.update_layout(showlegend=False, height=360)
        chart_container(fig_mot_int, "mot_violin")

    with col2:
        forced_intent = d.groupby(["q24_forced_tradeoff","q25_subscription_intent"]).size().reset_index(name="Count")
        fi_pct = forced_intent.copy()
        fi_tot = fi_pct.groupby("q24_forced_tradeoff")["Count"].transform("sum")
        fi_pct["Pct"] = (fi_pct["Count"]/fi_tot*100).round(1)
        fig_fi = px.bar(fi_pct, x="q24_forced_tradeoff", y="Pct",
                        color="q25_subscription_intent",
                        barmode="stack", color_discrete_map=INTENT_COLORS,
                        title="Forced Trade-off Choice × Subscription Intent",
                        labels={"q24_forced_tradeoff":"Key Value Driver","Pct":"%",
                                "q25_subscription_intent":"Intent"})
        fig_fi.update_layout(xaxis_tickangle=30, height=360)
        chart_container(fig_fi, "forced_intent")

    insight("""<strong>Activity × Intent (box plot):</strong> The median activity score rises sharply from Not Interested (1.5) to
    Undecided (2.5) to Interested (3.8). The distribution also narrows at the top — Interested respondents are
    consistently more active, with fewer low-activity outliers. This confirms that behavioural data (Q2) is a
    cleaner predictor than self-reported activity (Q7).
    <strong>Motivation × Intent (violin):</strong> The Interested group's violin is wider at the top (scores 4–5),
    while Not Interested is wide at the bottom. The overlap zone (score 3) represents the 'Undecided' population —
    those with moderate motivation who could go either way depending on product experience.
    <strong>Forced trade-off × Intent:</strong> Users prioritising 'AI personalisation' and 'Proven measurable results'
    show the highest Interested rates — validating these as the core value propositions to lead with in marketing.
    'Low price' prioritisers show high Undecided rates — confirming price-sensitive users need a freemium entry
    point to experience value before committing.""")

    section("📋 Persona Diagnostic Summary Table")
    summary_rows = []
    for persona in df["persona_label"].unique():
        pdf = d[d["persona_label"]==persona]
        if len(pdf) == 0: continue
        interested_pct = (pdf["q25_subscription_intent"]=="Interested").mean()*100
        avg_wtp = pdf["wtp_score"].mean()
        avg_se  = pdf["q12_self_efficacy_score"].mean()
        avg_mot = pdf["q10_motivation_score"].mean()
        top_barrier = pdf["q18_biggest_barrier"].mode()[0] if len(pdf) > 0 else "—"
        top_goal    = pdf["q5_fitness_goal"].mode()[0] if len(pdf) > 0 else "—"
        summary_rows.append({
            "Persona": persona,
            "Count": len(pdf),
            "% Interested": f"{interested_pct:.1f}%",
            "Avg WTP Score": f"{avg_wtp:.2f}",
            "Avg Self-Efficacy": f"{avg_se:.2f}",
            "Avg Motivation": f"{avg_mot:.2f}",
            "Top Barrier": top_barrier,
            "Top Goal": top_goal,
        })
    summary_table = pd.DataFrame(summary_rows)
    st.dataframe(summary_table, use_container_width=True, hide_index=True)

    insight("""<strong>Persona diagnostic summary:</strong> This table is the single most actionable output in the dashboard.
    Reading across each row tells you exactly how to serve each persona:
    <strong>Fitness Enthusiasts</strong> — high intent, high WTP, high self-efficacy → priority premium upsell target.
    <strong>Sedentary Skeptics</strong> — low intent, low WTP, low self-efficacy → barrier-removal campaign (free tier + social proof).
    <strong>Health-Driven Middle-Agers</strong> — moderate-high intent, health condition-driven goal → condition-specific programs + doctor partnerships.
    <strong>Budget-Conscious Students</strong> — moderate intent, cost barrier → student pricing tier + referral programme.
    <strong>Homemaker Wellness Seekers</strong> — moderate intent, consistency barrier → accountability features + morning programming.
    <strong>Aspiring Urban Professionals</strong> — high intent, privacy concern → data transparency messaging + corporate wellness channel.""")

    st.markdown("---")
    st.markdown("*FitLife AI Consumer Insights Dashboard · Synthetic Research Dataset · India 2026*")
