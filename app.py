import re

import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# ==========================
# THEME — "instrument cluster" palette
# ==========================

COLOR = {
    "bg": "#0B0E14",
    "surface": "#131A24",
    "border": "#232C39",
    "line": "#1B2733",
    "cyan": "#00D9FF",
    "amber": "#FFA826",
    "green": "#3DDC84",
    "red": "#FF4D6D",
    "violet": "#8B7FFF",
    "text": "#E8EDF4",
    "muted": "#6B7785",
}

DISCRETE_SEQUENCE = [COLOR["cyan"], COLOR["amber"],
                     COLOR["green"], COLOR["violet"], COLOR["red"]]

HERO_IMAGE = "smart-cockpit01.jpg"

SECONDARY_IMAGE = "continental_pp_pillar_to_pillar_display.jpg"


PAGES = [
    ("01", "Executive Summary", "Top-line numbers from the latest survey wave"),
    ("02", "Persona Explorer", "Unsupervised clusters of respondent behavior"),
    ("03", "Trust Analytics", "What it would take to earn driver confidence"),
    ("04", "Feature Prioritization", "Ranked demand for missing capabilities"),
    # ("05", "Purchase Drivers", "Which product features move the buying decision"),
    ("05", "Product Roadmap", "Demand-ranked release plan, auto-generated"),
]
PAGE_LABELS = [f"{num} — {title}" for num, title, _ in PAGES]
PAGE_LOOKUP = {f"{num} — {title}": (num, title, sub)
               for num, title, sub in PAGES}

st.set_page_config(
    page_title="Connected Car Dashboard",
    layout="wide",
)

# ==========================
# CSS — fonts, surfaces, signature elements
# ==========================

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    h1, h2, h3 {{
        font-family: 'Rajdhani', sans-serif !important;
        letter-spacing: 0.01em;
    }}

    [data-testid="stSidebar"] {{
        border-right: 1px solid {COLOR["border"]};
    }}

    [data-testid="stSidebar"] label {{
        font-family: 'Inter', sans-serif;
        font-size: 0.92rem;
    }}

    div[role="radiogroup"] > label {{
        padding: 0.4rem 0.6rem;
        border-radius: 6px;
        margin-bottom: 2px;
    }}

    [data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace;
        color: {COLOR["cyan"]};
    }}

    [data-testid="stMetricLabel"] {{
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-size: 0.72rem;
        color: {COLOR["muted"]};
    }}

    [data-testid="stImage"] img {{
        border-radius: 10px;
        border: 1px solid {COLOR["border"]};
    }}

    [data-testid="stImage"] figcaption {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: {COLOR["muted"]};
    }}

    .hud-eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        color: {COLOR["cyan"]};
        font-size: 0.74rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.15rem;
    }}

    .hud-title {{
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        letter-spacing: 0.01em;
        color: {COLOR["text"]};
        margin: 0 0 0.15rem 0;
    }}

    .hud-subtitle {{
        font-family: 'Inter', sans-serif;
        color: {COLOR["muted"]};
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }}

    .app-banner {{
        padding: 1rem 1.4rem;
        margin-bottom: 0.4rem;
        background: linear-gradient(90deg, {COLOR["surface"]} 0%, rgba(19,26,36,0.25) 100%);
        border: 1px solid {COLOR["border"]};
        border-left: 3px solid {COLOR["cyan"]};
        border-radius: 8px;
    }}

    .app-banner .kicker {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.16em;
        color: {COLOR["muted"]};
        text-transform: uppercase;
        margin-bottom: 0.1rem;
    }}

    .app-banner .brand {{
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        font-size: 1.7rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        color: {COLOR["text"]};
    }}

    .section-label {{
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        font-size: 1.15rem;
        letter-spacing: 0.02em;
        margin: 0.3rem 0 0.6rem 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def tick_divider(n_ticks: int = 48) -> None:
    """Signature section divider styled like an unrolled speedometer arc."""
    ticks = "".join(
        f'<line x1="{i * 1000 // n_ticks}" y1="3" x2="{i * 1000 // n_ticks}" '
        f'y2="{11 if i % 4 == 0 else 8}" stroke="{COLOR["line"]}" stroke-width="1"/>'
        for i in range(n_ticks + 1)
    )
    st.markdown(
        f"""
        <div style="margin:1.6rem 0;">
          <svg viewBox="0 0 1000 14" width="100%" height="14" preserveAspectRatio="none">
            <line x1="0" y1="7" x2="1000" y2="7" stroke="{COLOR["line"]}" stroke-width="1"/>
            {ticks}
          </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(page_label: str) -> None:
    num, title, subtitle = PAGE_LOOKUP[page_label]
    st.markdown(
        f"""
        <div class="hud-eyebrow">Module {num} / {len(PAGES):02d}</div>
        <div class="hud-title">{title}</div>
        <div class="hud-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def section_label(text: str, color: str = None) -> None:
    color = color or COLOR["text"]
    st.markdown(
        f'<div class="section-label" style="color:{color};">{text}</div>', unsafe_allow_html=True)


def style_chart(fig, show_colorbar: bool = True):
    fig.update_layout(
        paper_bgcolor=COLOR["surface"],
        plot_bgcolor=COLOR["surface"],
        font=dict(family="Inter, sans-serif", color="#C9D2DC", size=13),
        title_font=dict(family="Rajdhani, sans-serif",
                        size=18, color=COLOR["text"]),
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=COLOR["line"], zerolinecolor=COLOR["line"])
    fig.update_yaxes(gridcolor=COLOR["line"], zerolinecolor=COLOR["line"])
    if not show_colorbar:
        fig.update_coloraxes(showscale=False)
    return fig


def ranking_bar(data: pd.DataFrame, x: str, y: str, title: str, accent: str):
    fig = px.bar(
        data,
        x=x,
        y=y,
        orientation="h",
        title=title,
        color=x,
        color_continuous_scale=[[0, COLOR["surface"]], [1, accent]],
    )
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    return style_chart(fig, show_colorbar=False)


def priority_card(tag: str, text: str, color: str) -> None:
    st.markdown(
        f"""
        <div style="background:{COLOR['surface']}; border:1px solid {COLOR['border']};
                    border-left:4px solid {color}; border-radius:6px;
                    padding:0.8rem 1.05rem; margin-bottom:0.55rem;
                    display:flex; align-items:center; gap:0.85rem;">
            <span style="font-family:'JetBrains Mono', monospace; font-size:0.72rem;
                        font-weight:600; color:{color}; background:{color}1A;
                        border:1px solid {color}55; border-radius:4px;
                        padding:0.15rem 0.5rem; letter-spacing:0.06em; white-space:nowrap;">
                {tag}
            </span>
            <span style="color:{COLOR['text']}; font-size:0.95rem;">{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("connected_car_ml_ready.csv")

st.markdown(
    """
    <div class="app-banner">
        <div class="kicker">Product Intelligence · Survey-driven dashboard</div>
        <div class="brand">Connected Car Dashboard</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================
# SIDEBAR
# ==========================

with st.sidebar:
    st.markdown(
        f"""<div style="font-family:'JetBrains Mono', monospace; font-size:0.72rem;
        letter-spacing:0.1em; color:{COLOR['muted']}; text-transform:uppercase;
        margin-bottom:0.4rem;">Navigation</div>""",
        unsafe_allow_html=True,
    )

    page_label = st.radio("Navigation", PAGE_LABELS,
                          label_visibility="collapsed")

    st.markdown("---")
    with st.expander("About this dashboard"):
        st.caption(
            "Built from a connected-vehicle owner survey. Each module reads the "
            "same underlying dataset and applies a different analytical lens — "
            "from raw distributions to ML-driven driver and roadmap analysis."
        )
    st.caption(f"{len(df)} respondents loaded")

page = PAGE_LOOKUP[page_label][1]  # plain page title, used by the logic below

# ==========================
# EXECUTIVE SUMMARY
# ==========================

if page == "Executive Summary":

    page_header(page_label)

    st.image(HERO_IMAGE, use_container_width=True)

    question_cols = [
        'What is the make, model, and year of your vehicle',
        'How old are you',
        'Which smartphone OS do you primarily use',
        'How often do you use phone mirroring in the car',
        'Which infotainment setup does your car have',
        'Which screen layout does your car have',
        'Why do you prefer your current infotainment setup',
        'Which map provider do you prefer while driving',
        'Why do you prefer the above selected map provider',
        'Which music/media platforms do you use most',
        'Do you use an in-car app store Apps provided by Auto OEM',
        'Which in-car apps do you use the most at present',
        'For most in-car digital tasks, do you prefer built-in apps or smartphone mirroring',
        'Which of these features do you expect as a standard offering in your next car',
        'Do you actively use the OEM companion app to interact with your vehicle remotely',
        'How do you usually respond when a connected vehicle feature fails while driving',
        'Based on the UI designs shown below, which connected car app experience would you prefer using daily',
        'What could be the main reasons of not using the OEM mobile app more often',
        'How comfortable are you with over-the-air vehicle software updates',
        'What is your preferred frequency for vehicle OTA (Over-the-Air) updates',
        'Does the quality of infotainment and connected features influence your car purchase decision',
        'Which pricing model feels most acceptable for connected car features',
    ]

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.metric("Respondents", len(df))
    with c2:
        with st.container(border=True):
            st.metric("Business Questions", len(question_cols))
    with c3:
        with st.container(border=True):
            st.metric("Segments", 3)

tick_divider()

section_label("Dataset Preview")

st.dataframe(
    df.head(),
    use_container_width=True,
    hide_index=True
)

tick_divider()

# ==========================
# SMARTPHONE OS DISTRIBUTION
# ==========================

if "Which smartphone OS do you primarily use" in df.columns:

    st.markdown(
        """
        <h3 style='text-align:center;'>
        📱 Smartphone OS Distribution
        </h3>
        """,
        unsafe_allow_html=True
    )

    os_counts = (
        df["Which smartphone OS do you primarily use"]
        .value_counts()
        .reset_index()
    )

    os_counts.columns = [
        "OS",
        "Count"
    ]

    left_space, chart_col, right_space = st.columns(
        [1, 2, 1]
    )

    with chart_col:

        fig = px.pie(
            os_counts,
            names="OS",
            values="Count",
            hole=0.30,
            color_discrete_sequence=[
                "#00D4FF",
                "#FFB020",
                "#52E38B"
            ]
        )

    fig.update_traces(
        textinfo="percent",
        textfont_size=16
    )

    fig.update_layout(
        title=None,
        height=500,
        showlegend=False,
        paper_bgcolor="#071426",
        plot_bgcolor="#071426",
        margin=dict(l=0, r=0, t=0, b=0),
        annotations=[
            dict(
                text=f"{len(df)}<br>Users",
                showarrow=False,
                font=dict(size=24)
            )
        ]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    tick_divider()


# ==========================
# PERSONA EXPLORER
# ==========================

elif page == "Persona Explorer":

    page_header(page_label)

    persona_cols = [
        "How old are you",
        "Which smartphone OS do you primarily use",
        "How often do you use phone mirroring in the car",
    ]

    persona_df = df[persona_cols].copy()

    for col in persona_df.columns:
        persona_df[col] = LabelEncoder().fit_transform(
            persona_df[col].astype(str))

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    persona_df["Cluster"] = kmeans.fit_predict(persona_df)

    cluster_names = {
        0: "Tech Enthusiasts",
        1: "Practical Drivers",
        2: "Connected Mobility Users",
    }
    persona_df["Persona"] = persona_df["Cluster"].map(cluster_names)

    persona_counts = persona_df["Persona"].value_counts().reset_index()
    persona_counts.columns = ["Persona", "Count"]

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        fig_pie = px.pie(
            persona_counts,
            names="Persona",
            values="Count",
            color="Persona",
            title="Persona Distribution",
            hole=0.45,
            color_discrete_map={
                "Tech Enthusiasts": COLOR["cyan"],
                "Practical Drivers": COLOR["green"],
                "Connected Mobility Users": COLOR["amber"],
            },
        )
        st.plotly_chart(style_chart(fig_pie), use_container_width=True)

    with col_table:
        section_label("Segment sizes")
        st.dataframe(persona_counts, use_container_width=True, hide_index=True)

    tick_divider()

    fig_hist = px.histogram(
        persona_df, x="Cluster", title="Customer Segments",
        color_discrete_sequence=[COLOR["violet"]],
    )
    st.plotly_chart(style_chart(fig_hist), use_container_width=True)

# ==========================
# TRUST ANALYTICS
# ==========================

elif page == "Trust Analytics":

    page_header(page_label)

    trust_cols = [c for c in df.columns if c.startswith(
        "What would increase your trust")]

    if len(trust_cols) > 0:

        trust_scores = df[trust_cols].sum()
        trust_scores = pd.DataFrame(
            {"Factor": trust_scores.index, "Count": trust_scores.values})
        trust_scores["Factor"] = trust_scores["Factor"].str.replace(
            "What would increase your trust in connected vehicle services the most_", "", regex=False
        )
        trust_scores = trust_scores.sort_values("Count", ascending=False)

        fig = ranking_bar(trust_scores, "Count", "Factor",
                          "Trust Drivers", COLOR["cyan"])
        st.plotly_chart(fig, use_container_width=True)

        tick_divider()
        st.dataframe(trust_scores, use_container_width=True, hide_index=True)

    else:
        st.info("No trust-related columns found in the dataset.")

# ==========================
# FEATURE PRIORITIZATION
# ==========================


elif page == "Feature Prioritization":

    page_header(page_label)

    feature_cols = [c for c in df.columns if c.startswith(
        "Which features are missing from your current car")]
    st.image(SECONDARY_IMAGE, use_container_width=True
             )

    if len(feature_cols) > 0:

        feature_scores = df[feature_cols].sum()
        feature_scores = pd.DataFrame(
            {"Feature": feature_scores.index, "Demand": feature_scores.values})
        feature_scores["Feature"] = feature_scores["Feature"].str.replace(
            "Which features are missing from your current car that you actively wish you had_", "", regex=False
        )
        feature_scores = feature_scores.sort_values("Demand", ascending=False)

        fig = ranking_bar(feature_scores, "Demand", "Feature",
                          "Feature Demand Ranking", COLOR["amber"])
        st.plotly_chart(fig, use_container_width=True)

        tick_divider()
        st.dataframe(feature_scores, use_container_width=True, hide_index=True)

    else:
        st.info("No feature-demand columns found in the dataset.")


# # ==========================
# # PURCHASE DRIVERS
# # ==========================

# elif page == "Purchase Drivers":

#     page_header(page_label)

#     target = "Does the quality of infotainment and connected features influence your car purchase decision"

#     # Demographic / identifier columns aren't product features — keep them out
#     # of the driver model entirely so they can never surface as a "purchase driver".
#     non_feature_cols = [
#         "What is the make, model, and year of your vehicle",
#         "How old are you",
#         "Which smartphone OS do you primarily use",
#     ]

#     y = LabelEncoder().fit_transform(df[target].astype(str))

#     X = df.drop(columns=[target] + non_feature_cols, errors="ignore")
#     X = pd.get_dummies(X.fillna("Missing"))

#     with st.spinner("Training purchase-driver model..."):
#         model = RandomForestClassifier(n_estimators=300, random_state=42)
#         model.fit(X, y)

#     importance = pd.DataFrame(
#         {"Feature": X.columns, "Importance": model.feature_importances_})

#     keywords = [
#         "AI", "Voice", "Navigation", "HUD", "AR", "Bluetooth", "Mirroring",
#         "OTA", "Assistant", "Language", "Connectivity", "Feature", "Interface", "Screen",
#     ]
#     # Whole-word match only — plain substring matching let things like "AR" match
#     # inside "are"/"car"/"smart" and "AI" match inside "air", pulling in unrelated
#     # questions (e.g. age, OTA-comfort) that just happened to contain those letters.
#     keyword_pattern = re.compile(
#         r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b", re.IGNORECASE)

#     importance = importance[importance["Feature"].apply(
#         lambda x: bool(keyword_pattern.search(str(x))))]
#     importance = importance.sort_values("Importance", ascending=False).head(15)

#     fig = ranking_bar(importance, "Importance", "Feature",
#                       "Top Product Purchase Drivers", COLOR["violet"])
#     st.plotly_chart(fig, use_container_width=True)

#     tick_divider()
#     st.dataframe(importance, use_container_width=True, hide_index=True)

# ==========================
# PRODUCT ROADMAP
# ==========================

elif page == "Product Roadmap":

    page_header(page_label)

    feature_cols = [c for c in df.columns if c.startswith(
        "Which features are missing from your current car")]

    if len(feature_cols) == 0:
        st.info("No feature-demand columns found in the dataset.")
    else:
        feature_scores = df[feature_cols].sum()
        feature_scores = pd.DataFrame(
            {"Feature": feature_scores.index, "Demand": feature_scores.values})
        feature_scores["Feature"] = feature_scores["Feature"].str.replace(
            "Which features are missing from your current car that you actively wish you had_", "", regex=False
        )
        feature_scores = feature_scores.sort_values("Demand", ascending=False)
        top_features = feature_scores.head(10)

        section_label("P0 — Immediate Release", COLOR["red"])
        for item in top_features.head(3)["Feature"]:
            priority_card("P0", item, COLOR["red"])

        tick_divider()

        section_label("P1 — Next Release", COLOR["amber"])
        for item in top_features.iloc[3:7]["Feature"]:
            priority_card("P1", item, COLOR["amber"])

        tick_divider()

        section_label("P2 — Future Roadmap", COLOR["green"])
        for item in top_features.iloc[7:10]["Feature"]:
            priority_card("P2", item, COLOR["green"])

        tick_divider()

        section_label("Roadmap Summary")
        c1, c2, c3 = st.columns(3)
        with c1:
            with st.container(border=True):
                st.metric("P0 Features", len(top_features.head(3)))
        with c2:
            with st.container(border=True):
                st.metric("P1 Features", len(top_features.iloc[3:7]))
        with c3:
            with st.container(border=True):
                st.metric("P2 Features", len(top_features.iloc[7:10]))
