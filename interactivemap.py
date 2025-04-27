import os
import pandas as pd
import streamlit as st
import pydeck as pdk
import openai
import datetime
from dotenv import load_dotenv

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --------- PAGE-LIKE BORDER & STYLING ---------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Lato:wght@700&family=Roboto+Slab:wght@700&display=swap');
.stApp {
    border: 4px solid #174ea6 !important;
    border-radius: 28px !important;
    margin: 24px auto !important;
    max-width: 1040px;
    background: #fff !important;
    padding: 16px 0 16px 0 !important;
    box-shadow: 0 7px 44px rgba(23,78,166,0.13);
}
.heading-animate {
    font-family: 'Roboto Slab', serif !important;
    font-size: 3.2rem;
    font-weight: 800;
    color: #111 !important;
    line-height: 1;
    letter-spacing: 1px;
    display: block;
    margin-top: 0.3em;
    margin-bottom: 0.5em;
    text-align: center;
    transition: color 0.23s, text-shadow 0.25s, transform 0.25s;
    cursor: pointer;
}
.heading-animate:hover {
    color: #174ea6 !important;
    text-shadow: 0 4px 28px rgba(23,78,166,0.21), 0 1px 0 #fff;
    transform: scale(1.045) translateY(-2px);
}
.last-updated-box {
    color: #111 !important;
    background: #f6f6f7 !important;
    border-radius: 999px;
    padding: 7px 18px;
    display: inline-block;
    font-size: 1.09em;
    font-weight: 600;
    margin: 12px auto 20px auto;
    text-align: center;
    border: 1.6px solid #e4e7ec;
    box-shadow: 0 1px 4px rgba(80,100,140,0.05);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Lato', Arial, sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em;
    font-weight: 700 !important;
    font-size: 0.86em !important;
    border-radius: 999px !important;
    background: #f6f6f7 !important;
    color: #111 !important;
    padding: 10px 28px !important;
    margin-bottom: 6px !important;
    border: none !important;
    transition: background 0.2s, color 0.2s, box-shadow 0.23s, transform 0.22s;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    position: relative;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #e8eefc !important;
    color: #174ea6 !important;
    box-shadow: 0 2px 8px rgba(23,78,166,0.12);
    border: none !important;
    transform: scale(1.06) translateY(-2px);
}
.stTabs [data-baseweb="tab"]:hover {
    background: #e2e8f6 !important;
    color: #174ea6 !important;
    box-shadow: 0 4px 22px rgba(23,78,166,0.10), 0 2px 0 #fff;
    transform: scale(1.045) translateY(-2px);
    z-index: 10;
}
.stTabs [data-baseweb="tab"]::after {
    display: none !important;
}
.stHeader, .stSubheader, h2, h3, h4, h5 {
    font-family: 'Roboto Slab', serif !important;
    font-weight: 700 !important;
    color: #111 !important;
    letter-spacing: 0.03em;
    margin-top: 1.6em !important;
    margin-bottom: 0.6em !important;
}
.stMarkdown, .stText, .stDataFrame, .stMetric, .stExpander {
    font-family: 'Inter', Arial, sans-serif !important;
    color: #111 !important;
}
.streamlit-expanderHeader {
    font-size: 1.12em !important;
    font-weight: 800 !important;
    color: #174ea6 !important;
    font-family: 'Lato', sans-serif !important;
    letter-spacing: 0.01em;
    padding-left: 3px !important;
}
.expander-card-content {
    background: #f7fbfe;
    border: 1.3px solid #e4e7ec;
    border-radius: 13px;
    margin-top: 8px;
    padding: 16px 14px 10px 18px;
    box-shadow: 0 1px 7px rgba(23,78,166,0.07);
    max-width: 98%;
    margin-left: auto;
    margin-right: auto;
}
.deck-tooltip {
    color: #111 !important;
    background: #f5f8fa !important;
    border-radius: 7px;
    font-size: 1em;
    font-family: 'Inter', Arial, sans-serif !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.10);
}
hr.section-divider {
    border: none;
    border-top: 2px solid #e4e7ec;
    margin: 36px 0 30px 0;
    width: 100%;
}
div.stButton > button {
    background-color: #1a73e8 !important;
    color: #fff !important;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px 24px;
    border: none !important;
    transition: background 0.2s;
}
div.stButton > button:hover {
    background-color: #174ea6 !important;
}
/* Download button: white background, blue text */
.stDownloadButton button {
    background-color: #fff !important;
    color: #174ea6 !important;
    border: 1.7px solid #174ea6 !important;
    font-weight: 600;
    border-radius: 9px !important;
    transition: background 0.19s, color 0.19s;
    box-shadow: 0 1px 6px rgba(23,78,166,0.07);
}
.stDownloadButton button:hover {
    background-color: #174ea6 !important;
    color: #fff !important;
    border: 1.7px solid #174ea6 !important;
}
.streamlit-expanderHeader {
    font-weight: 900 !important;
    font-size: 1.18em !important;
    color: #174ea6 !important;
    font-family: 'Lato', sans-serif !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
                <style>
        .css-1v3fvcr {
            color: black !important;
        }
        .css-1p2bdos {
            color: black !important;
        }
        .stSlider>div>div>div>div {
            color: black !important;
        }
    </style>
            
</style>
            
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Home", "Map Viewer", "AI Site Analysis",
    "Community Pulse", "Resident Matching", "Post-Site Feedback"
])

# --------- HOME TAB ---------
# --------- HOME TAB ---------
with tab1:
    st.markdown('<div style="text-align:center;"><span class="heading-animate">San Jose EIH Site Explorer</span></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.image("HomeFinderLogo.png", width=64)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;">
        <span style="font-family: 'Roboto Slab', serif; font-size: 1.27em; font-weight: 700; color: #222;">
            Welcome to the <span style="color:#174ea6;">Emergency Interim Housing (EIH) Site Explorer</span> powered by AI.
        </span>
    </div>
    <br>
    <div style="background: #f6fafd; border-radius: 12px; padding: 28px 16px 20px 30px; border: 1.5px solid #e4e7ec; max-width: 480px; margin: 0 auto 0 auto;">
        <span style="font-family: 'Lato', sans-serif; font-weight:700; color: #111; font-size: 1.09em;">
            What you can do here:
        </span>
        <ul style="margin-top: 12px; margin-bottom: 0; color: #333; font-size:1.02em; line-height: 2;">
            <li>View candidate EIH sites on a map</li>
            <li>Analyze infrastructure & community sentiment</li>
            <li>Use AI to recommend optimal sites for development</li>
            <li>Match individuals to appropriate sites</li>
            <li>Track performance feedback post-deployment</li>
        </ul>
    </div>
                
    """, unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ---------- BEAUTIFUL DATA PREVIEW ----------
    import pandas as pd
    import streamlit as st

    # --- Bold expander headers CSS (add once at top of script) ---
    st.markdown("""
    <style>
    .streamlit-expanderHeader {
        font-weight: 700 !important;
        font-size: 1.08em !important;
        color: #174ea6 !important;
        font-family: 'Lato', sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)

    df = pd.read_csv('san_jose_eih_sites.csv')

    # Convert sentiment to percent if needed
    if "sentiment_score" in df.columns and df["sentiment_score"].max() <= 1.5:
        df["sentiment_score"] = df["sentiment_score"] * 100

    # Proximity word helpers
    def proximity_word(meters):
        try:
            m = float(meters)
        except Exception:
            return ''
        if m <= 300:
            return "Close"
        if m <= 700:
            return "Nearby"
        return "Far"

    if "proximity_to_library" in df.columns:
        df["Library Proximity"] = df["proximity_to_library"].apply(proximity_word)
    if "proximity_to_hospital" in df.columns:
        df["Hospital Proximity"] = df["proximity_to_hospital"].apply(proximity_word)

    pretty_names = {
        "site_name": "Site Name",
        "zoning": "Zoning",
        "sentiment_score": "Community Sentiment Score (%)",
        "Library Proximity": "Library Proximity",
        "Hospital Proximity": "Hospital Proximity",
        "proximity_to_library": "Library (m)",
        "proximity_to_hospital": "Hospital (m)",
    }
    df_display = df.rename(columns=pretty_names)

    default_cols = [c for c in [
        "Site Name", "Zoning", "Community Sentiment Score (%)",
        "Library Proximity", "Hospital Proximity"
    ] if c in df_display.columns]
    if not default_cols:
        default_cols = list(df_display.columns)
    cols = st.multiselect("Columns to display:", df_display.columns, default=default_cols)

    # Style functions
    def color_sentiment(val):
        try:
            v = float(val)
        except Exception:
            return ''
        if v >= 75:
            return 'color: #19a53f; font-weight: bold;'
        if v >= 50:
            return 'color: #f39c12; font-weight: bold;'
        return 'color: #e05d48; font-weight: bold;'

    def bold_proximity(val):
        return 'font-weight: bold;'

    styled = (
        df_display[cols]
        .style
        .set_properties(**{'background-color': '#fff', 'color': '#111'})
        .format({
            "Community Sentiment Score (%)": "{:.1f}",
            "Library (m)": "{:.0f}",
            "Hospital (m)": "{:.0f}",
        })
        .applymap(
            color_sentiment,
            subset=["Community Sentiment Score (%)"] if "Community Sentiment Score (%)" in cols else []
        )
        .applymap(
            bold_proximity,
            subset=[col for col in ["Library Proximity", "Hospital Proximity"] if col in cols]
        )
    )

    # Column descriptions (open by default)
    with st.expander("Column Descriptions", expanded=True):
        st.markdown("""
        <div style="
            background: #f5f8fa;
            border-left: 4px solid #a7c7e7;
            border-radius: 5px;
            color: #23272e;
            font-size: 1.05em;
            padding: 0.9em 1.2em 0.9em 1.2em;
            margin-bottom: 8px;
        ">
        <ul style="margin-left: 1.2em;">
        <li><b>Site Name:</b> <span style="color:#174ea6;">Official name of the candidate EIH site</span></li>
        <li><b>Zoning:</b> <span style="color:#174ea6;">Zoning type (e.g., residential, commercial)</span></li>
        <li>
            <b>Community Sentiment Score (%):</b>
            <span style="color:#174ea6;">Higher is better (0–100)</span><br>
            <span style="font-size: 0.98em;">
            <b style="color:#19a53f;">&#9679; Green: Excellent (75+)</b> &nbsp; | &nbsp;
            <b style="color:#f39c12;">&#9679; Orange: Average (50–74)</b> &nbsp; | &nbsp;
            <b style="color:#e05d48;">&#9679; Red: Needs improvement (&lt;50)</b>
            </span>
        </li>
        <li>
            <b>Library Proximity:</b>
            <span style="color:#174ea6;">"Close" (≤ 300 m), "Nearby" (301–700 m), "Far" (&gt; 700 m)</span>
        </li>
        <li>
            <b>Hospital Proximity:</b>
            <span style="color:#174ea6;">"Close" (≤ 300 m), "Nearby" (301–700 m), "Far" (&gt; 700 m)</span>
        </li>
        <li><b>Library (m):</b> <span style="color:#174ea6;">Distance to the nearest library in meters</span></li>
        <li><b>Hospital (m):</b> <span style="color:#174ea6;">Distance to the nearest hospital in meters</span></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # Data table (open by default)
    with st.expander("Preview Raw Data Table", expanded=True):
        st.dataframe(styled, use_container_width=True)
        st.download_button("Download CSV", df_display[cols].to_csv(index=False), "EIH_sites.csv")
        file_path = 'san_jose_eih_sites.csv'
        if os.path.exists(file_path):
            modified_time = os.path.getmtime(file_path)
            st.markdown(
                f"<div class='last-updated-box'>Dataset last updated: {datetime.datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')}</div>",
                unsafe_allow_html=True)
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# --------- MAP VIEWER TAB ---------

with tab2:
    st.header("Interactive Map of Candidate EIH Sites")
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Add a helpful tip for users
    st.markdown("""
    <div style="background-color: #f5f8fa; padding: 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); font-size: 1.1em; color: #111;">
        <b>Tip:</b> Hover over a site on the map to see more information about the <i>Emergency Interim Housing (EIH)</i> sites, including proximity to <b>libraries</b> and <b>hospitals</b>, the <b>community sentiment score</b>, and a recommendation of the site's <b>suitability</b> for development.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    data = pd.read_csv('san_jose_eih_sites.csv')  # Load your CSV
    data = data.dropna(subset=['latitude', 'longitude'])  # Drop rows without location data


    # Weighting for IIS calculation
    weight_sentiment = 0.33
    weight_library = 0.33
    weight_hospital = 0.34

    # Function to calculate Infrastructure Influence Score (IIS)
    def calculate_iis(row, w_sent, w_lib, w_hosp):
        norm_sent = row['sentiment_score'] / 100
        norm_lib = 1 - min(row['proximity_to_library'], 1000) / 1000
        norm_hosp = 1 - min(row['proximity_to_hospital'], 1000) / 1000
        return (w_sent * norm_sent) + (w_lib * norm_lib) + (w_hosp * norm_hosp)

    data['iis_score'] = data.apply(lambda row: calculate_iis(row, weight_sentiment, weight_library, weight_hospital), axis=1)

    # Function to tag site suitability
    def tag_site(score):
        if score >= 0.75:
            return "Ideal"
        elif score >= 0.5:
            return "Moderate"
        else:
            return "Poor"
    data['suitability_tag'] = data['iis_score'].apply(tag_site)

    # Map layer for scatter plot
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=data,
        get_position='[longitude, latitude]',
        get_radius=200,
        get_color='[200, 30, 0, 160]',
        pickable=True
    )

    # Initial view of the map
    view_state = pdk.ViewState(
        latitude=data['latitude'].mean(),
        longitude=data['longitude'].mean(),
        zoom=11,
        pitch=0
    )

    # Tooltip with image URL
    tooltip_text = """
    <b>Site:</b> {site_name}<br>
    <b>Library:</b> {proximity_to_library}m<br>
    <b>Hospital:</b> {proximity_to_hospital}m<br>
    <b>Sentiment Score:</b> {sentiment_score}<br>
    <b>Suitability:</b> {suitability_tag}<br>
    <img src="{image_url}" width="200">
    """

    # Pydeck deck with the map and tooltip
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"html": tooltip_text, "style": {"color": "black"}}
    )

    st.pydeck_chart(r)  # Display the map
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# --------- AI SITE ANALYSIS TAB -----------------------------------------------------------------

st.markdown("""
    <style>
    /* Make all text and numbers in sliders black */
    .stSlider label, .stSlider span, .stSlider div[role="slider"], .stSlider div[role="presentation"] span {
        color: #111 !important;
        font-weight: 600 !important;
    }
    .stSlider .css-1lv4j2l { color: #111 !important; }
    .stSlider .css-14xtw13 { color: #111 !important; }
    .stSlider .css-1y4p8pa { color: #111 !important; }
    </style>
""", unsafe_allow_html=True)


with tab3:
    st.header("AI-Powered Site Analysis")
    # --- User-friendly explanation under the title ---
    st.markdown(
        "<div style='color:#222; font-size:1.08em;'>"
        "Use this tool to evaluate candidate sites for Emergency Interim Housing (EIH) based on their proximity to key infrastructure and resident sentiment. "
        "Adjust the weights to prioritize the factors that matter most to your analysis. The AI will help interpret the data and provide recommendations.</div>",
        unsafe_allow_html=True
    )
    st.markdown("Select one or more candidate sites below for analysis:")

    selected = st.multiselect("Choose sites to analyze:", options=data['site_name'].unique())

    with st.expander("About this AI Analysis"):
        st.markdown("""
        - The recommendations provided here are generated using **OpenAI's GPT model**.
        - They are based on proximity metrics and sentiment scores in the uploaded dataset.
        - Please note these insights are **not absolute facts**—they are generated based on patterns in the data and may reflect inherent biases.
        - Use them as **a guide**, not a final decision-making tool.
        """)

    # ---- Make slider text black (labels + numbers) ----
    st.markdown("""
        <style>
        /* Make slider label and value text black */
        .stSlider > div[data-baseweb="slider"] span {
            color: #111 !important;
            font-weight: 600 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("**Customize Weighting (optional)**")
    weight_sentiment = st.slider("Weight for Sentiment Score", 0.0, 1.0, 0.33)
    weight_library = st.slider("Weight for Proximity to Library", 0.0, 1.0, 0.33)
    weight_hospital = st.slider("Weight for Proximity to Hospital", 0.0, 1.0, 0.34)

    if st.button("Run AI Analysis") and selected:
        selected_data = data[data['site_name'].isin(selected)].copy()
        selected_data['iis_score'] = selected_data.apply(
            lambda row: calculate_iis(row, weight_sentiment, weight_library, weight_hospital), axis=1
        )

        site_summary = ""
        for _, row in selected_data.iterrows():
            site_summary += (
                f"- {row['site_name']}: Library {row['proximity_to_library']}m, "
                f"Hospital {row['proximity_to_hospital']}m, "
                f"Sentiment {row['sentiment_score']}, IIS Score: {row['iis_score']:.2f}\n"
            )

        weight_summary = f"""
User-defined weights:
- Sentiment: {weight_sentiment}
- Library Proximity: {weight_library}
- Hospital Proximity: {weight_hospital}
"""

        prompt = f"""You are a policy analyst. Analyze the following Emergency Interim Housing (EIH) candidate sites based on proximity to infrastructure and resident sentiment. Recommend which sites seem more viable and why:

{site_summary}

User-defined priority:
{weight_summary}

Be specific in your reasoning based on the numbers given."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful urban planning assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        st.subheader("AI Recommendation")
        st.markdown(
            f"""
            <div style="
                background: #f5f8fa;
                border-left: 5px solid #2a73cc;
                border-radius: 10px;
                padding: 20px 18px 14px 18px;
                margin-top: 8px;
                margin-bottom: 12px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.07);
            ">
                <div style="color:#222; font-size:1.10em; line-height:1.6;">
                    {response.choices[0].message.content.replace('\n', '<br>')}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# --------- COMMUNITY PULSE TAB ---------
with tab4:
    st.header("Community Sentiment & Infrastructure Pulse")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("""
    This dashboard provides a snapshot of community sentiment and infrastructure access across candidate sites for Emergency Interim Housing (EIH) in the city.
    
    **How to read these tables:**
    - **Top 3 Sites by Sentiment:**  
      These are the locations with the most positive overall community sentiment, reflecting where local residents and stakeholders have expressed the highest levels of support or optimism regarding the potential EIH project.
    
    - **Bottom 3 Sites by Sentiment:**  
      These sites have the lowest sentiment scores, indicating areas where community feedback may be more cautious, negative, or resistant. They can highlight locations that may need additional engagement or outreach.
    
    - **Average Infrastructure Influence Score (IIS):**  
      This metric summarizes how well sites across the city are positioned in terms of proximity and access to vital resources (like libraries, hospitals, and transit). A higher IIS suggests better overall infrastructure support for residents.
    """)
    
    st.subheader("Top 3 Sites by Sentiment")
    st.dataframe(data.sort_values(by="sentiment_score", ascending=False).head(3))

    st.subheader("Bottom 3 Sites by Sentiment")
    st.dataframe(data.sort_values(by="sentiment_score", ascending=True).head(3))

    st.subheader("Average Infrastructure Influence Score (IIS)")

    # Custom CSS to style the metric label and value as black
    st.markdown("""
        <style>
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
            color: #111 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.metric("Citywide Avg IIS", f"{data['iis_score'].mean():.2f}")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# --------- RESIDENT MATCHING TAB ---------
with tab5:
    # Custom CSS: Make all main text black in this tab
    st.markdown("""
        <style>
            /* Make all text elements black inside this tab */
            .stApp, .stMarkdown, .stTextInput label, .stTextArea label, .stSubheader, .stInfo {
                color: #111 !important;
            }
            /* st.info uses a blue background—set text to black for contrast */
            [data-testid="stAlertContent"] {
                color: #111 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.header("Resident-to-Site Matching")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("This AI-powered tool helps caseworkers or planners assign individuals to the most suitable EIH site based on personal needs.")

    name = st.text_input("Resident Name")
    needs = st.text_area("Describe the resident's needs, preferences, or constraints:")

    if st.button("Match to Site") and name and needs:
        match_prompt = f"""You are a social housing advisor. Based on the following resident profile and available EIH site data, recommend the best site for placement and explain your reasoning:

Resident Info:
{name}
{needs}

Site Options:
{data[['site_name', 'proximity_to_library', 'proximity_to_hospital', 'sentiment_score']].to_string(index=False)}

Be thoughtful, empathetic, and specific."""

        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        match_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You match residents with appropriate temporary housing based on their needs."},
                {"role": "user", "content": match_prompt}
            ]
        )
        st.subheader("Best Match Recommendation")
        st.info(match_response.choices[0].message.content)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# --------- FEEDBACK TAB ---------
with tab6:
    st.header("Post-Site Feedback Tracker")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("Use this section to log performance data or feedback about EIH sites post-deployment for future learning.")

    feedback_site = st.selectbox("Select Site for Feedback:", data['site_name'].unique())
    feedback = st.text_area("Enter qualitative or performance-based feedback:")

    if st.button("Save Feedback"):
        with open("site_feedback_log.txt", "a") as f:
            f.write(f"\n{datetime.datetime.now()} | Site: {feedback_site} | Feedback: {feedback}")
        st.success("Feedback logged. Thank you!")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)