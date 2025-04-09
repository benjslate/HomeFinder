import os
import pandas as pd
import streamlit as st
import pydeck as pdk
import openai

# Load dataset
data = pd.read_csv('san_jose_eih_sites.csv')

# Set OpenAI key from environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Streamlit app title
st.title("San Jose EIH Site Explorer")

# Show preview of dataset
with st.expander("Preview Data"):
    st.write(data.head())

# Drop rows with missing coordinates
data = data.dropna(subset=['latitude', 'longitude'])

# Create interactive map using pydeck
st.subheader("📍 Interactive Map of Candidate EIH Sites")

layer = pdk.Layer(
    'ScatterplotLayer',
    data=data,
    get_position='[longitude, latitude]',
    get_radius=200,
    get_color='[200, 30, 0, 160]',
    pickable=True
)

view_state = pdk.ViewState(
    latitude=data['latitude'].mean(),
    longitude=data['longitude'].mean(),
    zoom=11,
    pitch=0
)

r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "Site: {site_name}\nLibrary Proximity: {proximity_to_library}m\nHospital Proximity: {proximity_to_hospital}m\nSentiment: {sentiment_score}"}
)

st.pydeck_chart(r)

# Optional: Select specific rows to analyze
selected = st.multiselect("Select sites for analysis:", options=data['site_name'].unique())

# Generate AI summary
if st.button("🔍 Analyze with AI") and len(selected) > 0:
    selected_data = data[data['site_name'].isin(selected)]
    
    # Create a summary string for AI
    site_summary = ""
    for index, row in selected_data.iterrows():
        site_summary += f"- {row['site_name']}: Library {row['proximity_to_library']}m, Hospital {row['proximity_to_hospital']}m, Sentiment {row['sentiment_score']}\n"

    prompt = f"""You are a policy analyst. Analyze the following EIH candidate sites based on proximity to infrastructure and resident sentiment. Recommend which seem more viable.

{site_summary}

Give reasoning based on proximity and sentiment.
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful urban planning assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    st.markdown("### 🤖 AI Analysis")
    st.write(response['choices'][0]['message']['content'])

