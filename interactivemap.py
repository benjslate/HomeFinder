import os
import pandas as pd
import streamlit as st
import pydeck as pdk
import openai

# ✅ Initialize OpenAI client using v1.0+ style
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 📊 Load dataset
data = pd.read_csv('san_jose_eih_sites.csv')

# 🏷️ Title
st.title("San Jose EIH Site Explorer")

# 📋 Show preview of data
with st.expander("Preview Data"):
    st.dataframe(data)

# 🧹 Clean data
data = data.dropna(subset=['latitude', 'longitude'])

# 🗺️ Interactive map with pydeck
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
    tooltip={
        "text": "Site: {site_name}\nLibrary Proximity: {proximity_to_library}m\nHospital Proximity: {proximity_to_hospital}m\nSentiment: {sentiment_score}"
    }
)

st.pydeck_chart(r)

# 🎯 Select sites for AI analysis
selected = st.multiselect("Select sites for analysis:", options=data['site_name'].unique())

if st.button("🔍 Analyze with AI") and selected:
    selected_data = data[data['site_name'].isin(selected)]

    site_summary = ""
    for _, row in selected_data.iterrows():
        site_summary += (
            f"- {row['site_name']}: Library {row['proximity_to_library']}m, "
            f"Hospital {row['proximity_to_hospital']}m, "
            f"Sentiment {row['sentiment_score']}\n"
        )

    prompt = f"""You are a policy analyst. Analyze the following Emergency Interim Housing (EIH) candidate sites based on proximity to infrastructure and resident sentiment. Recommend which sites seem more viable and why:

{site_summary}

Be specific in your reasoning based on the numbers given.
"""

    # ✅ NEW OpenAI SDK chat completion call
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful urban planning assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # 🧠 Show result
    st.markdown("### 🤖 AI Analysis")
    st.write(response.choices[0].message.content)


