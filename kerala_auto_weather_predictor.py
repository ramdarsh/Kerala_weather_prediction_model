import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import datetime

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Kerala Smart Weather Predictor", page_icon="🌦️", layout="wide")

# -------------------- BACKGROUND & STYLING --------------------
st.markdown("""
<style>
/* ---------- Background Image ---------- */
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.pexels.com/photos/1118873/pexels-photo-1118873.jpeg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white;
}

/* ---------- Global Text ---------- */
body, p, div, span {
    font-family: 'Segoe UI', sans-serif !important;
}

/* ---------- Transparent Containers ---------- */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 20px;
}

/* ---------- Titles ---------- */
.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.subtitle {
    text-align: center;
    font-size: 1.2rem;
    color: #f5f5f5;
    margin-bottom: 1.5rem;
}

/* ---------- Cards ---------- */
.glass-card {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(10px);
}

/* ---------- Buttons ---------- */
div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #11998e, #38ef7d);
    color: white;
    border: none;
    padding: 0.8rem 1.3rem;
    border-radius: 12px;
    font-size: 1.1rem;
    font-weight: 600;
    transition: 0.3s ease;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #38ef7d, #11998e);
    transform: scale(1.02);
}

/* ---------- Prediction Card ---------- */
.prediction-card {
    background: rgba(255, 255, 255, 0.85);
    color: #00332e;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 4px 25px rgba(0, 0, 0, 0.25);
    margin-top: 2rem;
}

/* ---------- Side Watermark ---------- */
.footer-credit {
    text-align: center;
    font-size: 0.9rem;
    color: rgba(255,255,255,0.7);
    margin-top: 3rem;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown('<div class="main-title">Kerala Smart Weather Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict the future weather in your Kerala district using AI 🌦️</div>', unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    return pd.read_csv("kerala_district_weather_20000.csv")

data = load_data()
st.success(f"✅ Loaded {len(data)} Kerala weather records successfully!")

# -------------------- ENCODING --------------------
le_weather = LabelEncoder()
le_monsoon = LabelEncoder()
le_district = LabelEncoder()

data["Weather"] = le_weather.fit_transform(data["Weather"])
data["MonsoonPhase"] = le_monsoon.fit_transform(data["MonsoonPhase"])
data["District"] = le_district.fit_transform(data["District"])

# -------------------- TRAIN MODEL --------------------
X = data.drop("Weather", axis=1)
y = data["Weather"]
model = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
model.fit(X, y)

# -------------------- HELPER FUNCTIONS --------------------
def get_monsoon_phase(day):
    if 150 <= day <= 270:
        return "SouthWest Monsoon"
    elif 271 <= day <= 330:
        return "NorthEast Monsoon"
    elif 90 <= day < 150:
        return "Pre-Monsoon"
    else:
        return "Post-Monsoon"

def generate_district_weather(district, day):
    coastal = ["Thiruvananthapuram", "Kollam", "Alappuzha", "Ernakulam", "Kozhikode", "Kannur", "Kasaragod"]
    highland = ["Idukki", "Wayanad"]
    midland = ["Kottayam", "Pathanamthitta", "Thrissur", "Palakkad", "Malappuram"]

    monsoon = get_monsoon_phase(day)
    base_temp, base_hum, base_precip, base_cloud, base_wind = 29, 80, 5, 60, 10

    if district in coastal:
        base_temp -= 1; base_hum += 5; base_wind += 2
    elif district in highland:
        base_temp -= 3; base_hum -= 2; base_precip += 5
    elif district in midland:
        base_temp += 0.5; base_hum += 1

    if monsoon == "SouthWest Monsoon":
        base_precip += 10; base_cloud += 15; base_hum += 5
    elif monsoon == "NorthEast Monsoon":
        base_precip += 6; base_cloud += 10
    elif monsoon == "Pre-Monsoon":
        base_temp += 2; base_hum -= 5; base_precip -= 3
    else:
        base_precip -= 2; base_temp -= 1

    temperature = np.random.normal(base_temp, 1.5)
    humidity = np.random.normal(base_hum, 5)
    pressure = np.random.normal(1010, 5)
    wind_speed = np.random.normal(base_wind, 2)
    precipitation = max(0, np.random.normal(base_precip, 4))
    cloud_cover = np.clip(np.random.normal(base_cloud, 10), 0, 100)

    return {
        "Temperature": round(temperature, 2),
        "Humidity": round(humidity, 2),
        "Pressure": round(pressure, 2),
        "WindSpeed": round(wind_speed, 2),
        "Precipitation": round(precipitation, 2),
        "CloudCover": round(cloud_cover, 2),
        "MonsoonPhase": monsoon
    }

# -------------------- USER INPUT --------------------
with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🌎 Enter Prediction Details")
    districts = le_district.classes_
    col1, col2 = st.columns(2)
    with col1:
        selected_district = st.selectbox("🏙️ Select District", districts)
    with col2:
        future_date = st.date_input("📅 Select Date", datetime.date.today())
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- PREDICTION --------------------
if st.button("🔮 Predict Weather"):
    day_of_year = future_date.timetuple().tm_yday
    weather_data = generate_district_weather(selected_district, day_of_year)
    monsoon_value = le_monsoon.transform([weather_data["MonsoonPhase"]])[0]
    district_value = le_district.transform([selected_district])[0]

    new_data = pd.DataFrame({
        "District": [district_value],
        "Temperature": [weather_data["Temperature"]],
        "Humidity": [weather_data["Humidity"]],
        "Pressure": [weather_data["Pressure"]],
        "WindSpeed": [weather_data["WindSpeed"]],
        "Precipitation": [weather_data["Precipitation"]],
        "CloudCover": [weather_data["CloudCover"]],
        "DayOfYear": [day_of_year],
        "MonsoonPhase": [monsoon_value]
    })

    prediction = model.predict(new_data)
    predicted_label = le_weather.inverse_transform(prediction)[0]

    st.markdown(f"""
    <div class="prediction-card">
        <h3>🌤️ Weather Forecast</h3>
        <p><b>🏙️ District:</b> {selected_district}</p>
        <p><b>📅 Date:</b> {future_date.strftime('%d %B %Y')} (Day {day_of_year})</p>
        <p><b>🌀 Monsoon Phase:</b> {weather_data['MonsoonPhase']}</p>
        <hr>
        <h4>🌈 Predicted Weather: <span style="color:#00796b;">{predicted_label}</span></h4>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🌡️ Generated Weather Features")
    st.json(weather_data)

# -------------------- FOOTER --------------------
st.markdown('<div class="footer-credit">🌿 App by RMS | Designed with ❤️ using Streamlit</div>', unsafe_allow_html=True)

