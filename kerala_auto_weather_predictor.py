import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import datetime

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Kerala Smart Weather Predictor", page_icon="🌦️", layout="centered")

# -------------------- CUSTOM STYLING --------------------
st.markdown("""
    <style>
    /* Background gradient */
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.pexels.com/photos/1118873/pexels-photo-1118873.jpeg?_gl=1*p98tuv*_ga*NzQ4Mjk5OTI0LjE3NjI0MDYzMjc.*_ga_8JE65Q40S6*czE3NjI0MDYzMjYkbzEkZzAkdDE3NjI0MDYzMjckajU5JGwwJGgw");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
""", unsafe_allow_html=True)


    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(255,255,255,0.8);
        backdrop-filter: blur(10px);
        border-right: 2px solid rgba(255,255,255,0.3);
    }

    /* Title */
    .title {
        text-align: center;
        font-size: 2.4rem;
        color: #004d40;
        font-weight: 700;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
        margin-bottom: 0.2rem;
    }

    /* Info cards */
    .info-card {
        padding: 1.2rem;
        background-color: rgba(255,255,255,0.8);
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        margin-bottom: 1.5rem;
    }

    /* Prediction result */
    .prediction {
        background-color: #ffffffcc;
        border-left: 6px solid #009688;
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1.2rem;
        box-shadow: 0px 3px 8px rgba(0,0,0,0.1);
        font-size: 1.2rem;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #43cea2, #185a9d);
        color: white;
        border: none;
        padding: 0.7rem 1.3rem;
        border-radius: 12px;
        font-size: 1rem;
        font-weight: 600;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #185a9d, #43cea2);
        transform: scale(1.03);
    }

    /* Footer or side watermark */
    .side-text {
        position: fixed;
        top: 50%;
        right: -35px;
        transform: rotate(-90deg);
        transform-origin: right top;
        font-size: 0.9rem;
        font-weight: 600;
        color: rgba(0, 77, 64, 0.6);
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    </style>
    <div class="side-text">🌿 App by RMS 🌿</div>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown('<div class="title">🌴 Kerala Smart Weather Predictor 🌦️</div>', unsafe_allow_html=True)
st.markdown("""
### _Enter your district and date to get a smart weather forecast for Kerala!_  
Model trained on **20,000+ Kerala weather samples** 🌤️  
""")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    return pd.read_csv("kerala_district_weather_20000.csv")

data = load_data()
st.success(f"✅ Loaded {len(data)} Kerala weather records successfully!")

# -------------------- ENCODE --------------------
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
st.markdown('<div class="info-card">', unsafe_allow_html=True)
st.subheader("🌏 Enter Prediction Details")
districts = le_district.classes_
selected_district = st.selectbox("🏙️ Select District", districts)
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
        <div class='prediction'>
        🌆 <b>District:</b> {selected_district}  
        📅 <b>Date:</b> {future_date.strftime('%d %B %Y')} (Day {day_of_year})  
        🌀 <b>Monsoon Phase:</b> {weather_data['MonsoonPhase']}  
        <br><br>
        🌈 <b>Predicted Weather:</b> <span style='font-size:1.3rem;color:#00796b;'>{predicted_label}</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🌡️ Generated Weather Features")
    st.json(weather_data)



