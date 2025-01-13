import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('crop_production_model.pkl')  # Replace with your actual model file

# Page configuration
st.set_page_config(
    page_title="Smart Crop Predictor",
    page_icon="🌾",
    layout="wide",
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        font-size: 42px;
        color: #2E7D32;
        font-weight: bold;
        text-align: center;
    }
    .sub-title {
        font-size: 20px;
        color: #555555;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #2E7D32;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='main-title'>Smart Crop Predictor 🌱</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Predict the best crop yield production for your region and conditions.</p>", unsafe_allow_html=True)

# Mapping of states to districts
state_district_map = {
    "Andaman and Nicobar Islands": ["Nicobar", "North and Middle Andaman", "South Andaman"],
    "Andhra Pradesh": ["Anantapur", "Chittoor", "East Godavari", "Guntur", "Kadapa", "Krishna", "Kurnool"],
    "Arunachal Pradesh": ["Anjaw", "Changlang", "Dibang Valley", "East Kameng", "East Siang"],
    "Assam": ["Baksa", "Barpeta", "Biswanath", "Bongaigaon", "Cachar"],
    "Bihar": ["Araria", "Arwal", "Aurangabad", "Banka", "Begusarai"],
    "Chandigarh": ["Chandigarh"],
    "Chhattisgarh": ["Balod", "Baloda Bazar", "Balrampur", "Bastar", "Bemetara"],
    # Add more states and districts here
}

# Input fields for State, District, Season, and Year
st.header("Location and Season Details")
col1, col2 = st.columns(2)
with col1:
    state = st.selectbox("State", list(state_district_map.keys()))
    district = st.selectbox("District", state_district_map[state])
with col2:
    season = st.selectbox("Season", ["Kharif", "Rabi", "Whole Year", "Summer", "Winter", "Autumn"])
    crop_year = st.number_input("Crop Year", min_value=2000, max_value=2025, value=2021, step=1)

# Environmental parameters
st.header("Environmental Parameters")
col3, col4, col5, col6 = st.columns(4)
with col3:
    temperature = st.slider('Temperature (°C)', 0.0, 50.0, 25.0, step=0.1)
with col4:
    humidity = st.slider('Humidity (%)', 0.0, 100.0, 60.0, step=0.1)
with col5:
    soil_moisture = st.slider('Soil Moisture (%)', 0.0, 100.0, 50.0, step=0.1)
with col6:
    area = st.number_input('Area (acres)', min_value=0.0, max_value=1000.0, value=1.0, step=0.1)

# Predict button
if st.button('Predict'):
    # Prepare input data for the model
    input_data = np.array([[temperature, humidity, soil_moisture, area]])

    # Make a prediction
    prediction = model.predict(input_data)

    # Display the result
    st.markdown("### Prediction Result")
    st.success(f"The recommended crop for your conditions is: **{prediction[0]}** tons 🌾")
    # st.balloons()
# Footer
st.markdown("<p style='text-align:center; color:#888888; margin-top:50px;'>🌱 Powered by Smart Agriculture Solutions</p>", unsafe_allow_html=True)
