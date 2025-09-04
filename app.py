import streamlit as st
import pandas as pd
import joblib
from prophet import Prophet
from prophet.plot import plot_plotly

# ========================
# Load dataset / model
# ========================
df = pd.read_csv('data/your_dataset.csv')
xgb_model = joblib.load('model/xgb_pipeline.pkl')  # previously saved pipeline

# ========================
# Streamlit App Layout
# ========================
st.title("Car Resale Price & Trend Predictor")
st.write("Predict resale price today + future market trend for European cars.")

# -------------------------
# User Inputs for ML
# -------------------------
st.header("Predict Resale Price for Your Car")
make = st.selectbox("Make", sorted(df['make'].unique()))
model_car = st.selectbox("Model", sorted(df[df['make']==make]['model'].unique()))
fuel = st.selectbox("Fuel Type", sorted(df['fuel'].unique()))
gear = st.selectbox("Gear Type", sorted(df['gear'].unique()))
mileage = st.number_input("Mileage (km)", min_value=0)
hp = st.number_input("Horsepower (hp)", min_value=0)
year = st.number_input("Year of Manufacture", min_value=1980, max_value=2025)

car_age = 2025 - year  # or use current year dynamically

input_df = pd.DataFrame({
    'make':[make],
    'model':[model_car],
    'fuel':[fuel],
    'gear':[gear],
    'mileage':[mileage],
    'hp':[hp],
    'car_age':[car_age]
})

if st.button("Predict Resale Price"):
    predicted_price = xgb_model.predict(input_df)[0]
    st.success(f"Estimated resale price today: €{predicted_price:,.0f}")

# -------------------------
# Brand Forecast
# -------------------------
st.header("Market Trend Forecast for Your Brand")
forecast_years = 3  # next 3 years
df_brand = df[df['make']==make].groupby('year')['price'].mean().reset_index()
df_brand = df_brand.rename(columns={'year':'ds','price':'y'})
df_brand['ds'] = pd.to_datetime(df_brand['ds'], format='%Y')

if df_brand.shape[0] >= 2:
    prophet_model = Prophet()
    prophet_model.fit(df_brand)
    future = prophet_model.make_future_dataframe(periods=forecast_years, freq='YE')
    forecast = prophet_model.predict(future)

    st.plotly_chart(plot_plotly(prophet_model, forecast))
else:
    st.warning("Not enough historical data for this brand to forecast.")
