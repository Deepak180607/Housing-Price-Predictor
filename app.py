from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "housing_model.joblib"
TYPICAL_ERROR = 47_000  # approx. CV/test RMSE from your notebook; update with your test RMSE

st.set_page_config(page_title="California House Price Predictor", page_icon="🏠", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("🏠 California House Price Predictor")
st.caption(
    "Predicts the **median house value of a census block group** using a tuned "
    "HistGradientBoosting model trained on the California Housing dataset."
)

if not MODEL_PATH.exists():
    st.error(
        "`housing_model.joblib` not found. Run `joblib.dump(grid.best_estimator_, "
        "'housing_model.joblib')` in your notebook and put the file next to app.py."
    )
    st.stop()

model = load_model()

# ---------------- Inputs ----------------
st.sidebar.header("Neighbourhood details")

median_income = st.sidebar.slider(
    "Median income (in $10,000s)", 0.5, 15.0, 3.5, 0.1,
    help="3.5 means $35,000 median household income in the area.",
)
housing_median_age = st.sidebar.slider("Median house age (years)", 1, 52, 28)
ocean_proximity = st.sidebar.selectbox(
    "Ocean proximity", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
)

st.sidebar.subheader("Location")
latitude = st.sidebar.number_input("Latitude", 32.5, 42.0, 34.05, 0.01, format="%.2f")
longitude = st.sidebar.number_input("Longitude", -124.4, -114.3, -118.25, 0.01, format="%.2f")

st.sidebar.subheader("Block group size")
total_rooms = st.sidebar.number_input("Total rooms", 1, 40_000, 2_100, 50)
total_bedrooms = st.sidebar.number_input("Total bedrooms", 1, 7_000, 430, 10)
population = st.sidebar.number_input("Population", 1, 36_000, 1_400, 50)
households = st.sidebar.number_input("Households", 1, 6_000, 500, 10)

# ---------------- Validation ----------------
problems = []
if total_bedrooms > total_rooms:
    problems.append("Total bedrooms cannot exceed total rooms.")
if households > population:
    problems.append("Households cannot exceed population.")

# ---------------- Prediction ----------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Prediction")
    for p in problems:
        st.warning(p)

    if st.button("Predict price", type="primary", disabled=bool(problems)):
        X_new = pd.DataFrame([{
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean_proximity,
        }])

        pred = float(model.predict(X_new)[0])
        st.metric("Estimated median house value", f"${pred:,.0f}")
        st.write(
            f"Typical error is around ±${TYPICAL_ERROR:,.0f}, so a reasonable range is "
            f"**${max(pred - TYPICAL_ERROR, 0):,.0f} – ${pred + TYPICAL_ERROR:,.0f}**."
        )
        if pred >= 480_000:
            st.info("The training data is capped at $500,001, so values near this are underestimates.")

        with st.expander("Inputs sent to the model"):
            st.dataframe(X_new, width="stretch")

with right:
    st.subheader("Location")
    st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=6)

st.divider()
st.caption(
    "⚠️ Based on 1990 California census data, so prices are far below today's market. "
    "This is a learning project, not financial advice."
)
