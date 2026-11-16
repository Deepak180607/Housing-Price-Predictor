from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
MODEL_PATH = BASE / "housing_model.joblib"
DATA_PATH = BASE / "housing.csv"
RESULTS_PATH = BASE / "test_results.csv"          # made by export_artifacts.py
IMPORTANCE_PATH = BASE / "feature_importance.csv"  # made by export_artifacts.py

FALLBACK_ERROR = 47_000   # used only if test_results.csv is missing
PRICE_CAP = 500_001
MAP_CENTER = dict(lat=37.2, lon=-119.5)

st.set_page_config(page_title="California House Price Dashboard", page_icon="🏠", layout="wide")


# ----------------------------------------------------------------- loaders
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


# ----------------------------------------------------------------- helpers
def make_map(data, color, scale, title, midpoint=None, range_color=None,
             hover=None, point=None, height=520):
    """Scatter map of California coloured by `color`. Optional red marker for `point`."""
    fig = px.scatter_map(
        data, lat="latitude", lon="longitude", color=color,
        color_continuous_scale=scale, color_continuous_midpoint=midpoint,
        range_color=range_color, hover_data=hover, opacity=0.65,
        zoom=4.6, center=MAP_CENTER, height=height, map_style="carto-positron",
        title=title,
    )
    if point is not None:
        fig.add_trace(go.Scattermap(
            lat=[point[0]], lon=[point[1]], mode="markers",
            marker=dict(size=18, color="red"), name="Your location",
            hovertext=["Your location"], showlegend=True,
        ))
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig


def money(x):
    return f"${x:,.0f}"


# ----------------------------------------------------------------- load everything
if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error("Missing `housing_model.joblib` or `housing.csv` next to app.py.")
    st.stop()

model = load_model()
df = load_csv(DATA_PATH)
results = load_csv(RESULTS_PATH)
importance = load_csv(IMPORTANCE_PATH)

if results is not None:
    TYPICAL_ERROR = float(np.sqrt((results["residual"] ** 2).mean()))
else:
    TYPICAL_ERROR = FALLBACK_ERROR

map_sample = df.sample(min(len(df), 6000), random_state=0)

# ----------------------------------------------------------------- header
st.title("🏠 California House Price Dashboard")
st.caption(
    "Predict the **median house value of a census block group** with a tuned "
    "HistGradientBoosting model, and explore the data behind it."
)

# ----------------------------------------------------------------- sidebar inputs
st.sidebar.header("Neighbourhood details")
median_income = st.sidebar.slider(
    "Median income (in $10,000s)", 0.5, 15.0, 3.5, 0.1,
    help="3.5 means a $35,000 median household income in the area.",
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

problems = []
if total_bedrooms > total_rooms:
    problems.append("Total bedrooms cannot exceed total rooms.")
if households > population:
    problems.append("Households cannot exceed population.")

base_row = {
    "longitude": float(longitude), "latitude": float(latitude),
    "housing_median_age": float(housing_median_age), "total_rooms": float(total_rooms),
    "total_bedrooms": float(total_bedrooms), "population": float(population),
    "households": float(households), "median_income": float(median_income),
    "ocean_proximity": ocean_proximity,
}
X_new = pd.DataFrame([base_row])
pred = None if problems else float(model.predict(X_new)[0])

tab_predict, tab_explore, tab_model = st.tabs(
    ["🔮 Predict", "📊 Explore the data", "🎯 Model performance"]
)

# =================================================================== TAB 1: PREDICT
with tab_predict:
    for p in problems:
        st.warning(p)

    if pred is not None:
        pct = (df["median_house_value"] < pred).mean() * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated median house value", money(pred))
        c2.metric("Likely range", f"{money(max(pred - TYPICAL_ERROR, 0))} – {money(pred + TYPICAL_ERROR)}",
                  help="Estimate ± the model's typical error (RMSE on the test set).")
        c3.metric("Compared with the data", f"Higher than {pct:.0f}%",
                  help="Share of California block groups with a lower median value.")
        if pred >= 480_000:
            st.info("Training data is capped at $500,001, so estimates near this are underestimates.")

        left, right = st.columns(2)

        with left:
            fig = px.histogram(df, x="median_house_value", nbins=50,
                               title="Where your estimate sits in California",
                               labels={"median_house_value": "Median house value ($)"})
            fig.add_vline(x=pred, line_color="red", line_width=3,
                          annotation_text="Your estimate", annotation_position="top")
            fig.update_layout(yaxis_title="Number of block groups", showlegend=False)
            st.plotly_chart(fig, width="stretch")

        with right:
            options = {
                "Median income ($10k)": "median_income",
                "Median house age": "housing_median_age",
                "Total rooms": "total_rooms",
                "Households": "households",
                "Population": "population",
            }
            label = st.selectbox("What if this changed? Vary:", list(options))
            feat = options[label]
            lo, hi = df[feat].quantile([0.01, 0.99])
            grid_vals = np.linspace(lo, hi, 40)
            sweep = pd.concat([X_new] * len(grid_vals), ignore_index=True)
            sweep[feat] = grid_vals
            sweep_pred = model.predict(sweep)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=grid_vals, y=sweep_pred, mode="lines", name="Predicted value"))
            fig.add_trace(go.Scatter(x=[base_row[feat]], y=[pred], mode="markers",
                                     marker=dict(size=14, color="red"), name="Your input"))
            fig.update_layout(title=f"Predicted price as {label.lower()} changes",
                              xaxis_title=label, yaxis_title="Predicted value ($)")
            st.plotly_chart(fig, width="stretch")
            st.caption("All other inputs are held at your current values.")

        fig = make_map(map_sample, "median_house_value", "Viridis",
                       "Your location among California block groups (colour = median house value)",
                       hover=["median_income", "housing_median_age"], point=(latitude, longitude))
        st.plotly_chart(fig, width="stretch")

# =================================================================== TAB 2: EXPLORE
with tab_explore:
    n_capped = int((df["median_house_value"] >= PRICE_CAP).sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Block groups", f"{len(df):,}")
    k2.metric("Median house value", money(df["median_house_value"].median()))
    k3.metric("Median income", f"${df['median_income'].median() * 10_000:,.0f}")
    k4.metric("Capped at $500,001", f"{n_capped:,}", help="Values above the cap are recorded as 500,001.")

    color_by = st.radio("Colour the map by", ["median_house_value", "median_income"], horizontal=True)
    fig = make_map(map_sample, color_by, "Viridis" if color_by == "median_house_value" else "Plasma",
                   f"California block groups, coloured by {color_by}",
                   hover=["median_house_value", "median_income", "ocean_proximity"])
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    with a:
        numeric_cols = [c for c in df.select_dtypes("number").columns]
        feat = st.selectbox("Distribution of", numeric_cols,
                            index=numeric_cols.index("median_income"))
        fig = px.histogram(df, x=feat, nbins=50, title=f"Distribution of {feat}")
        fig.update_layout(yaxis_title="Count")
        st.plotly_chart(fig, width="stretch")
    with b:
        fig = px.box(df, x="ocean_proximity", y="median_house_value", color="ocean_proximity",
                     title="House value by ocean proximity")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Median house value ($)")
        st.plotly_chart(fig, width="stretch")

    c, d = st.columns(2)
    with c:
        fig = px.scatter(map_sample, x="median_income", y="median_house_value",
                         color="ocean_proximity", opacity=0.4,
                         title="Income vs house value",
                         labels={"median_income": "Median income ($10k)",
                                 "median_house_value": "Median house value ($)"})
        st.plotly_chart(fig, width="stretch")
        st.caption("The flat line at the top is the $500,001 cap.")
    with d:
        corr = df.select_dtypes("number").corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto", title="Correlation between features")
        st.plotly_chart(fig, width="stretch")

# =================================================================== TAB 3: MODEL
with tab_model:
    if results is None:
        st.info(
            "Model performance charts need `test_results.csv` and `feature_importance.csv`. "
            "Run `export_artifacts.py` at the end of your notebook and copy both files next to `app.py`."
        )
    else:
        res = results
        rmse = float(np.sqrt((res["residual"] ** 2).mean()))
        mae = float(res["residual"].abs().mean())
        ss_res = float((res["residual"] ** 2).sum())
        ss_tot = float(((res["actual"] - res["actual"].mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test RMSE", money(rmse))
        m2.metric("Test MAE", money(mae))
        m3.metric("Test R²", f"{r2:.3f}")
        m4.metric("Test samples", f"{len(res):,}")
        st.caption("All numbers and charts on this tab use the held-out test set, which the model never trained on.")

        a, b = st.columns(2)
        with a:
            fig = px.scatter(res, x="actual", y="predicted", opacity=0.4,
                             title="Actual vs predicted",
                             labels={"actual": "Actual value ($)", "predicted": "Predicted value ($)"})
            lim = [float(min(res["actual"].min(), res["predicted"].min())),
                   float(max(res["actual"].max(), res["predicted"].max()))]
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect prediction",
                                     line=dict(color="red", dash="dash")))
            st.plotly_chart(fig, width="stretch")
            st.caption("Points on the red line are perfect. Notice the cap at $500,001 on the right.")
        with b:
            fig = px.histogram(res, x="residual", nbins=60, title="Prediction errors (actual − predicted)",
                               labels={"residual": "Error ($)"})
            fig.add_vline(x=0, line_color="red", line_dash="dash")
            fig.update_layout(yaxis_title="Count")
            st.plotly_chart(fig, width="stretch")
            st.caption("Centred near zero means no systematic over- or under-prediction.")

        c, d = st.columns(2)
        with c:
            fig = px.scatter(res, x="predicted", y="residual", opacity=0.4,
                             title="Errors vs predicted value",
                             labels={"predicted": "Predicted value ($)", "residual": "Error ($)"})
            fig.add_hline(y=0, line_color="red", line_dash="dash")
            st.plotly_chart(fig, width="stretch")
        with d:
            if "ocean_proximity" in res.columns:
                by_loc = (res.assign(abs_error=res["residual"].abs())
                          .groupby("ocean_proximity", as_index=False)["abs_error"].mean())
                fig = px.bar(by_loc, x="ocean_proximity", y="abs_error", color="ocean_proximity",
                             title="Average error by ocean proximity",
                             labels={"abs_error": "Mean absolute error ($)", "ocean_proximity": ""})
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width="stretch")

        fig = make_map(res, "residual", "RdBu", "Where the model is wrong (test set)",
                       midpoint=0, range_color=[-150_000, 150_000],
                       hover=["actual", "predicted"])
        st.plotly_chart(fig, width="stretch")
        st.caption("Blue = model underestimates the price, red = model overestimates. "
                   "Colour scale is limited to ±$150,000 so patterns stay visible.")

        if importance is not None:
            imp = importance.sort_values("importance", ascending=True)
            fig = px.bar(imp, x="importance", y="feature", orientation="h",
                         error_x="std" if "std" in imp.columns else None,
                         title="Feature importance (permutation, on the test set)",
                         labels={"importance": "Increase in RMSE ($) when the feature is shuffled",
                                 "feature": ""})
            st.plotly_chart(fig, width="stretch")
            st.caption("A longer bar means the model relies on that input more.")

st.divider()
st.caption(
    "⚠️ Based on 1990 California census data, so prices are far below today's market. "
    "This is a learning project, not financial advice."
)
