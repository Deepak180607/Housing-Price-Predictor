"""Feature engineering for the California housing model.

Kept in its own module (not inside the notebook) so that the saved model can be
loaded later by the Streamlit app. Pickled pipelines store a *reference* to this
function, so features.py must sit next to housing_model.joblib / app.py.
"""
import numpy as np
import pandas as pd

# (latitude, longitude) of major California cities
CITIES = {
    "los_angeles": (34.05, -118.25),
    "san_francisco": (37.77, -122.42),
    "san_diego": (32.72, -117.16),
    "san_jose": (37.34, -121.89),
    "sacramento": (38.58, -121.49),
}


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-household / per-room ratios. Raw totals depend on block-group size;
    ratios describe the *kind* of neighbourhood."""
    df = df.copy()
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    return df


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Distance (in degrees) to each major city, and to the nearest one."""
    df = df.copy()
    dists = []
    for name, (lat, lon) in CITIES.items():
        d = np.sqrt((df["latitude"] - lat) ** 2 + (df["longitude"] - lon) ** 2)
        df[f"dist_{name}"] = d
        dists.append(d)
    df["dist_nearest_city"] = np.min(dists, axis=0)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature set used by the pipeline."""
    return add_location_features(add_ratio_features(df))