# 🏠 California House Price Predictor

A machine learning project that predicts the **median house value of a California census block group** from neighbourhood features such as income, location, and housing characteristics. Includes a full modelling notebook and an interactive **Streamlit dashboard** with live predictions, data exploration and model diagnostics.

## App features

- **🔮 Predict:** live price estimate with a likely range, where it sits in the California price distribution, a "what if" curve showing how the price changes as one input changes, and a map of the location.
- **📊 Explore the data:** price and income maps, distributions, house value by ocean proximity, income vs price, and a correlation heatmap.
- **🎯 Model performance:** test-set RMSE / MAE / R², actual vs predicted, error distribution, an error map, error by region, and feature importance.

🔗 **Live demo:** _add your Streamlit link here after deploying_

---

## Project overview

| | |
|---|---|
| **Problem** | Regression: predict `median_house_value` |
| **Dataset** | California Housing (1990 US census), 20,640 rows, 9 input features |
| **Final model** | Tuned `HistGradientBoostingRegressor` inside a scikit-learn `Pipeline` |
| **Metric** | RMSE (also MAE and R²) |
| **App** | Streamlit |

## Workflow

1. **Exploratory data analysis:** distributions, missing values (`total_bedrooms`), correlations, and the capped target.
2. **Preprocessing pipeline:** median imputation and scaling for numeric columns, most-frequent imputation and one-hot encoding for `ocean_proximity`.
3. **Baseline:** Linear Regression.
4. **Model comparison:** several regressors compared with 5-fold cross-validation.
5. **Feature engineering:** added 8 features (see below).
6. **Hyperparameter tuning:** `GridSearchCV` on the best model.
7. **Final evaluation:** a single run on the held-out test set.
8. **Deployment:** model saved with `joblib` and served through Streamlit.

## Feature engineering

| Feature | Idea |
|---|---|
| `rooms_per_household` | Size of homes, independent of block-group size |
| `bedrooms_per_room` | Layout of homes |
| `population_per_household` | Crowding |
| `dist_<city>` (5 features) | Distance to Los Angeles, San Francisco, San Diego, San Jose and Sacramento |
| `dist_nearest_city` | Distance to the closest major city |

Adding these reduced cross-validated RMSE from about **48,250** to about **45,958** (roughly 5%) with a default HistGradientBoosting model.

## Results

| Model | CV RMSE | Test RMSE |
|---|---|---|
| Linear Regression (baseline) | TODO | TODO |
| Tuned HistGradientBoosting + features | TODO | TODO |

_Replace the TODO values with the numbers from your notebook. Test metrics should come from a single final evaluation on the held-out test set._

## Limitations

- **Old data:** the dataset is from 1990, so predicted prices are far below today's market.
- **Capped target:** house values are capped at $500,001, so the model underestimates very expensive areas.
- **Block-group level:** predictions are for a census block group's median value, not for an individual house.
- This is a learning project and **not financial advice**.

## Project structure

```
.
├── app.py                  # Streamlit dashboard (3 tabs)
├── features.py             # Feature engineering (used by the notebook and the app)
├── housing_model.joblib    # Trained model pipeline
├── housing.csv             # Dataset (used by the Explore tab)
├── test_results.csv        # Test-set predictions (used by the Model performance tab)
├── feature_importance.csv  # Permutation importance (used by the Model performance tab)
├── export_artifacts.py     # Code run at the end of the notebook to create the files above
├── Try2.ipynb              # Full analysis and modelling notebook
├── requirements.txt
└── README.md
```

## Run it locally

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

> `features.py` must stay in the same folder as `app.py` and `housing_model.joblib`, because the saved pipeline refers to it.

## Tech stack

Python · pandas · NumPy · scikit-learn · Plotly · Streamlit · joblib

## Dataset

California Housing dataset, derived from the 1990 US Census (also available on [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices)).

## Working Model

| [Deployed App][(https://github.com/Deepak180607/Housing-Price-Predictor](https://housing-price-predictor-5d5dpt3krxqmkdhu4rnrh5.streamlit.app/)) |

## Author

**Deepak**: B.Tech Data Science, ITER, SOA University

