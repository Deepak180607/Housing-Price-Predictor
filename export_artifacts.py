# Run this at the END of your notebook, after the grid search and after you have
# X_test / y_test. It saves the files the "Model performance" tab needs.
# Everything here is computed on the TEST set, so the charts are honest.
import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

best = grid.best_estimator_

# 1) the trained model (same as save_model.py)
joblib.dump(best, "housing_model.joblib")

# 2) test-set predictions + errors
pred = best.predict(X_test)
results = X_test.copy()
results["actual"] = y_test.values
results["predicted"] = pred
results["residual"] = results["actual"] - results["predicted"]
results.to_csv("test_results.csv", index=False)

# 3) permutation importance of the raw input columns (takes ~1 minute)
r = permutation_importance(
    best, X_test, y_test,
    n_repeats=5, random_state=42, n_jobs=-1,
    scoring="neg_root_mean_squared_error",
)
imp = pd.DataFrame({
    "feature": X_test.columns,
    "importance": r.importances_mean,   # how much RMSE gets worse when the column is shuffled
    "std": r.importances_std,
}).sort_values("importance", ascending=False)
imp.to_csv("feature_importance.csv", index=False)

print("Saved: housing_model.joblib, test_results.csv, feature_importance.csv")
