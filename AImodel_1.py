import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import joblib

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
INPUT_CSV = "structured_real_estate_accumulated.csv"
df = pd.read_csv(INPUT_CSV)

# Basic cleaning
df["bhk"] = df["bhk"].fillna(0).astype(float)
df["sqft_builtup"] = df["sqft_builtup"].fillna(0).astype(float)
df["sqft_uds"] = df["sqft_uds"].fillna(0).astype(float)
df["is_rental_flag"] = df["is_rental"].astype(bool)

# Text features
df["text_len"] = df["listing_text"].astype(str).str.len()
df["text_word_count"] = df["listing_text"].astype(str).str.split().str.len()

# ---------------------------------------------------------
# FEATURE SETUP
# ---------------------------------------------------------
CATEGORICAL_COLS = ["locality", "property_type", "facing", "floor"]
NUMERIC_COLS = ["bhk", "sqft_builtup", "sqft_uds", "text_len", "text_word_count"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ("num", "passthrough", NUMERIC_COLS),
    ]
)

# ---------------------------------------------------------
# SALE PRICE MODEL
# ---------------------------------------------------------
sale_df = df[df["price_in_inr"].notna() & (df["is_rental_flag"] == False)].copy()

X_sale = sale_df[CATEGORICAL_COLS + NUMERIC_COLS]
y_sale = sale_df["price_in_inr"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sale, y_sale, test_size=0.2, random_state=42
)

sale_model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )),
    ]
)

sale_model.fit(X_train_s, y_train_s)
y_pred_s = sale_model.predict(X_test_s)

mae_sale = mean_absolute_error(y_test_s, y_pred_s)
print(f"\nSale Model MAE: {mae_sale:,.0f} INR")

# ---------------------------------------------------------
# SALE MODEL PLOTS
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test_s, y_pred_s, alpha=0.6, color="blue")
plt.plot([y_test_s.min(), y_test_s.max()],
         [y_test_s.min(), y_test_s.max()],
         "r--", linewidth=2)
plt.xlabel("Actual Sale Price (INR)")
plt.ylabel("Predicted Sale Price (INR)")
plt.title("Sale Model: Actual vs Predicted")
plt.tight_layout()
plt.show()

# Residuals
residuals_s = y_test_s - y_pred_s
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_s, residuals_s, alpha=0.6, color="green")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Sale Price (INR)")
plt.ylabel("Residuals")
plt.title("Sale Model Residual Plot")
plt.tight_layout()
plt.show()

# Feature Importances
rf = sale_model.named_steps["model"]
ohe = sale_model.named_steps["preprocess"].named_transformers_["cat"]
cat_features = ohe.get_feature_names_out(CATEGORICAL_COLS)
all_features = list(cat_features) + NUMERIC_COLS

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 8))
plt.barh(np.array(all_features)[indices][:20],
         importances[indices][:20],
         color="teal")
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importances (Sale Model)")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# RENT MODEL
# ---------------------------------------------------------
rent_df = df[df["rent_in_inr"].notna() & (df["is_rental_flag"] == True)].copy()

if len(rent_df) > 20:
    X_rent = rent_df[CATEGORICAL_COLS + NUMERIC_COLS]
    y_rent = rent_df["rent_in_inr"]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_rent, y_rent, test_size=0.2, random_state=42
    )

    rent_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                random_state=42,
                n_jobs=-1
            )),
        ]
    )

    rent_model.fit(X_train_r, y_train_r)
    y_pred_r = rent_model.predict(X_test_r)

    mae_rent = mean_absolute_error(y_test_r, y_pred_r)
    print(f"Rent Model MAE: {mae_rent:,.0f} INR")

    # RENT MODEL PLOTS
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_r, y_pred_r, alpha=0.6, color="purple")
    plt.plot([y_test_r.min(), y_test_r.max()],
             [y_test_r.min(), y_test_r.max()],
             "r--", linewidth=2)
    plt.xlabel("Actual Rent (INR)")
    plt.ylabel("Predicted Rent (INR)")
    plt.title("Rent Model: Actual vs Predicted")
    plt.tight_layout()
    plt.show()

    residuals_r = y_test_r - y_pred_r
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred_r, residuals_r, alpha=0.6, color="orange")
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Rent (INR)")
    plt.ylabel("Residuals")
    plt.title("Rent Model Residual Plot")
    plt.tight_layout()
    plt.show()

    # Save both models
    joblib.dump(rent_model, "terrallytix_rent_model.joblib")

else:
    print("Not enough rental samples to train rent model.")
    rent_model = None

# Save sale model
joblib.dump(sale_model, "terrallytix_sale_price_model.joblib")

print("\nModels saved successfully.")
