# train.py
import pandas as pd
from pycaret.classification import *

# ========================
# 1️⃣ Load and clean data
# ========================
data = pd.read_csv("data/churn_data.csv")

# Drop ID column
if "customerID" in data.columns:
    data.drop(columns=["customerID"], inplace=True)

# Convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

# Drop rows with missing target
data.dropna(subset=["Churn"], inplace=True)

# ========================
# 2️⃣ Setup PyCaret
# ========================
clf = setup(
    data=data,
    target="Churn",
    session_id=42,

    # Preprocessing
    normalize=True,
    remove_outliers=True,

    # Feature types
    categorical_features=[
        "PhoneService",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ],
    numeric_features=[
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ],

    # Handle imbalance
    fix_imbalance=True,
    fix_imbalance_method="SMOTE",

    # Cross-validation
    fold=5,
    fold_strategy="stratifiedkfold",

    # Silent execution
    verbose=False
)

# ========================
# 3️⃣ Compare models (F1-score)
# ========================
best_model = compare_models(sort="F1")

# ========================
# 4️⃣ Tune best model (F1-score)
# ========================
tuned_model = tune_model(
    best_model,
    optimize="F1",
    choose_better=True
)

# ========================
# 5️⃣ Finalize model
# ========================
final_model = finalize_model(tuned_model)

# ========================
# 6️⃣ Save model
# ========================
save_model(final_model, "models/churn_model")

# ========================
# 7️⃣ Extract REAL algorithm name
# ========================
# PyCaret wraps everything in a pipeline
real_model = final_model.steps[-1][1]
algorithm_name = type(real_model).__name__

# ========================
# 8️⃣ Get CV results for F1
# ========================
results = pull()
cv_f1 = results.loc["Mean", "F1"]

# ========================
# 9️⃣ Model summary (THIS IS WHAT YOU EXPLAIN)
# ========================
print("\n================ MODEL SUMMARY ================")
print(f"✅ Final algorithm used: {algorithm_name}")
print(f"✅ Cross-validated F1-score: {cv_f1:.4f}")
print("✅ CV Strategy: 5-Fold Stratified")
print("✅ Imbalance Handling: SMOTE")
print("✅ Model saved at: models/churn_model.pkl")
print("================================================\n")

# ========================
# 🔟 Evaluation plots
# ========================
plot_model(final_model, plot="confusion_matrix")
plot_model(final_model, plot="auc")
plot_model(final_model, plot="pr")
