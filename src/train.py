"""Train, compare and save churn models. Run from repo root: python -m src.train"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,
                             brier_score_loss, roc_curve, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.features import add_features, FEATURES, CAT, reasons, retention_action

SEED = 42
df = pd.read_csv("data/customer_data.csv")
df = add_features(df)
X, y = df[FEATURES], df["churn"]
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, df["customer_id"], test_size=0.2, stratify=y, random_state=SEED)

num_cols = [c for c in FEATURES if c not in CAT]
pre_scaled = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
pre_tree = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # class imbalance handling
models = {
    "Logistic Regression": Pipeline([("pre", pre_scaled),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
    "Random Forest": Pipeline([("pre", pre_tree),
        ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
                                       random_state=SEED, n_jobs=-1))]),
    "XGBoost": Pipeline([("pre", pre_tree),
        ("clf", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="logloss",
                              random_state=SEED))]),
}

rows, probs = [], {}
for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    probs[name] = p
    rows.append({"Model": name, "ROC-AUC": roc_auc_score(y_test, p),
                 "Precision": precision_score(y_test, pred), "Recall": recall_score(y_test, pred),
                 "F1": f1_score(y_test, pred), "Brier (calibration)": brier_score_loss(y_test, p)})
res = pd.DataFrame(rows).round(4)
print(res.to_string(index=False))
res.to_csv("outputs/model_comparison.csv", index=False)

best_name = res.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]
best = models[best_name]
print("Best model:", best_name)
joblib.dump(best, "models/best_model.joblib")

# ---- Plots ----
plt.figure(figsize=(6, 5))
for n, p in probs.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(y_test, p):.3f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves"); plt.legend(); plt.tight_layout(); plt.savefig("outputs/roc_curves.png", dpi=130); plt.close()

plt.figure(figsize=(6, 5))
for n, p in probs.items():
    fp, mp = calibration_curve(y_test, p, n_bins=10)
    plt.plot(mp, fp, marker="o", label=n)
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("Mean predicted probability"); plt.ylabel("Observed churn rate")
plt.title("Calibration curves"); plt.legend(); plt.tight_layout(); plt.savefig("outputs/calibration_curves.png", dpi=130); plt.close()

cm = confusion_matrix(y_test, (probs[best_name] >= 0.5).astype(int))
plt.figure(figsize=(4.5, 4)); plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
plt.xticks([0, 1], ["Stay", "Churn"]); plt.yticks([0, 1], ["Stay", "Churn"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix - {best_name}")
plt.tight_layout(); plt.savefig("outputs/confusion_matrix.png", dpi=130); plt.close()

# Feature importance (tree model if best is tree, else from RF for interpretability)
imp_model = best if best_name != "Logistic Regression" else models["Random Forest"]
names = imp_model.named_steps["pre"].get_feature_names_out()
imp = pd.Series(imp_model.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
imp.index = [i.split("__")[1] for i in imp.index]
imp.head(15).iloc[::-1].plot.barh(figsize=(7, 5), color="#2a6f97")
plt.title("Top 15 feature importances"); plt.tight_layout(); plt.savefig("outputs/feature_importance.png", dpi=130); plt.close()
imp.head(15).round(4).to_csv("outputs/feature_importance.csv", header=["importance"])

# Churn rate by contract plot for EDA
df.groupby("contract")["churn"].mean().plot.bar(color="#e07a5f", figsize=(4, 3.5), rot=0)
plt.ylabel("Churn rate"); plt.title("Churn rate by contract"); plt.tight_layout()
plt.savefig("outputs/eda_contract.png", dpi=130); plt.close()

# ---- Ranked high-risk list (all customers scored with final model) ----
df["churn_probability"] = best.predict_proba(df[FEATURES])[:, 1]
df["risk_tier"] = pd.cut(df["churn_probability"], [-0.01, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"])
df["review_reasons"] = df.apply(reasons, axis=1)
df["suggested_action"] = df["review_reasons"].apply(retention_action)
ranked = df.sort_values("churn_probability", ascending=False)[
    ["customer_id", "churn_probability", "risk_tier", "review_reasons", "suggested_action",
     "plan", "contract", "tenure_months", "monthly_charges"]].round(4)
ranked.insert(0, "rank", range(1, len(ranked) + 1))
ranked.to_csv("outputs/ranked_high_risk_customers.csv", index=False)

# Held-out lift: how many true churners in top 10% of test set
t = pd.DataFrame({"y": y_test.values, "p": probs[best_name]}).sort_values("p", ascending=False)
top = t.head(int(len(t) * 0.1))
meta = {"best_model": best_name, "n_rows": int(len(df)), "churn_rate": float(df.churn.mean()),
        "test_size": int(len(t)), "top10_precision": float(top.y.mean()),
        "top10_lift": float(top.y.mean() / t.y.mean()),
        "high_risk_count": int((df.risk_tier == "High").sum()),
        "medium_risk_count": int((df.risk_tier == "Medium").sum()),
        "confusion": cm.tolist()}
json.dump(meta, open("outputs/summary.json", "w"), indent=2)
print(meta)
