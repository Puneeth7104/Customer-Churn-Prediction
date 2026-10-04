"""Generate a synthetic subscription-service customer dataset (no real/confidential data)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 5000

customer_id = [f"CUST{100000 + i}" for i in range(N)]
age = np.clip(rng.normal(36, 11, N), 18, 75).round().astype(int)
gender = rng.choice(["Male", "Female", "Other"], N, p=[0.49, 0.49, 0.02])
region = rng.choice(["South", "North", "East", "West"], N, p=[0.35, 0.25, 0.15, 0.25])
plan = rng.choice(["Basic", "Standard", "Premium"], N, p=[0.45, 0.35, 0.20])
contract = rng.choice(["Monthly", "Annual"], N, p=[0.65, 0.35])
payment_method = rng.choice(["Card", "UPI", "NetBanking", "Wallet"], N, p=[0.35, 0.40, 0.15, 0.10])
tenure_months = np.clip(rng.exponential(18, N), 1, 72).round().astype(int)

base_price = pd.Series(plan).map({"Basic": 199, "Standard": 399, "Premium": 699}).values
monthly_charges = (base_price * rng.normal(1.0, 0.05, N)).round(2)

usage_hours_m1 = np.clip(rng.gamma(4, 5, N), 0, 120)  # most recent month
trend = rng.normal(0, 0.25, N)
usage_hours_m2 = np.clip(usage_hours_m1 / (1 + trend * 0.5) * rng.normal(1, 0.1, N), 0, 140)
usage_hours_m3 = np.clip(usage_hours_m2 / (1 + trend * 0.5) * rng.normal(1, 0.1, N), 0, 160)
logins_last_30d = np.clip((usage_hours_m1 * rng.normal(0.8, 0.2, N)).round(), 0, None).astype(int)

support_tickets_90d = rng.poisson(1.2, N)
avg_resolution_days = np.clip(rng.gamma(2, 1.5, N), 0.2, 20).round(1)
avg_resolution_days[support_tickets_90d == 0] = 0
satisfaction_score = np.clip(rng.normal(3.8, 1.0, N) - 0.25 * support_tickets_90d, 1, 5).round(1)

late_payments_6m = rng.poisson(0.5, N)
prev_monthly_charges = (monthly_charges * rng.choice([1.0, 1.0, 1.0, 0.9, 1.1, 1.2], N)).round(2)
discount_active = rng.choice([0, 1], N, p=[0.8, 0.2])

# True churn mechanism (hidden): logistic of risk drivers
usage_drop = (usage_hours_m3 - usage_hours_m1) / (usage_hours_m3 + 1)
price_change = (monthly_charges - prev_monthly_charges) / prev_monthly_charges
z = (
    -1.2
    + 0.9 * (contract == "Monthly")
    - 0.03 * tenure_months
    + 1.6 * usage_drop
    - 0.03 * usage_hours_m1
    + 0.28 * support_tickets_90d
    + 0.08 * avg_resolution_days
    - 0.45 * (satisfaction_score - 3.5)
    + 0.45 * late_payments_6m
    + 2.5 * np.clip(price_change, 0, None)
    - 0.5 * discount_active
    + 0.3 * (payment_method == "Wallet")
)
p = 1 / (1 + np.exp(-z))
churn = (rng.random(N) < p).astype(int)

df = pd.DataFrame({
    "customer_id": customer_id, "age": age, "gender": gender, "region": region,
    "plan": plan, "contract": contract, "payment_method": payment_method,
    "tenure_months": tenure_months, "monthly_charges": monthly_charges,
    "prev_monthly_charges": prev_monthly_charges, "discount_active": discount_active,
    "usage_hours_m1": usage_hours_m1.round(1), "usage_hours_m2": usage_hours_m2.round(1),
    "usage_hours_m3": usage_hours_m3.round(1), "logins_last_30d": logins_last_30d,
    "support_tickets_90d": support_tickets_90d, "avg_resolution_days": avg_resolution_days,
    "satisfaction_score": satisfaction_score, "late_payments_6m": late_payments_6m,
    "churn": churn,
})

# Inject realistic messiness: missing values and outliers
for col, frac in [("satisfaction_score", 0.06), ("age", 0.02), ("avg_resolution_days", 0.03), ("usage_hours_m2", 0.02)]:
    df.loc[rng.choice(N, int(N * frac), replace=False), col] = np.nan
df.loc[rng.choice(N, 15, replace=False), "monthly_charges"] *= 6   # billing outliers
df.loc[rng.choice(N, 10, replace=False), "usage_hours_m1"] += 200   # usage outliers

df.to_csv("data/customer_data.csv", index=False)
print(df.shape, "churn rate:", round(df.churn.mean(), 3))
