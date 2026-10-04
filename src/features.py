"""Cleaning + feature engineering shared by training and the app."""
import numpy as np
import pandas as pd

NUM_BASE = ["age", "tenure_months", "monthly_charges", "prev_monthly_charges", "discount_active",
            "usage_hours_m1", "usage_hours_m2", "usage_hours_m3", "logins_last_30d",
            "support_tickets_90d", "avg_resolution_days", "satisfaction_score", "late_payments_6m"]
CAT = ["gender", "region", "plan", "contract", "payment_method"]
ENGINEERED = ["usage_trend", "usage_avg_3m", "support_freq_per_month", "price_change_pct",
              "charges_per_tenure", "low_engagement", "payment_risk"]
FEATURES = NUM_BASE + ENGINEERED + CAT


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Cap outliers (1st/99th percentile style fixed caps) so extreme values don't dominate."""
    df = df.copy()
    caps = {"monthly_charges": 1500, "usage_hours_m1": 150, "usage_hours_m2": 150, "usage_hours_m3": 160}
    for c, cap in caps.items():
        if c in df:
            df[c] = df[c].clip(upper=cap)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean(df)
    u2 = df["usage_hours_m2"].fillna((df["usage_hours_m1"] + df["usage_hours_m3"]) / 2)
    df["usage_trend"] = (df["usage_hours_m1"] - df["usage_hours_m3"]) / (df["usage_hours_m3"] + 1)
    df["usage_avg_3m"] = (df["usage_hours_m1"] + u2 + df["usage_hours_m3"]) / 3
    df["support_freq_per_month"] = df["support_tickets_90d"] / 3
    df["price_change_pct"] = (df["monthly_charges"] - df["prev_monthly_charges"]) / df["prev_monthly_charges"]
    df["charges_per_tenure"] = df["monthly_charges"] / (df["tenure_months"] + 1)
    df["low_engagement"] = ((df["usage_hours_m1"] < 5) | (df["logins_last_30d"] < 3)).astype(int)
    df["payment_risk"] = (df["late_payments_6m"] >= 2).astype(int)
    return df


def reasons(row) -> str:
    """Rule-based 'suggested review reasons' for a high-risk customer."""
    r = []
    if row["usage_trend"] < -0.25: r.append("Usage dropped sharply over 3 months")
    if row["low_engagement"] == 1: r.append("Very low engagement (usage/logins)")
    if row["support_tickets_90d"] >= 3: r.append("Frequent support contacts")
    if row["avg_resolution_days"] >= 4: r.append("Slow support resolution")
    if row["satisfaction_score"] <= 2.5: r.append("Low satisfaction score")
    if row["late_payments_6m"] >= 2: r.append("Repeated late payments")
    if row["price_change_pct"] > 0.05: r.append("Recent price increase")
    if row["contract"] == "Monthly": r.append("Month-to-month contract (no lock-in)")
    if row["tenure_months"] <= 6: r.append("New customer (tenure <= 6 months)")
    return "; ".join(r[:4]) if r else "General elevated risk - manual review"


def retention_action(reason: str) -> str:
    if "price" in reason.lower(): return "Offer loyalty discount / plan review"
    if "support" in reason.lower() or "satisfaction" in reason.lower(): return "Priority callback from support lead"
    if "usage" in reason.lower() or "engagement" in reason.lower(): return "Send re-engagement campaign / feature tips"
    if "late" in reason.lower(): return "Offer flexible payment / autopay setup"
    return "Offer annual-plan incentive"
