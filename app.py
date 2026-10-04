import joblib
import pandas as pd
import streamlit as st
from src.features import add_features, FEATURES, reasons, retention_action

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")
st.title("📉 Customer Churn Prediction")
st.caption("Predict which subscribers are likely to leave and why. Data in this project is synthetic.")


@st.cache_resource
def load_model():
    return joblib.load("models/best_model.joblib")


@st.cache_data
def load_ranked():
    return pd.read_csv("outputs/ranked_high_risk_customers.csv")


model = load_model()
tab1, tab2, tab3 = st.tabs(["🎯 Predict one customer", "📋 Ranked high-risk list", "📊 Model results"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 18, 75, 35)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        region = st.selectbox("Region", ["South", "North", "East", "West"])
        plan = st.selectbox("Plan", ["Basic", "Standard", "Premium"])
        contract = st.selectbox("Contract", ["Monthly", "Annual"])
        payment_method = st.selectbox("Payment method", ["Card", "UPI", "NetBanking", "Wallet"])
    with c2:
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        charges = st.number_input("Monthly charges", 100.0, 1500.0, 399.0)
        prev = st.number_input("Previous monthly charges", 100.0, 1500.0, 399.0)
        discount = st.selectbox("Discount active", [0, 1])
        late = st.slider("Late payments (6 months)", 0, 6, 0)
    with c3:
        u1 = st.number_input("Usage hours - last month", 0.0, 150.0, 18.0)
        u2 = st.number_input("Usage hours - 2 months ago", 0.0, 150.0, 20.0)
        u3 = st.number_input("Usage hours - 3 months ago", 0.0, 160.0, 22.0)
        logins = st.slider("Logins (last 30 days)", 0, 100, 15)
        tickets = st.slider("Support tickets (90 days)", 0, 12, 1)
        res = st.slider("Avg resolution days", 0.0, 20.0, 2.0)
        sat = st.slider("Satisfaction score", 1.0, 5.0, 3.8, 0.1)

    if st.button("Predict churn risk", type="primary"):
        row = pd.DataFrame([{
            "age": age, "gender": gender, "region": region, "plan": plan, "contract": contract,
            "payment_method": payment_method, "tenure_months": tenure, "monthly_charges": charges,
            "prev_monthly_charges": prev, "discount_active": discount, "usage_hours_m1": u1,
            "usage_hours_m2": u2, "usage_hours_m3": u3, "logins_last_30d": logins,
            "support_tickets_90d": tickets, "avg_resolution_days": res,
            "satisfaction_score": sat, "late_payments_6m": late}])
        feat = add_features(row)
        prob = float(model.predict_proba(feat[FEATURES])[0, 1])
        tier = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
        st.metric("Churn probability", f"{prob:.1%}", tier + " risk")
        st.progress(min(prob, 1.0))
        why = reasons(feat.iloc[0])
        st.write("**Suggested review reasons:**", why)
        st.write("**Suggested action:**", retention_action(why))

with tab2:
    ranked = load_ranked()
    tier = st.multiselect("Risk tier", ["High", "Medium", "Low"], default=["High"])
    n = st.slider("Rows to show", 10, 500, 50)
    view = ranked[ranked["risk_tier"].isin(tier)].head(n)
    st.dataframe(view, hide_index=True)
    st.download_button("Download CSV", view.to_csv(index=False), "high_risk_customers.csv", "text/csv")

with tab3:
    st.subheader("Model comparison")
    st.dataframe(pd.read_csv("outputs/model_comparison.csv"), hide_index=True)
    a, b = st.columns(2)
    a.image("outputs/roc_curves.png", caption="ROC curves")
    b.image("outputs/calibration_curves.png", caption="Calibration curves")
    a.image("outputs/feature_importance.png", caption="Feature importance")
    b.image("outputs/confusion_matrix.png", caption="Confusion matrix (best model)")
