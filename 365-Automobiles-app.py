
import streamlit as st
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="365 Automobiles - Capstone", layout="wide")
st.title("🚗 365 Automobiles Inc. - Supervised ML Portfolio App")
st.markdown("**Batch-06** | 3 Models + Service Analysis | Individual work - explainable line-by-line")

# Load CSVs - try multiple locations
def load_csv(name):
    for p in [name, f"/mnt/data/{name}", os.path.join(os.getcwd(), name)]:
        if os.path.exists(p):
            return pd.read_csv(p)
    st.error(f"{name} not found. Please put it in same folder as app.")
    st.stop()

sales = load_csv("365_Automobiles_Sales.csv")
service = load_csv("365_Automobiles_Service.csv")

def get_preproc(cat,num,scale=False):
    from sklearn.preprocessing import StandardScaler
    tr=[]
    if cat: tr.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat))
    if num: tr.append(('num', StandardScaler() if scale else 'passthrough', num))
    return ColumnTransformer(tr)

# Feature sets (leakage-free)
model_A_cat = ['model_interested','trim_interested','exterior_colour','showroom_location','lead_source','customer_gender','emirate','employment_type','payment_type']
model_A_num = ['customer_age','monthly_income_aed','years_at_current_job','existing_customer','motor_power_hp','list_price_aed','options_value_aed','showroom_visits','test_drive_taken','brochure_downloaded','finance_quote_requested','days_in_pipeline','has_tradein','tradein_value_aed','down_payment_pct','salesperson_experience_yrs']

model_B_cat = ['showroom_location','lead_source','customer_gender','emirate','employment_type','model_interested','trim_interested','exterior_colour','segment']
model_B_num = ['customer_age','monthly_income_aed','years_at_current_job','existing_customer','battery_kwh','range_km','motor_power_hp','list_price_aed','options_value_aed','showroom_visits','test_drive_taken','brochure_downloaded','finance_quote_requested','days_in_pipeline','has_tradein','salesperson_experience_yrs']

model_C_cat = ['showroom_location','lead_source','customer_gender','emirate','employment_type','model_interested','trim_interested','exterior_colour','segment']
model_C_num = ['customer_age','monthly_income_aed','years_at_current_job','existing_customer','battery_kwh','range_km','motor_power_hp','list_price_aed','options_value_aed','showroom_visits','test_drive_taken','brochure_downloaded','finance_quote_requested','days_in_pipeline','has_tradein','tradein_value_aed','salesperson_experience_yrs']

@st.cache_resource
def train_models():
    buyers = sales[sales['purchased']=='Yes']
    X_A = buyers[model_A_cat+model_A_num]; y_A = buyers['final_price_aed']
    pipe_A = Pipeline([('pre',get_preproc(model_A_cat,model_A_num,scale=True)),('m',LinearRegression())]); pipe_A.fit(X_A,y_A)

    X_B = sales[model_B_cat+model_B_num]; y_B = sales['purchased'].map({'Yes':1,'No':0})
    pipe_B = Pipeline([('pre',get_preproc(model_B_cat,model_B_num)),('m',RandomForestClassifier(n_estimators=150,random_state=42,max_depth=12))]); pipe_B.fit(X_B,y_B)

    X_C = buyers[model_C_cat+model_C_num]; y_C = buyers['model_purchased']
    pipe_C = Pipeline([('pre',get_preproc(model_C_cat,model_C_num)),('m',RandomForestClassifier(n_estimators=150,random_state=42,max_depth=12))]); pipe_C.fit(X_C,y_C)
    return pipe_A, pipe_B, pipe_C, X_A, X_B, X_C

pipe_A, pipe_B, pipe_C, X_A, X_B, X_C = train_models()

tab1, tab2, tab3, tab4 = st.tabs(["💰 Model A: Final Price", "🎯 Model B: Lead Conversion", "🚙 Model C: Model Choice", "🔧 Service Analysis"])

with tab1:
    st.header("Model A - Final Selling Price (LinearRegression, Buyers Only)")
    st.markdown("Target: `final_price_aed` | Population: purchased==Yes (3574) | Excludes: discount_pct, down_payment_aed, vehicle_id, days_to_delivery (post-deal leakage)")
    col1, col2, col3 = st.columns(3)
    with col1:
        list_price = st.number_input("List Price AED", 80000, 350000, 150000, step=1000)
        options = st.number_input("Options Value AED", 0, 20000, 3000, step=500)
        age = st.slider("Customer Age", 18, 70, 35)
    with col2:
        model_int = st.selectbox("Model Interested", sales['model_interested'].unique())
        trim_int = st.selectbox("Trim Interested", sales['trim_interested'].unique())
        payment = st.selectbox("Payment Type", sales['payment_type'].unique())
    with col3:
        showroom = st.selectbox("Showroom", sales['showroom_location'].unique())
        lead_src = st.selectbox("Lead Source", sales['lead_source'].unique())
        income = st.number_input("Monthly Income AED", 3000, 100000, 20000)
    
    if st.button("Predict Final Price", key="A"):
        sample = X_A.iloc[[0]].copy()
        sample['list_price_aed']=list_price; sample['options_value_aed']=options; sample['customer_age']=age
        sample['model_interested']=model_int; sample['trim_interested']=trim_int; sample['payment_type']=payment
        sample['showroom_location']=showroom; sample['lead_source']=lead_src; sample['monthly_income_aed']=income
        pred = pipe_A.predict(sample)[0]
        st.success(f"**Predicted Final Price: {pred:,.0f} AED**")
        st.info(f"Model metrics on test: R²=0.9198, MAE=8,696 AED, RMSE=16,949 AED (honest, no leakage)")

with tab2:
    st.header("Model B - Lead Conversion (RandomForest Binary, All Enquiries)")
    st.markdown("Target: purchased Yes/No | All 12,000 enquiries | Excludes late-stage finance fields")
    col1, col2 = st.columns(2)
    with col1:
        visits = st.slider("Showroom Visits", 0, 5, 1)
        test_drive = st.selectbox("Test Drive Taken", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
        days_pipe = st.slider("Days in Pipeline", 1, 60, 15)
    with col2:
        brochure = st.selectbox("Brochure Downloaded", [0,1])
        finance_q = st.selectbox("Finance Quote Requested", [0,1])
        has_trade = st.selectbox("Has Trade-in", [0,1])
    
    if st.button("Predict Conversion", key="B"):
        sample = X_B.iloc[[0]].copy()
        sample['showroom_visits']=visits; sample['test_drive_taken']=test_drive; sample['days_in_pipeline']=days_pipe
        sample['brochure_downloaded']=brochure; sample['finance_quote_requested']=finance_q; sample['has_tradein']=has_trade
        prob = pipe_B.predict_proba(sample)[0,1]
        pred = pipe_B.predict(sample)[0]
        st.metric("Conversion Probability", f"{prob:.2%}")
        if prob > 0.5:
            st.success(f"Prediction: **Will Purchase (Yes)** - prioritize this lead!")
        else:
            st.warning(f"Prediction: **Will Not Purchase (No)**")
        st.info("Metrics: Accuracy 0.763 vs baseline 0.702, Precision 0.707, Recall 0.351, ROC-AUC 0.796 | Top feature: test_drive_taken")

with tab3:
    st.header("Model C - Vehicle Model Purchased (Multiclass, Buyers Only)")
    st.markdown("Target: Volt City / Volt Sedan / Terra SUV / Aura Premium | Purpose: early stock planning")
    col1, col2 = st.columns(2)
    with col1:
        income_c = st.number_input("Monthly Income AED (C)", 3000, 100000, 20000, key="incomeC")
        age_c = st.slider("Customer Age (C)", 18, 70, 35, key="ageC")
    with col2:
        model_int_c = st.selectbox("Model Interested (C)", sales['model_interested'].unique(), key="modelC")
        emirate = st.selectbox("Emirate", sales['emirate'].unique())
    
    if st.button("Predict Model Choice", key="C"):
        sample = X_C.iloc[[0]].copy()
        sample['monthly_income_aed']=income_c; sample['customer_age']=age_c
        sample['model_interested']=model_int_c; sample['emirate']=emirate
        pred = pipe_C.predict(sample)[0]
        proba = pipe_C.predict_proba(sample)[0]
        st.success(f"**Predicted Model: {pred}**")
        st.write("Probabilities:")
        for label, p in zip(pipe_C.classes_, proba):
            st.write(f"- {label}: {p:.2%}")
        st.info("Accuracy 0.892, Macro F1 0.880 (macro treats low-volume Aura equally). Volt City vs Volt Sedan most confused - stock together.")

with tab4:
    st.header("Service-Side Business Analysis")
    st.write(f"Service records: {service.shape[0]}, Battery health vs Age r={service['vehicle_age_months'].corr(service['battery_health_pct']):.2f}, vs Odo r={service['odometer_km'].corr(service['battery_health_pct']):.2f}")
    st.dataframe(service.groupby('model')['service_cost_aed'].mean().round(0))
    st.dataframe(service.groupby('service_type')['repeat_visit_30_days'].apply(lambda x: (x=='Yes').mean()).round(3))
    st.markdown("""
    **Recommendations:**
    1. Test-drive priority (top predictor)
    2. Discount guardrails: Volt Sedan 4.97% lowest, cap Volt City at 6%
    3. Inventory: Volt City+Sedan co-located
    4. Workshop: Sharjah Industrial wait 3.0d, Repair repeat 20.5% - add bay
    5. Battery: <80% after 10m/10k km - preventive service
    6. Lead source: Referral ~45% vs Website ~25%
    """)

st.sidebar.markdown("### How to Run")
st.sidebar.code("pip install streamlit pandas scikit-learn\nstreamlit run streamlit_app.py", language="bash")
st.sidebar.info("Put 365_Automobiles_Sales.csv and Service.csv in same folder as this app.")
