"""
Streamlit UI for the Credit Score classifier hosted on SageMaker.
 
Reads endpoint name and region from environment variables.
boto3 picks up AWS credentials from:
  - the EC2 instance profile (when running on EC2 with LabInstanceProfile), OR
  - ~/.aws/credentials (when running locally)
"""
 
import json
import os
 
import boto3
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError
 
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpoint-v3")
REGION = os.environ.get("AWS_REGION", "us-east-1")
 
 
@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)
 
 
def invoke_endpoint(raw_input: dict) -> dict:
    """Kirim satu record mentah (dict) ke endpoint, sesuai format yang
    diharapkan input_fn/predict_fn di inference.py (dict langsung, BUKAN
    dibungkus {"instances": [...]})."""
    runtime = get_runtime_client()
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(raw_input),
    )
    return json.loads(response["Body"].read().decode("utf-8"))
 
 
st.set_page_config(page_title="Credit Score Predictor", page_icon="💳", layout="centered")
 
st.title("💳 Credit Score Prediction")
st.caption("Prediksi kategori credit score nasabah: Poor / Standard / Good (via SageMaker Endpoint)")
 
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August"]
LOAN_OPTIONS = [
    "Auto Loan", "Credit-Builder Loan", "Personal Loan", "Home Equity Loan",
    "Mortgage Loan", "Student Loan", "Debt Consolidation Loan", "Payday Loan",
]
OCCUPATION_OPTIONS = [
    "Engineer", "Lawyer", "Architect", "Media_Manager", "Accountant",
    "Entrepreneur", "Developer", "Scientist", "Teacher", "Mechanic", "Journalist", "Doctor", "Manager",
    "Musician", "Writer", "Unknown"
]
 
with st.form("credit_form"):
    st.subheader("Data Nasabah")
 
    col1, col2 = st.columns(2)
    with col1:
        month = st.selectbox("Month", MONTHS)
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        occupation = st.selectbox("Occupation", OCCUPATION_OPTIONS)
        annual_income = st.number_input("Annual Income", min_value=0.0, value=50000.0, step=1000.0)
        monthly_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, value=4000.0, step=100.0)
        num_bank_accounts = st.number_input("Num Bank Accounts", min_value=0, max_value=20, value=3)
        num_credit_card = st.number_input("Num Credit Card", min_value=0, max_value=30, value=4)
        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.0)
        num_of_loan = st.number_input("Num of Loan", min_value=0, value=2)
        loan_types = st.multiselect("Type of Loan", LOAN_OPTIONS, default=["Auto Loan"])
        delay_due = st.number_input("Delay from Due Date (hari)", value=5)
        num_delayed_payment = st.number_input("Num of Delayed Payment", min_value=0, value=3)
 
    with col2:
        changed_credit_limit = st.number_input("Changed Credit Limit", value=5.5)
        num_credit_inquiries = st.number_input("Num Credit Inquiries", min_value=0.0, value=2.0)
        credit_mix = st.selectbox("Credit Mix", ["Good", "Standard", "Bad", "Unknown"])
        outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, value=500.0)
        credit_util = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, value=30.0)
        history_years = st.number_input("Credit History (tahun)", min_value=0, value=10)
        history_months = st.number_input("Credit History (bulan tambahan)", min_value=0, max_value=11, value=3)
        payment_min = st.selectbox("Payment of Min Amount", ["Yes", "No", "NM"])
        total_emi = st.number_input("Total EMI per Month", min_value=0.0, value=100.0)
        amount_invested = st.number_input("Amount Invested Monthly", min_value=0.0, value=50.0)
        payment_behaviour = st.selectbox(
            "Payment Behaviour",
            [
                "Low_spent_Small_value_payments", "Low_spent_Medium_value_payments",
                "Low_spent_Large_value_payments", "High_spent_Small_value_payments",
                "High_spent_Medium_value_payments", "High_spent_Large_value_payments",
            ],
        )
        monthly_balance = st.number_input("Monthly Balance", value=300.0)
 
    submitted = st.form_submit_button("🔮 Predict", use_container_width=True)
 
if submitted:
    type_of_loan_str = ", and ".join(loan_types) if loan_types else "Not Specified"
 
    raw_input = {
        "Month": month,
        "Age": age,
        "Occupation": occupation,
        "Annual_Income": str(annual_income),
        "Monthly_Inhand_Salary": monthly_salary,
        "Num_Bank_Accounts": num_bank_accounts,
        "Num_Credit_Card": num_credit_card,
        "Interest_Rate": interest_rate,
        "Num_of_Loan": str(num_of_loan),
        "Type_of_Loan": type_of_loan_str,
        "Delay_from_due_date": delay_due,
        "Num_of_Delayed_Payment": str(num_delayed_payment),
        "Changed_Credit_Limit": str(changed_credit_limit),
        "Num_Credit_Inquiries": num_credit_inquiries,
        "Credit_Mix": credit_mix,
        "Outstanding_Debt": str(outstanding_debt),
        "Credit_Utilization_Ratio": credit_util,
        "Credit_History_Age": f"{history_years} Years and {history_months} Months",
        "Payment_of_Min_Amount": payment_min,
        "Total_EMI_per_month": total_emi,
        "Amount_invested_monthly": str(amount_invested),
        "Payment_Behaviour": payment_behaviour,
        "Monthly_Balance": str(monthly_balance),
    }
 
    try:
        result = invoke_endpoint(raw_input)
    except NoCredentialsError:
        st.error(
            "No AWS credentials found. If running on EC2, attach an instance profile "
            "with sagemaker:InvokeEndpoint permission. If running locally, configure ~/.aws/credentials."
        )
    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")
    else:
        st.divider()
        st.subheader("Hasil Prediksi")
 
        label = result["prediction"]
        color = {"Poor": "🔴", "Standard": "🟡", "Good": "🟢"}.get(label, "⚪")
        st.markdown(f"### {color} **{label}**")
 
        proba_df = pd.DataFrame(
            {"Kelas": list(result["probability"].keys()), "Probabilitas": list(result["probability"].values())}
        ).sort_values("Probabilitas", ascending=False)
        st.bar_chart(proba_df.set_index("Kelas"))
        st.dataframe(proba_df, hide_index=True, use_container_width=True)
 
        with st.expander("Lihat raw JSON response"):
            st.json(result)
