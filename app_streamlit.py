"""
Streamlit UI for the Credit Score classifier hosted on SageMaker.
 
Preprocessing (feature engineering + encode Loan_List) tetap dilakukan lokal,
tapi prediksi FINAL dilakukan lewat SageMaker endpoint (bukan load pipeline.pkl lokal),
mengikuti pola app_streamlit.py dari dosen.
 
boto3 mengambil AWS credentials dari:
  - EC2 instance profile (kalau jalan di EC2 dengan LabInstanceProfile), ATAU
  - ~/.aws/credentials (kalau jalan lokal)
"""
 
import json
import os
 
import boto3
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError
 
st.set_page_config(page_title="Credit Score Predictor", page_icon="💳", layout="centered")
 
st.title("💳 Credit Score Prediction")
st.caption("Prediksi kategori credit score nasabah: Poor / Standard / Good")
 
ARTIFACT_DIR = "model"
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")
 
LABEL_MAP = {0: "Poor", 1: "Standard", 2: "Good"}
ERROR_COLS = [
    "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
    "Outstanding_Debt", "Monthly_Balance", "Amount_invested_monthly",
    "Changed_Credit_Limit",
]
 
 
@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)
 
 
@st.cache_resource
def load_mlb():
    """loan_type_mlb.pkl tetap dimuat LOKAL karena cuma transformer ringan (bukan model
    besar), dipakai untuk menyiapkan fitur sebelum dikirim ke endpoint. Model klasifikasi
    utamanya (credit_score_pipeline.pkl) sekarang tidak di-load di sini lagi -- itu sudah
    di-deploy di SageMaker dan dipanggil lewat invoke_endpoint()."""
    return joblib.load(f"{ARTIFACT_DIR}/loan_type_mlb.pkl")
 
 
try:
    mlb = load_mlb()
except FileNotFoundError as e:
    st.error(
        f"Artifact tidak ditemukan: {e}. Pastikan folder '{ARTIFACT_DIR}/' berisi "
        f"loan_type_mlb.pkl (hasil dari preprocess.py)."
    )
    st.stop()
 
 
def invoke_endpoint(record: dict) -> dict:
    """Kirim satu baris fitur (setelah preprocessing) ke SageMaker endpoint.
 
    NOTE: Berbeda dari contoh Iris/Wine dosen yang kirim list of float murni
    ([sepal_length, sepal_width, ...]), fitur credit score bertipe campuran
    (angka + kategori seperti Occupation/Credit_Mix/Month) dan pakai nama kolom,
    jadi payload dikirim sebagai dict {kolom: nilai}, bukan list posisional.
    Sesuaikan lagi kalau format input_fn di inference script endpoint kamu beda.
    """
    runtime = get_runtime_client()
    payload = {"instances": [record]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))
 
 
def clean_loans(text):
    """Parse string Type_of_Loan jadi list unik. Sama persis dengan preprocess.py."""
    if pd.isna(text):
        return []
    text = text.replace(", and ", ", ").replace(" and ", ", ")
    loans = [loan.strip() for loan in text.split(",")]
    loans = [loan for loan in loans if loan not in ["", "Not Specified"]]
    return list(dict.fromkeys(loans))
 
 
def clean_features(raw: dict) -> pd.DataFrame:
    """Replikasi CreditScorePreprocessor._clean_raw untuk SATU record (tanpa filter baris,
    tanpa kolom Credit_Score, karena ini bukan proses training)."""
    df = pd.DataFrame([raw])
 
    for col in ERROR_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("_", "", regex=False).replace("", np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")
 
    df["Occupation"] = df["Occupation"].replace("_______", "Unknown")
 
    if (df["Interest_Rate"] > 100).any():
        df.loc[df["Interest_Rate"] > 100, "Interest_Rate"] = np.nan
    if (df["Num_of_Loan"] < 0).any():
        df.loc[df["Num_of_Loan"] < 0, "Num_of_Loan"] = np.nan
 
    df["Loan_List"] = df["Type_of_Loan"].apply(clean_loans)
    df["Num_Loan_Types"] = df["Loan_List"].apply(len)
    df["Loan_Info_Missing"] = df["Type_of_Loan"].isna().astype(int)
 
    df["Paid_Early"] = (df["Delay_from_due_date"] < 0).astype(int)
    df["Delay_from_due_date"] = df["Delay_from_due_date"].clip(lower=0)
 
    if (df["Num_of_Delayed_Payment"] < 0).any():
        df.loc[df["Num_of_Delayed_Payment"] < 0, "Num_of_Delayed_Payment"] = np.nan
 
    df["Credit_Mix"] = df["Credit_Mix"].replace("-", "Unknown")
 
    temp = df["Credit_History_Age"].astype(str).str.extract(r"(\d+)\s+Years?\s+and\s+(\d+)\s+Months?")
    df["Credit_History_Months"] = temp[0].astype(float) * 12 + temp[1].astype(float)
    df = df.drop(columns=["Credit_History_Age"])
 
    df["Payment_Behaviour"] = df["Payment_Behaviour"].replace("!@9#%8", "Unknown")
 
    if (df["Monthly_Balance"] < 0).any():
        df.loc[df["Monthly_Balance"] < 0, "Monthly_Balance"] = np.nan
 
    return df
 
 
def encode_loan_types(df: pd.DataFrame, mlb) -> pd.DataFrame:
    """Transform Loan_List pakai MultiLabelBinarizer yang SUDAH fit (loan_type_mlb.pkl)."""
    encoded = pd.DataFrame(
        mlb.transform(df["Loan_List"]),
        columns=mlb.classes_,
        index=df.index,
    )
    out = pd.concat([df, encoded], axis=1)
    return out.drop(columns=["Type_of_Loan", "Loan_List"], errors="ignore")
 
 
def to_json_safe_record(df: pd.DataFrame) -> dict:
    """Convert baris pertama df jadi dict dengan tipe native Python (bukan np.int64/
    np.float64/np.bool_), karena json.dumps tidak bisa serialize tipe numpy langsung."""
    record = df.iloc[0].to_dict()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in record.items()}
 
 
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
        age = st.number_input("Age", min_value=1, max_value=100, value=30)
        occupation = st.selectbox("Occupation", OCCUPATION_OPTIONS)
        annual_income = st.number_input("Annual Income", min_value=0.0, value=50000.0, step=1000.0)
        monthly_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, value=4000.0, step=100.0)
        num_bank_accounts = st.number_input("Num Bank Accounts", min_value=1, max_value=20, value=3)
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
        df = clean_features(raw_input)
        df = encode_loan_types(df, mlb)
        record = to_json_safe_record(df)
 
        response = invoke_endpoint(record)
 
        # NOTE: format response ini mengikuti pola app_streamlit.py dosen
        # (result["labels"][0], result["probabilities"][0]). Sesuaikan key-nya
        # kalau inference script endpoint kamu mengembalikan struktur berbeda.
        label = response["labels"][0]
        probs = response["probabilities"][0]
        proba_dict = {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)}
 
        result = {
            "prediction": label,
            "probability": proba_dict,
        }
    except NoCredentialsError:
        st.error(
            "AWS credentials tidak ditemukan. Kalau jalan di EC2, attach LabInstanceProfile. "
            "Kalau jalan lokal, konfigurasi ~/.aws/credentials."
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
